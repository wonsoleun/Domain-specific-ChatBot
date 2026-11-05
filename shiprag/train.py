"""Domain adaptation (논문 3.1, 4.2): bi-encoder를 MarginMSE로 contrastive fine-tuning.

loss = mean( (pred_margin - true_margin)^2 ),  pred_margin = q·p - q·n   (식 1~3)
"""
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .parse import chunk_texts
from .utils import read_jsonl


def _split_by_group(group, val_ratio, seed):
    """group 값이 같은 샘플은 train/val 한쪽에만 들어가도록 분리."""
    rng = np.random.default_rng(seed)
    ug = np.unique(group)
    val_g = rng.choice(ug, size=max(1, int(len(ug) * val_ratio)), replace=False)
    is_val = np.isin(group, val_g)
    return np.where(~is_val)[0], np.where(is_val)[0]


def train(base_model, chunks_path, queries_path, triplets_path, out_dir,
          batch_size=64, lr=1e-5, max_steps=30000, eval_every=1000, patience=5,
          val_ratio=0.01, max_seq_length=256, fp16=True, seed=42, table_lookup=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    model = SentenceTransformer(base_model, device=device)
    model.max_seq_length = max_seq_length

    texts = chunk_texts(read_jsonl(chunks_path), table_lookup)  # 02 인덱스와 같은 텍스트로 학습
    queries = [q["query"] for q in read_jsonl(queries_path)]
    t = np.load(triplets_path)
    q_idx, pos, neg, margin = t["query"], t["pos"], t["neg"], t["margin"]
    # positive chunk 단위로 분리: 같은 chunk에서 나온 질문 3개가 train/val에 나뉘면 val loss가 낙관적으로 나옴
    train_ids, val_ids = _split_by_group(pos, val_ratio, seed)

    def embed(batch_texts):
        feats = model.tokenize(batch_texts)
        feats = {k: v.to(device) if torch.is_tensor(v) else v for k, v in feats.items()}
        return model(feats)["sentence_embedding"]

    def batch_loss(ids):
        q = embed([queries[i] for i in q_idx[ids]])
        p = embed([texts[i] for i in pos[ids]])
        n = embed([texts[i] for i in neg[ids]])
        pred = (q * p).sum(-1) - (q * n).sum(-1)
        target = torch.tensor(margin[ids], device=device)
        return torch.nn.functional.mse_loss(pred.float(), target)

    @torch.no_grad()
    def evaluate():
        model.eval()
        losses = []
        for i in range(0, len(val_ids), batch_size):
            with torch.autocast(device_type=device, enabled=fp16 and device == "cuda"):
                losses.append(batch_loss(val_ids[i:i + batch_size]).item())
        model.train()
        return float(np.mean(losses))

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(enabled=fp16 and device == "cuda")
    rng = np.random.default_rng(seed)
    order = rng.permutation(train_ids)
    best, bad, cursor = float("inf"), 0, 0

    model.train()
    for step in tqdm(range(1, max_steps + 1), desc="train"):
        if cursor + batch_size > len(order):
            order, cursor = rng.permutation(train_ids), 0
        ids = order[cursor:cursor + batch_size]
        cursor += batch_size

        with torch.autocast(device_type=device, enabled=fp16 and device == "cuda"):
            loss = batch_loss(ids)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        if step % eval_every == 0:
            val = evaluate()
            print(f"step {step}  train_loss {loss.item():.4f}  val_loss {val:.4f}")
            if val < best:
                best, bad = val, 0
                model.save(str(Path(out_dir)))
            else:
                bad += 1
                if bad >= patience:
                    print(f"early stopping at step {step} (best val_loss {best:.4f})")
                    break
    if best == float("inf"):  # max_steps < eval_every인 경우
        best = evaluate()
        model.save(str(Path(out_dir)))
    return best
