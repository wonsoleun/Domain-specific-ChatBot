"""GPL 기반 학습 데이터 생성 (논문 3.2, Fig.12).

1) Query synthesis   : GPT-4o로 Content당 질문 3개 생성
2) Similarity eval   : cross-encoder로 (query, content) 점수
3) Negative mining   : 유사도 하위 30% 구간에서 50개 랜덤 샘플링
4) Data labeling     : margin = CE(q, pos) - CE(q, neg)

[논문과 다른 점] 논문은 생성된 query와 DB 전체 Content를 cross-encoder로 채점한 뒤
하위 30%를 고릅니다. 이는 query 6만 개 × chunk 15만 개 = 90억 쌍이라 Colab에서 불가능하므로,
하위 30% 구간은 base bi-encoder 점수로 정하고(행렬곱 한 번), cross-encoder는 실제 학습에 쓰이는
(q, pos), (q, neg) 쌍의 라벨링에만 사용합니다.
"""
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .retrieve import load_index
from .utils import read_jsonl

QUERY_INSTRUCTIONS = """1. Based on the following rule of ship regulations, generate {n} different questions.
2. Generate questions users are likely to ask about ship regulations.
3. Each question should cover a different aspect of the rule, providing diverse perspectives to understand it better.
Return JSON: {{"questions": ["...", "..."]}}"""


def synthesize_queries(chunks, out_path, model="gpt-4o", n_contents=20000, per_content=3,
                       workers=8, seed=42):
    """chunk를 n_contents개 샘플링해 질문 생성. 중단 후 재실행하면 이어서 진행."""
    from openai import OpenAI
    client = OpenAI()  # OPENAI_API_KEY 환경변수 사용

    rng = random.Random(seed)
    sample = rng.sample(chunks, min(n_contents, len(chunks)))

    out_path = Path(out_path)
    done = {r["chunk_id"] for r in read_jsonl(out_path)} if out_path.exists() else set()
    todo = [c for c in sample if c["chunk_id"] not in done]
    lock = threading.Lock()

    def work(chunk):
        # 한 chunk 실패(rate limit, 잘못된 JSON 등)로 전체가 멈추지 않도록 건너뜀 → 재실행 시 다시 시도
        try:
            resp = client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": QUERY_INSTRUCTIONS.format(n=per_content)},
                          {"role": "user", "content": f"Rule:\n{chunk['content']}"}],
            )
            qs = json.loads(resp.choices[0].message.content).get("questions", [])
        except Exception as e:
            tqdm.write(f"chunk {chunk['chunk_id']} 실패: {e!r}")
            return 0
        qs = [q.strip() for q in qs if isinstance(q, str) and q.strip()][:per_content]
        with lock, open(out_path, "a", encoding="utf-8") as f:
            for q in qs:
                # doc/node는 01 재실행으로 chunk_id가 바뀌었는지 검사하는 용도
                f.write(json.dumps({"chunk_id": chunk["chunk_id"], "doc": chunk["doc"], "node": chunk["node"],
                                    "query": q}, ensure_ascii=False) + "\n")
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(workers) as ex:
        ok = sum(tqdm(ex.map(work, todo), total=len(todo), desc="query synthesis"))
    if ok < len(todo):
        print(f"{len(todo) - ok}개 chunk 실패 — 다시 실행하면 실패한 것만 이어서 생성합니다.")


@torch.no_grad()
def _cross_encoder_scores(model, tok, pairs, batch_size=256, device="cuda"):
    out = []
    for i in tqdm(range(0, len(pairs), batch_size), desc="cross-encoder"):
        q, p = zip(*pairs[i:i + batch_size])
        enc = tok(list(q), list(p), truncation=True, max_length=512, padding=True, return_tensors="pt").to(device)
        with torch.autocast(device_type=device, enabled=device == "cuda"):
            logits = model(**enc).logits.squeeze(-1)
        out.append(logits.float().cpu())
    return torch.cat(out).numpy()


def mine_and_label(chunks, queries_path, base_index_dir, out_path, cross_encoder,
                   negatives_per_query=50, bottom_ratio=0.3, seed=42):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(seed)
    queries = read_jsonl(queries_path)
    stale = [q for q in queries if q["chunk_id"] >= len(chunks) or
             ("node" in q and (chunks[q["chunk_id"]]["doc"], chunks[q["chunk_id"]]["node"]) != (q["doc"], q["node"]))]
    if stale:
        raise ValueError(f"queries.jsonl의 {len(stale)}개 질문이 현재 chunks.jsonl과 맞지 않습니다 "
                         f"(01을 다시 실행해 chunk_id가 바뀜). queries.jsonl을 지우고 --step queries부터 다시 실행하세요.")

    meta, corpus = load_index(base_index_dir, len(chunks))
    corpus = torch.from_numpy(corpus).to(device)
    bi = SentenceTransformer(meta["model"], device=device)
    q_emb = bi.encode([q["query"] for q in queries], convert_to_tensor=True, show_progress_bar=True).to(device)

    article_of = [(c["doc"], c["section"], c["article_no"]) for c in chunks]
    n_bottom = min(len(chunks), max(negatives_per_query, int(len(chunks) * bottom_ratio)))

    # 3) Negative mining: 하위 30% 구간에서 랜덤 샘플 (같은 article의 chunk는 제외 — 정답을 포함할 수 있음)
    pos_ids, neg_ids = [], []
    for start in tqdm(range(0, len(queries), 256), desc="negative mining"):
        scores = q_emb[start:start + 256] @ corpus.T
        bottom = torch.topk(scores, n_bottom, dim=1, largest=False).indices.cpu().numpy()
        for row, q in zip(bottom, queries[start:start + 256]):
            pos = q["chunk_id"]
            cands = [i for i in row if article_of[i] != article_of[pos]]
            picked = rng.choice(cands, size=min(negatives_per_query, len(cands)), replace=False)
            pos_ids.append(pos)
            neg_ids.append(picked)

    # 2) + 4) cross-encoder 점수 -> margin 라벨
    tok = AutoTokenizer.from_pretrained(cross_encoder)
    ce = AutoModelForSequenceClassification.from_pretrained(cross_encoder).to(device).eval()
    texts = [c["content"] for c in chunks]
    pos_scores = _cross_encoder_scores(ce, tok, [(q["query"], texts[p]) for q, p in zip(queries, pos_ids)], device=device)
    q_idx = np.repeat(np.arange(len(queries)), [len(n) for n in neg_ids])
    neg_flat = np.concatenate(neg_ids).astype(np.int64)
    neg_scores = _cross_encoder_scores(ce, tok, [(queries[qi]["query"], texts[n]) for qi, n in zip(q_idx, neg_flat)], device=device)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, query=q_idx, pos=np.asarray(pos_ids)[q_idx], neg=neg_flat,
             margin=(pos_scores[q_idx] - neg_scores).astype(np.float32))
    return len(q_idx)
