"""검색 모듈 (논문 2.2.2): bi-encoder로 Content를 미리 임베딩하고, query와 dot product로 Top-k 선택."""
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from .utils import read_jsonl


def build_index(model_name, chunks_path, index_dir, batch_size=64, max_seq_length=512):
    model = SentenceTransformer(model_name)
    model.max_seq_length = max_seq_length
    chunks = read_jsonl(chunks_path)
    emb = model.encode([c["content"] for c in chunks], batch_size=batch_size,
                       convert_to_numpy=True, show_progress_bar=True)
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / "embeddings.npy", emb.astype(np.float32))
    (index_dir / "meta.json").write_text(json.dumps(
        {"model": str(model_name), "chunks": str(chunks_path), "max_seq_length": max_seq_length}, indent=2))
    return emb


class Retriever:
    def __init__(self, index_dir):
        index_dir = Path(index_dir)
        meta = json.loads((index_dir / "meta.json").read_text())
        self.model = SentenceTransformer(meta["model"])
        self.model.max_seq_length = meta["max_seq_length"]
        self.emb = np.load(index_dir / "embeddings.npy")
        self.chunks = read_jsonl(meta["chunks"])

    def search(self, query, k=5):
        q = self.model.encode(query, convert_to_numpy=True)
        scores = self.emb @ q
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        top = top[np.argsort(-scores[top])]
        return [{**self.chunks[i], "score": float(scores[i]), "rank": r + 1} for r, i in enumerate(top)]
