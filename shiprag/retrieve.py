"""검색 모듈 (논문 2.2.2): bi-encoder로 Content를 미리 임베딩하고, query와 dot product로 Top-k 선택.
[논문과 다른 점] embed_tables가 켜져 있으면 Content 뒤에 참조된 표의 평문을 붙여 임베딩 (parse.embed_text)."""
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from .parse import chunk_texts
from .utils import read_jsonl


def build_index(model_name, chunks_path, index_dir, batch_size=64, max_seq_length=512, table_lookup=None):
    model = SentenceTransformer(model_name)
    model.max_seq_length = max_seq_length
    chunks = read_jsonl(chunks_path)
    emb = model.encode(chunk_texts(chunks, table_lookup), batch_size=batch_size,
                       convert_to_numpy=True, show_progress_bar=True)
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / "embeddings.npy", emb.astype(np.float32))
    (index_dir / "meta.json").write_text(json.dumps(
        {"model": str(model_name), "chunks": str(chunks_path), "max_seq_length": max_seq_length,
         "n_chunks": len(chunks), "embed_tables": table_lookup is not None}, indent=2), encoding="utf-8")
    return emb


def load_index(index_dir, n_chunks, embed_tables=None):
    """(meta, embeddings) 반환. 01을 다시 돌려 chunks.jsonl이 바뀌었는데 인덱스가 예전 것이면 에러.
    embed_tables를 주면 인덱스가 같은 텍스트(표 포함 여부)로 만들어졌는지도 검사."""
    index_dir = Path(index_dir)
    meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
    emb = np.load(index_dir / "embeddings.npy")
    if len(emb) != n_chunks:
        raise ValueError(f"{index_dir}: 임베딩 {len(emb)}개 != chunk {n_chunks}개. "
                         f"chunks.jsonl이 바뀌었으니 02_build_index.py를 다시 실행하세요.")
    if embed_tables is not None and meta.get("embed_tables", False) != embed_tables:
        raise ValueError(f"{index_dir}: 인덱스는 embed_tables={meta.get('embed_tables', False)}로 만들어졌는데 "
                         f"config는 {embed_tables}입니다. 02_build_index.py를 다시 실행하세요.")
    return meta, emb


class Retriever:
    def __init__(self, index_dir):
        meta = json.loads((Path(index_dir) / "meta.json").read_text(encoding="utf-8"))
        self.chunks = read_jsonl(meta["chunks"])
        _, self.emb = load_index(index_dir, len(self.chunks))
        self.model = SentenceTransformer(meta["model"])
        self.model.max_seq_length = meta["max_seq_length"]

    def search(self, query, k=5):
        q = self.model.encode(query, convert_to_numpy=True)
        scores = self.emb @ q
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        top = top[np.argsort(-scores[top])]
        return [{**self.chunks[i], "score": float(scores[i]), "rank": r + 1} for r, i in enumerate(top)]
