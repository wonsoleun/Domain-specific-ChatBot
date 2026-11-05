"""2단계: chunk Content를 임베딩해 인덱스 생성.
  --model base     : config의 base_model (GPL 전에 필요)
  --model adapted  : 4단계에서 학습한 모델
"""
import argparse

from shiprag.retrieve import build_index
from shiprag.utils import load_config, load_table_lookup

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config.yaml")
ap.add_argument("--model", choices=["base", "adapted"], default="base")
args = ap.parse_args()
cfg = load_config(args.config)
p, r = cfg["paths"], cfg["retriever"]

model = r["base_model"] if args.model == "base" else p["model_out"]
index_dir = p["index_base"] if args.model == "base" else p["index_adapted"]
table_lookup = load_table_lookup(p["tables"]) if r.get("embed_tables") else None
emb = build_index(model, p["chunks"], index_dir, r["batch_size"], r["max_seq_length"], table_lookup)
print(f"saved {emb.shape} embeddings to {index_dir} (embed_tables={table_lookup is not None})")
