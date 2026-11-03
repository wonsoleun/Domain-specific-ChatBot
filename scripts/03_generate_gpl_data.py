"""3단계: GPL 학습 데이터 생성 (OPENAI_API_KEY 필요, base 인덱스 필요).
  --step queries : GPT-4o로 질문 생성 (중단 후 재실행하면 이어서 진행)
  --step label   : negative mining + cross-encoder 라벨링
"""
import argparse

from shiprag.gpl import mine_and_label, synthesize_queries
from shiprag.utils import load_config, read_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config.yaml")
ap.add_argument("--step", choices=["queries", "label", "all"], default="all")
args = ap.parse_args()
cfg = load_config(args.config)
p, g = cfg["paths"], cfg["gpl"]
chunks = read_jsonl(p["chunks"])

if args.step in ("queries", "all"):
    synthesize_queries(chunks, p["queries"], g["openai_model"], g["n_contents"],
                       g["queries_per_content"], g["workers"], g["seed"])
if args.step in ("label", "all"):
    n = mine_and_label(chunks, p["queries"], p["index_base"], p["triplets"], g["cross_encoder"],
                       g["negatives_per_query"], g["bottom_ratio"], g["seed"])
    print(f"saved {n} triplets to {p['triplets']}")
