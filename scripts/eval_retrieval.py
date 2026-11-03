"""검색 정확도 비교 (논문 식 4~5). 테스트셋은 직접 작성한 JSONL:
  {"query": "...", "doc": "DNV-RU-SHIP-Pt2", "section": "Pt.2 Ch.1 Section 3 TESTING PROCEDURES", "article": "3.1"}
  doc / section / article 값은 chunks.jsonl의 doc / section / article_no를 그대로 복사하세요.
"""
import argparse

from shiprag.evaluate import evaluate
from shiprag.retrieve import Retriever
from shiprag.utils import load_config, read_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config.yaml")
ap.add_argument("--testset", required=True)
args = ap.parse_args()
cfg = load_config(args.config)
testset = read_jsonl(args.testset)

for name in ["base", "adapted"]:
    res = evaluate(Retriever(cfg["paths"][f"index_{name}"]), testset)
    print(f"{name:8s} accuracy {res['accuracy']:.2f}  recall@1 {res['recall@1']:.2f}  "
          f"recall@5 {res['recall@5']:.2f}")
