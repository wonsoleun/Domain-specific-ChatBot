"""5단계: 질의 → 검색 → (선택) LLaMA 답변 생성.
  python scripts/05_ask.py --query "..."                 # adapted 인덱스 + 답변 생성
  python scripts/05_ask.py --query "..." --index base --no-generate
"""
import argparse

from shiprag.retrieve import Retriever
from shiprag.utils import load_config, load_table_lookup

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config.yaml")
ap.add_argument("--query", required=True)
ap.add_argument("--index", choices=["base", "adapted"], default="adapted")
ap.add_argument("--no-generate", action="store_true")
args = ap.parse_args()
cfg = load_config(args.config)
p = cfg["paths"]

retriever = Retriever(p["index_adapted"] if args.index == "adapted" else p["index_base"])
hits = retriever.search(args.query, k=cfg["retriever"]["top_k"])
print("=== Related rules ===")
for h in hits:
    print(f"[{h['rank']}] {h['score']:.2f}  {h['title']}  (node {h['node']}, p.{h['page']})")

if not args.no_generate:
    from shiprag.generate import Generator
    g = cfg["generator"]
    gen = Generator(g["model"], g["load_in_4bit"], g["max_new_tokens"])
    print("\n=== Answer ===")
    print(gen.answer(args.query, hits, load_table_lookup(p["tables"])))
