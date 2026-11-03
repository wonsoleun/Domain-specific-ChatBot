"""1단계: data/raw의 PDF → Docling JSON/Markdown(수식 LaTeX) → chunks.jsonl, tables.jsonl.
이미 변환된 PDF(JSON이 있는 경우)는 다시 변환하지 않습니다.
"""
import argparse
from pathlib import Path

from shiprag.convert import convert_pdf, make_converter
from shiprag.parse import parse_docling_json, to_chunks
from shiprag.utils import load_config, write_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config.yaml")
cfg = load_config(ap.parse_args().config)
p, pc = cfg["paths"], cfg["parse"]

converter = make_converter()
all_articles, all_tables = [], []
for pdf in sorted(Path(p["raw_pdf_dir"]).glob("**/*.pdf")):
    json_path = convert_pdf(converter, pdf, p["docling_dir"])
    articles, tables = parse_docling_json(json_path, pdf, pc["article_depth"])
    print(f"{pdf.name}: {len(articles)} articles, {len(tables)} tables")
    all_articles += articles
    all_tables += tables

chunks = to_chunks(all_articles)
write_jsonl(p["chunks"], chunks)
write_jsonl(p["tables"], all_tables)
print(f"total: {len(all_articles)} articles -> {len(chunks)} chunks, {len(all_tables)} tables")
