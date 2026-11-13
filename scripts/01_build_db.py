"""1단계: data/raw의 PDF → Docling JSON/Markdown(수식 LaTeX) → chunks.jsonl, tables.jsonl, figures.jsonl + 그림 PNG.
이미 변환된 PDF(JSON이 있는 경우)는 다시 변환하지 않습니다.
"""
import argparse
import sys
from pathlib import Path

from shiprag.checks import run_checks
from shiprag.convert import convert_pdf, make_converter
from shiprag.figures import render_figures
from shiprag.parse import annotate_chunks, parse_docling_json, to_chunks
from shiprag.utils import load_config, write_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config.yaml")
cfg = load_config(ap.parse_args().config)
p, pc = cfg["paths"], cfg["parse"]

cc = cfg["convert"]
converter = make_converter(cc["do_ocr"], do_formula_enrichment=cc["do_formula_enrichment"],
                           fix_formula_bbox=cc.get("fix_formula_bbox", True),
                           formula_batch_size=cc.get("formula_batch_size"))
all_articles, all_tables, all_figures = [], [], []
for pdf in sorted(Path(p["raw_pdf_dir"]).glob("**/*.pdf")):
    parts = convert_pdf(converter, pdf, p["docling_dir"], cc["batch_pages"])
    articles, tables, figures = parse_docling_json(parts, pdf, pc["article_depth"])
    # 그림은 Docling JSON의 bbox대로 PDF 페이지를 잘라 저장 (벡터 그림도 포함, 재변환 불필요)
    render_figures(figures, pdf, p["figure_dir"], pc.get("figure_scale", 2.0))
    print(f"{pdf.name}: {len(articles)} articles, {len(tables)} tables, {len(figures)} figures")
    all_articles += articles
    all_tables += tables
    all_figures += figures

chunks = annotate_chunks(to_chunks(all_articles), all_tables, all_figures)  # ref_id, refs, formulas, table_ids, figure_ids
write_jsonl(p["chunks"], chunks)
write_jsonl(p["tables"], all_tables)
write_jsonl(p["figures"], all_figures)
print(f"total: {len(all_articles)} articles -> {len(chunks)} chunks, {len(all_tables)} tables, "
      f"{len(all_figures)} figures")

# 고정 검사: 책갈피 대조(조항 누락·오탐) + ref_id 유일성. 실패하면 종료 코드 1 (결과 파일은 확인용으로 남김)
if not run_checks(chunks, sorted(Path(p["raw_pdf_dir"]).glob("**/*.pdf"))):
    sys.exit("[check] 실패: 파서 변경을 확인하세요 (위 목록 참고)")
