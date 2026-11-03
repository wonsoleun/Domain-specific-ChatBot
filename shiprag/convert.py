"""PDF -> Docling JSON (+ 확인용 Markdown).

- 수식: formula enrichment(CodeFormula 모델)로 LaTeX 변환 (논문 2.2.1, Fig.7)
- 표:  TableFormer로 행/열 구조 인식 → parse 단계에서 HTML로 export (논문 Fig.6)
- 머리글/바닥글: Docling이 furniture 레이어로 분리 → 본문에서 자동 제외

JSON은 계층·페이지 정보를 담고 있어 parse.py가 읽고, Markdown은 사람이 결과를 확인하는 용도입니다.
처음 실행 시 Hugging Face에서 모델을 내려받으며, 수식 모델 때문에 GPU 사용을 권장합니다.
"""
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


def make_converter():
    opts = PdfPipelineOptions(do_table_structure=True, do_formula_enrichment=True)
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})


def convert_pdf(converter, pdf_path, out_dir):
    """변환 결과를 out_dir/<stem>.json, .md로 저장. 이미 JSON이 있으면 건너뜀 (Colab 재실행 대비)."""
    out_dir = Path(out_dir)
    json_path = out_dir / f"{Path(pdf_path).stem}.json"
    if json_path.exists():
        return json_path
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = converter.convert(pdf_path).document
    doc.save_as_markdown(out_dir / f"{Path(pdf_path).stem}.md")
    doc.save_as_json(json_path)  # 마지막에 저장 → 중간에 끊기면 다음 실행 때 다시 변환
    return json_path
