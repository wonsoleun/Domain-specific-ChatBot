"""PDF -> Docling JSON (+ 확인용 Markdown).

- 수식: formula enrichment(CodeFormula 모델)로 LaTeX 변환 (논문 2.2.1, Fig.7)
- 표:  TableFormer로 행/열 구조 인식 → parse 단계에서 HTML로 export (논문 Fig.6)
- 머리글/바닥글: Docling이 furniture 레이어로 분리 → 본문에서 자동 제외

JSON은 계층·페이지 정보를 담고 있어 parse.py가 읽고, Markdown은 사람이 결과를 확인하는 용도입니다.
처음 실행 시 Hugging Face에서 모델을 내려받으며, 수식 모델 때문에 GPU 사용을 권장합니다.

수백 쪽 PDF를 한 번에 변환하면 Colab RAM이 부족해 프로세스가 죽고(출력에 ^C) 결과가 전부 사라지므로,
batch_pages쪽 단위로 변환해 구간마다 저장합니다. 다시 실행하면 저장된 구간은 건너뜁니다.
"""
import gc
from pathlib import Path

import pypdfium2 as pdfium
import torch
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


def make_converter(do_ocr=False, do_table_structure=True, do_formula_enrichment=True):
    # DNV PDF는 모든 페이지에 텍스트 레이어가 있어 OCR(Docling 기본값 True)은 메모리·시간 낭비
    opts = PdfPipelineOptions(do_ocr=do_ocr, do_table_structure=do_table_structure,
                              do_formula_enrichment=do_formula_enrichment)
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})


def convert_pdf(converter, pdf_path, out_dir, batch_pages=50):
    """out_dir/<stem>/p0001-0050.json(.md) 형태로 구간별 저장. [(json_path, start_page)]를 페이지 순서로 반환."""
    pdf_path = Path(pdf_path)
    part_dir = Path(out_dir) / pdf_path.stem
    part_dir.mkdir(parents=True, exist_ok=True)
    pdf = pdfium.PdfDocument(str(pdf_path))
    n_pages = len(pdf)
    pdf.close()

    parts = []
    for start in range(1, n_pages + 1, batch_pages):
        end = min(start + batch_pages - 1, n_pages)
        json_path = part_dir / f"p{start:04d}-{end:04d}.json"
        if not json_path.exists():
            print(f"  converting {pdf_path.name} p.{start}-{end} / {n_pages}", flush=True)
            doc = converter.convert(pdf_path, page_range=(start, end)).document
            doc.save_as_markdown(json_path.with_suffix(".md"))
            doc.save_as_json(json_path)  # 마지막에 저장 → 중간에 끊기면 이 구간만 다시 변환
            del doc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        parts.append((json_path, start))
    return parts
