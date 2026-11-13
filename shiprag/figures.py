"""그림 이미지 저장: Docling picture bbox대로 PDF 페이지를 렌더링해 잘라 PNG로 저장.

DNV 그림은 벡터(path)로 그린 것과 비트맵 이미지 객체가 섞여 있어(전체 497개 중 벡터 283개),
PDF에 붙은 이미지 객체를 꺼내는 방식으로는 절반 넘게 빠짐 → 페이지 렌더링 후 잘라냄 (Docling get_image()와 같은 방식).
Docling JSON에는 bbox만 있으면 되므로 재변환이 필요 없음.
"""
from collections import defaultdict
from pathlib import Path

import pypdfium2 as pdfium

PAD_PT = 4  # 그림 가장자리 선이 잘리지 않도록 사방 여백 (pt)


def render_figures(figures, pdf_path, out_dir, scale=2.0):
    """figures(parse_docling_json 반환값 중 한 PDF 것)의 bbox 영역을 out_dir/<file>로 저장. 저장한 개수 반환."""
    out_dir = Path(out_dir)
    pdf = pdfium.PdfDocument(str(pdf_path))
    n = 0
    try:
        for f in figures:
            page = pdf[f["page"] - 1]
            w, h = page.get_size()
            l, t, r, b = f["bbox"]
            l, t = max(0.0, l - PAD_PT), max(0.0, t - PAD_PT)
            r, b = min(w, r + PAD_PT), min(h, b + PAD_PT)
            # pdfium crop = 왼쪽·아래·오른쪽·위에서 잘라낼 양 (pt)
            img = page.render(scale=scale, crop=(l, h - b, w - r, t)).to_pil()
            path = out_dir / f["file"]
            path.parent.mkdir(parents=True, exist_ok=True)
            img.save(path, optimize=True)
            page.close()
            n += 1
    finally:
        pdf.close()
    return n


def build_figure_index(figures):
    """그림 레코드 목록 → parse.chunk_figures가 쓰는 조회 구조.
    lookup 값은 리스트: 그림 하나가 picture 여러 개(좌우 조각, 다음 페이지로 이어진 부분)로 나뉘어
    같은 'Figure N' 캡션을 공유하는 경우가 있음 (예: Pt.3 Ch.1 Sec.4 Figure 6, p.40-41)."""
    lookup, by_article = defaultdict(list), defaultdict(list)
    for f in figures:
        lookup[(f["doc"], f["section"], f["index"])].append(f)
        by_article[(f["doc"], f["section"], f["article_no"])].append(f)
    return {"lookup": dict(lookup), "by_article": dict(by_article)}
