"""PDF -> Docling JSON (+ 확인용 Markdown).

- 수식: formula enrichment(CodeFormula 모델)로 LaTeX 변환 (논문 2.2.1, Fig.7)
- 표:  TableFormer로 행/열 구조 인식 → parse 단계에서 HTML로 export (논문 Fig.6)
- 머리글/바닥글: Docling이 furniture 레이어로 분리 → 본문에서 자동 제외

JSON은 계층·페이지 정보를 담고 있어 parse.py가 읽고, Markdown은 사람이 결과를 확인하는 용도입니다.
처음 실행 시 Hugging Face에서 모델을 내려받으며, 수식 모델 때문에 GPU 사용을 권장합니다.

수백 쪽 PDF를 한 번에 변환하면 Colab RAM이 부족해 프로세스가 죽고(출력에 ^C) 결과가 전부 사라지므로,
batch_pages쪽 단위로 변환해 구간마다 저장합니다. 다시 실행하면 저장된 구간은 건너뜁니다.

[수식 영역 보정] Docling layout 후처리(layout_postprocessor._adjust_cluster_bboxes)는 수식 bbox를
그 안의 PDF 텍스트 셀 크기로 줄입니다. DNV PDF는 수식 본체를 그림(이미지 또는 벡터 path)으로 넣는 경우가 많아,
본체가 잘리고 꼬리 텍스트 "for condition A."만 CodeFormula에 들어가 `R \\text{for condition} A`가 됩니다.
Docling 흐름은 그대로 두고, 수식 이미지를 자르기 직전(prepare_element)에 bbox를 보정하는 단계만 끼웁니다.
  1) 후보 수집: 같은 줄 띠 안의 PDF 이미지 객체(get_bitmap_rects)와 페이지 이미지의 잉크 연결 요소
     (잉크 기준 경계: Shah et al. 2021, Mali et al. 2020의 후처리 방식 참고)
  2) 걸러내기: 너무 크거나 띠 밖으로 이어지는 그래픽(실제 그림·다른 줄), 다른 항목의 텍스트 셀과 겹치는 요소.
     인접한 줄의 수식이 합쳐지는 것이 두 선행 연구의 주요 오류였으므로 기준을 보수적으로 잡습니다.
  3) 합치기: 남은 후보와 꼬리 텍스트를 하나의 수식 블록으로 (Hu et al. 2024)
  4) 넓힌 영역을 prov bbox에 기록하고 기존 CodeFormula의 prepare_element로 잘라 넣습니다.
"""
import gc
import re
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
import torch
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.models.base_model import GenericEnrichmentModel
from docling.models.stages.code_formula.code_formula_vlm_model import CodeFormulaVlmModel
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling_core.types.doc import BoundingBox, CoordOrigin, DocItemLabel, TextItem
from scipy import ndimage

# 보정 파라미터 (단위 pt, 1pt = 1/72 inch)
FORMULA_BAND = 80.0      # 꼬리 텍스트 중심 위아래로 이만큼 높이의 띠에서 후보를 찾음
FORMULA_MAX_H = 60.0     # 이보다 높은 그래픽은 그림으로 보고 제외 (F_b 같은 큰 괄호 수식 ~25pt)
FORMULA_MAX_W = 0.8      # 페이지 폭 대비 이보다 넓은 그래픽은 괘선·그림으로 보고 제외
FORMULA_GAP_X = 15.0     # 같은 줄에서 이어 붙일 최대 가로 간격 (본체와 꼬리 텍스트 사이 실측 ~7pt)
FORMULA_STACK_GAP = 1.5  # 분수 분자·분모처럼 위아래로 붙일 최대 세로 간격 (다른 줄은 보통 5pt 이상 떨어짐)
INK_SCALE = 2.0          # 잉크 연결 요소를 찾을 렌더링 배율 (144 dpi)
INK_LEVEL = 160          # 회색조 이 값 미만을 잉크로 봄 (분홍 워터마크·연한 배경 제외)


def _overlap(a0, a1, b0, b1):
    return min(a1, b1) - max(a0, b0)


def _ink_components(band_img, band_top, scale, level=INK_LEVEL):
    """띠 이미지에서 잉크 연결 요소 bbox [(l, t, r, b, touches_edge)] (TOPLEFT pt)."""
    ink = np.asarray(band_img.convert("L")) < level
    labels, _ = ndimage.label(ink, structure=np.ones((3, 3), bool))
    h = ink.shape[0]
    out = []
    for sl in ndimage.find_objects(labels):
        if sl is None:
            continue
        ys, xs = sl
        edge = ys.start == 0 or ys.stop == h
        out.append((xs.start / scale, band_top + ys.start / scale,
                    xs.stop / scale, band_top + ys.stop / scale, edge))
    return out


def expand_formula_bbox(seed, candidates, other_cells, page_w):
    """seed(TOPLEFT pt, (l,t,r,b))를 같은 줄 후보로 넓힌 (l,t,r,b). other_cells는 다른 항목의 텍스트 셀."""
    sl, st, sr, sb = seed
    kept = []
    for l, t, r, b, edge in candidates:
        if edge or b - t > FORMULA_MAX_H or r - l > FORMULA_MAX_W * page_w:
            continue  # 띠 밖으로 이어짐(다른 줄·큰 그림) 또는 지나치게 큰 그래픽
        if any(_overlap(l, r, cl, cr) > 0.5 and _overlap(t, b, ct, cb) > 0.5 for cl, ct, cr, cb in other_cells):
            continue  # 본문 등 다른 항목의 글자
        kept.append((l, t, r, b))

    l, t, r, b = seed
    changed = True
    while changed:
        changed = False
        for c in kept[:]:
            cl, ct, cr, cb = c
            near_x = cl <= r + FORMULA_GAP_X and cr >= l - FORMULA_GAP_X
            same_line = _overlap(ct, cb, st, sb) > 0          # 꼬리 텍스트와 같은 줄
            stacked = (_overlap(cl, cr, l, r) > 0              # 분수처럼 지금 영역의 바로 위·아래
                       and max(ct - b, t - cb) <= FORMULA_STACK_GAP)
            if near_x and (same_line or stacked) and max(b, cb) - min(t, ct) <= FORMULA_MAX_H:
                l, t, r, b = min(l, cl), min(t, ct), max(r, cr), max(b, cb)
                kept.remove(c)
                changed = True
    return l, t, r, b


LOOP_MIN_CHARS = 2000    # 생성 한도(2048토큰)까지 간 긴 출력만 검사. 정상 수식은 기존 chunk에서 최대 ~800자이고,
                         # Hog/Sag처럼 구조가 되풀이되는 정상 수식 목록(463·805자)을 루프로 오인하지 않게 함
LOOP_NGRAM = 24          # 이 길이(토큰)의 같은 구간이
LOOP_MIN_REPEATS = 3     # 이만큼 반복되고
LOOP_TAIL_RATIO = 0.75   # 마지막 반복이 출력 끝 25% 안에 있으면 생성 루프로 봄
_SPACING = {"\\quad", "\\qquad", "\\,", "\\;", "\\:", "\\!", "\\", "~", "\\\\"}
_OPEN, _CLOSE = {"{", "\\left"}, {"}", "\\right"}


def _drop_dangling(toks):
    """잘린 끝에서 닫히지 않은 그룹, 인자가 모자란 명령, 끝의 공백 명령을 걷어냄."""
    depth, last_zero = 0, 0
    for i, tok in enumerate(toks):
        depth += (tok in _OPEN) - (tok in _CLOSE)
        if depth == 0:
            last_zero = i + 1
    if last_zero < len(toks):  # 닫히지 않은 그룹이 있음 → 그 그룹을 인자로 받던 명령까지 제거
        toks = toks[:last_zero]
        while toks and toks[-1] == "}":  # 앞선 인자 그룹 { ... }
            d = 0
            for j in range(len(toks) - 1, -1, -1):
                d += (toks[j] == "}") - (toks[j] == "{")
                if d == 0:
                    toks = toks[:j]
                    break
            else:
                break
        if toks and toks[-1].startswith("\\") and toks[-1] not in _CLOSE and toks[-1] not in _SPACING:
            toks = toks[:-1]
    while toks and toks[-1] in _SPACING:
        toks = toks[:-1]
    return toks


def strip_formula_loop(latex):
    """CodeFormula가 정상 수식 뒤에 같은 내용을 반복 생성한 꼬리를 잘라냄.

    Docling의 TailRepetitionStopper는 64자 이하 단위가 16번 이상 *똑같이* 반복될 때만 잡는데,
    DNV F_b 수식(Pt.3 p.66)은 정상 수식 끝부분(~400자)을 조금씩 바꿔 가며 반복해 걸리지 않습니다.
    정상 끝부분 R 뒤에 쓰레기 C+J_k가 붙는 [R C J_1][R C J_2]... 형태이므로,
      1) 출력 끝까지 3번 이상 반복되는 24토큰 구간의 첫 위치 p0를 찾고
      2) 모든 반복과 공통인 범위(R+C)까지 자른 뒤
      3) C처럼 닫히지 않은 그룹·인자가 모자란 명령·끝 공백 명령을 걷어내 R의 끝으로 되돌립니다.
    반복이 없으면 원문을 그대로 돌려줍니다."""
    if len(latex) < LOOP_MIN_CHARS:
        return latex
    spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", latex)]
    toks = [latex[s:e] for s, e in spans]
    n = LOOP_NGRAM
    if len(toks) < n * LOOP_MIN_REPEATS:
        return latex
    seen = {}
    for i in range(len(toks) - n + 1):
        seen.setdefault(tuple(toks[i:i + n]), []).append(i)
    loops = [pos for pos in seen.values()
             if len(pos) >= LOOP_MIN_REPEATS and pos[-1] >= LOOP_TAIL_RATIO * len(toks)]
    if not loops:
        return latex
    pos = min(loops, key=lambda p: p[0])
    p0 = pos[0]
    common = pos[1] - p0  # 첫 반복 구간은 다음 반복 시작을 넘지 않음
    for p in pos[1:]:
        k = 0
        while k < common and p + k < len(toks) and toks[p0 + k] == toks[p + k]:
            k += 1
        common = min(common, k)
    kept = _drop_dangling(toks[:p0 + common])
    return latex[:spans[len(kept) - 1][1]] if kept else latex


class FormulaBBoxFix(GenericEnrichmentModel):
    """CodeFormula 모델을 감싸, 수식 이미지를 자르기 전에 prov bbox를 보정합니다.
    모델을 다시 로드하지 않도록 기존 인스턴스에 그대로 위임합니다."""

    def __init__(self, inner):
        self.inner = inner
        self.elements_batch_size = inner.elements_batch_size
        self.progress_step = inner.progress_step

    def is_processable(self, doc, element):
        return self.inner.is_processable(doc, element)

    def __call__(self, doc, element_batch):
        for item in self.inner(doc, element_batch):
            if isinstance(item, TextItem) and item.label == DocItemLabel.FORMULA and item.text:
                item.text = strip_formula_loop(item.text)
            yield item

    def prepare_element(self, conv_res, element):
        if (self.inner.is_processable(conv_res.document, element) and isinstance(element, TextItem)
                and element.label == DocItemLabel.FORMULA and element.prov and conv_res.pages):
            prov = element.prov[0]
            page = conv_res.pages[prov.page_no - conv_res.pages[0].page_no]
            if page._backend is not None and page.size is not None:
                prov.bbox = self._fix(page, prov.bbox)
        return self.inner.prepare_element(conv_res, element)

    @staticmethod
    def _fix(page, bbox):
        page_w, page_h = page.size.width, page.size.height
        seed_bb = bbox.to_top_left_origin(page_h)
        seed = (seed_bb.l, seed_bb.t, seed_bb.r, seed_bb.b)

        # 1) 후보: 같은 줄 띠 안의 이미지 객체 + 잉크 연결 요소
        cy = (seed[1] + seed[3]) / 2
        top, bottom = max(0.0, cy - FORMULA_BAND / 2), min(page_h, cy + FORMULA_BAND / 2)
        band = BoundingBox(l=0, t=top, r=page_w, b=bottom, coord_origin=CoordOrigin.TOPLEFT)
        img = page._backend.get_page_image(scale=INK_SCALE, cropbox=band)
        candidates = _ink_components(img, top, INK_SCALE)
        for bm in page._backend.get_bitmap_rects():
            bm = bm.to_top_left_origin(page_h)
            if _overlap(bm.t, bm.b, top, bottom) > 0:
                candidates.append((bm.l, bm.t, bm.r, bm.b, bm.t < top or bm.b > bottom))

        # 2) 다른 항목의 텍스트 셀 = 수식 bbox 안에 절반 이상 들어오지 않는 셀
        other = []
        for cell in page._backend.get_text_cells():
            cb = cell.rect.to_bounding_box().to_top_left_origin(page_h)
            if not cell.text.strip() or _overlap(cb.t, cb.b, top, bottom) <= 0:
                continue
            if cb.intersection_over_self(seed_bb) < 0.5:
                other.append((cb.l, cb.t, cb.r, cb.b))

        # 3) 합치기
        l, t, r, b = expand_formula_bbox(seed, candidates, other, page_w)
        fixed = BoundingBox(l=l, t=t, r=r, b=b, coord_origin=CoordOrigin.TOPLEFT)
        return fixed.to_bottom_left_origin(page_h) if bbox.coord_origin == CoordOrigin.BOTTOMLEFT else fixed


class ShipRagPdfPipeline(StandardPdfPipeline):
    """StandardPdfPipeline + 수식 bbox 보정(FormulaBBoxFix)."""

    def _init_models(self):
        super()._init_models()
        self.enrichment_pipe = [FormulaBBoxFix(m) if isinstance(m, CodeFormulaVlmModel) else m
                                for m in self.enrichment_pipe]


def make_converter(do_ocr=False, do_table_structure=True, do_formula_enrichment=True,
                   fix_formula_bbox=True, page_images_scale=None):
    """page_images_scale을 주면 페이지 이미지를 문서에 남겨 item.get_image(doc)로 잘린 영역을 볼 수 있음 (진단용)."""
    # DNV PDF는 모든 페이지에 텍스트 레이어가 있어 OCR(Docling 기본값 True)은 메모리·시간 낭비
    opts = PdfPipelineOptions(do_ocr=do_ocr, do_table_structure=do_table_structure,
                              do_formula_enrichment=do_formula_enrichment)
    if page_images_scale:
        opts.generate_page_images = True
        opts.images_scale = page_images_scale
    if fix_formula_bbox:
        fmt = PdfFormatOption(pipeline_options=opts, pipeline_cls=ShipRagPdfPipeline)
    else:
        fmt = PdfFormatOption(pipeline_options=opts)
    return DocumentConverter(format_options={InputFormat.PDF: fmt})


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
