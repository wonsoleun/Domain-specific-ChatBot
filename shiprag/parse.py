"""Docling JSON -> 계층형 nested chunks + 표(HTML).

입력은 convert.py가 만든 Docling JSON입니다 (수식은 LaTeX, 표는 셀 구조 포함, 머리글/바닥글 제외).

논문 2.2.1 재현:
  - Article 단위 chunk (Title = 상위 계층 제목 연결, Content = 본문)
  - Article 하위 계층(Paragraph, Sub-paragraph, ...)마다 nested chunk 추가
  - 표는 HTML로 변환해 별도 저장, 본문에서 "Table N"으로 참조되면 생성 단계에서 첨부
  - [논문과 다른 점] 참조된 표는 평문으로 펼쳐 임베딩 텍스트에도 붙임 (embed_text)

DNV 체계 (DNV-RU-SHIP Pt.2 실제 문서로 확인, config의 parse.article_depth로 조정):
  PDF 하나 = 여러 Chapter    -> PDF 책갈피로 page -> Chapter 매핑, Section 페이지만 본문으로 사용
                                (표지, Changes – current/historic, Contents 페이지 제외)
  SECTION 1 GENERAL          -> section (Chapter마다 1부터 다시 시작 → "Pt.2 Ch.1 Section 1 ..."로 구분)
  1 General                  -> depth 1 heading
  1.1 Application            -> depth 2 heading  = Article (article_depth=2)
  1.1.1 The rules apply ...  -> depth 3          = Paragraph
  a) ...                     -> Sub-paragraph
  i) ...                     -> 그 하위
"""
import collections
import re
from html.parser import HTMLParser
from pathlib import Path

import pypdfium2 as pdfium
from docling_core.types.doc import DocItemLabel, DoclingDocument, ListItem, PictureItem, TableItem

SECTION_RE = re.compile(r"^SECTION\s+(\d+)\s*(.*)$")
NUM_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(\S.*)$")
# 번호만 있는 줄: IGC 기반 장(Pt.5 Ch.7 등)은 "3.1.1 On ships ..."의 번호를 Docling이 별도 section_header로 떼어냄
BARE_NUM_RE = re.compile(r"^(\d+(?:\.\d+){2,})\.?$")
LETTER_RE = re.compile(r"^([a-z])\)\s*(.*)$")
ROMAN_RE = re.compile(r"^(i|ii|iii|iv|v|vi|vii|viii|ix|x)\)\s*(.*)$")
CAPTION_RE = re.compile(r"^Table\s+(\d+)\b")
FIGURE_CAPTION_RE = re.compile(r"^Figure\s+(\d+)\b")
TABLE_REF_RE = re.compile(r"(?:\b(Ch\.\s?\d+)\s+)?(?:\bSec\.\s?(\d+)\s+)?\bTable\s+(\d+)\b")
FIGURE_REF_RE = re.compile(r"(?:\b(Ch\.\s?\d+)\s+)?(?:\bSec\.\s?(\d+)\s+)?\bFigure\s+(\d+)\b")
MIN_FIGURE_PT = 30  # 가로·세로 모두 이보다 작은 picture는 아이콘·장식으로 보고 버림
SECTION_KEY_RE = re.compile(r"^(Pt\.\s*\d+)\s+(Ch\.\s*\d+)\s+Section\s+(\d+)\b")
CHAPTER_RE = re.compile(r"\b(Pt\.\s*\d+\s+Ch\.\s*\d+)")
# 실제 DNV PDF의 매 페이지 반복 텍스트: 다운로드 워터마크, 우측 세로 표기
NOISE_RE = re.compile(r"^(This copy of the document is intended for use by"
                      r"|Part \d+ Chapter \d+ Section \d+$)")
# Guidance note 끝 표시 "---e-n-d---o-f---g-u-i-d-a-n-c-e---n-o-t-e---": 문장 중간에 붙어 나오기도 함
GUIDANCE_END_RE = re.compile(r"-*e-n-d-+o-f-+[a-z-]*")
EQ_NUMBER_RE = re.compile(r"^\(\s*\d+\s*\)$")  # 수식 옆 번호 "(1)"만 따로 인식된 것

LETTER_LEVEL, ROMAN_LEVEL = 100, 101
LIST_START = {LETTER_LEVEL: "a", ROMAN_LEVEL: "i"}


class Node:
    def __init__(self, label, level, first_line, page):
        self.label = label          # "1.1", "1.1.1", "a", "ii", 같은 항목 안 두 번째 목록이면 "a(2)"
        self.base = label           # 목록 기호 그대로 ("a(2)"의 "a")
        self.list_no = 1            # 같은 상위 아래 몇 번째 a)/i) 목록인지
        self.level = level
        self.lines = [first_line]
        self.children = []
        self.page = page

    def text(self):
        return " ".join(self.lines)

    def subtree_text(self):
        parts = [self.text()]
        for c in self.children:
            parts.append(c.subtree_text())
        return "\n".join(parts)


def _is_successor(stack, parts):
    """번호가 현재 번호 스택의 올바른 다음 번호인지 (본문 속 '2 m' 같은 오탐 방지)."""
    d = len(parts)
    if d <= len(stack):
        return parts[:-1] == stack[: d - 1] and parts[-1] == stack[d - 1] + 1
    if d == len(stack) + 1:
        return parts[:-1] == stack and parts[-1] == 1
    return False


def _page_map(pdf_path):
    """PDF 책갈피(최상위)로 page_no(1부터) -> (chapter_label, chapter_title, is_body) 함수를 만든다.
    책갈피가 없으면 모든 페이지를 본문으로 본다."""
    top = []
    pdf = pdfium.PdfDocument(str(pdf_path))
    for b in pdf.get_toc():
        dest = b.get_dest()
        if b.level == 0 and dest is not None:
            top.append((dest.get_index() + 1, b.get_title().replace("\xa0", " ").strip()))
    pdf.close()

    def lookup(page):
        if not top:
            return "", "", True
        before = [t for p, t in top if p <= page]
        chapters = [t for t in before if CHAPTER_RE.search(t)]
        chapter_title = chapters[-1] if chapters else ""
        chapter = CHAPTER_RE.search(chapter_title).group(1) if chapter_title else ""
        is_body = bool(before) and before[-1].startswith("Section")
        return chapter, chapter_title, is_body
    return lookup


TEXT_LABELS = {DocItemLabel.TEXT, DocItemLabel.PARAGRAPH, DocItemLabel.SECTION_HEADER,
               DocItemLabel.TITLE, DocItemLabel.LIST_ITEM, DocItemLabel.CAPTION}


def _is_junk_latex(latex):
    """수식으로 잘못 인식된 쓰레기. LaTeX 명령(\\text, \\cdots …)과 기호를 지운 글자로 판단.
    - Guidance note 끝 표시: "\\cdots e ^ { - } n ^ { - } d ^ { - } 0 ^ { - } f ...",
      "\\cdots e \\text {-} n \\text {-} d ...", "e ^ { - n - d } 0 ^ { - } 0 ^ { - } g ^ { - } u ^ { - } | d - a - n - c - e ..."
      (OCR 오인식으로 o→0, i→1/| 등이 섞여 "endof"가 깨지므로 end + guid/dance/note 조합으로 판단)
    - 글자·숫자가 하나도 없는 것: "= \\\\ = \\\\ =" """
    letters = re.sub(r"\\[a-z]+|[^a-z0-9]", "", latex.lower())
    return not letters or ("end" in letters and re.search(r"guid|dance|n[o0]te", letters) is not None)


def _page_offset(doc, start):
    """구간 변환(page_range) JSON의 page_no가 구간 내 상대 번호면 PDF 절대 번호로 바꿀 offset."""
    return start - 1 if doc.pages and min(doc.pages) < start else 0


def _top(doc, item):
    """항목 bbox의 위쪽 y (페이지 위에서부터, pt)와 세로 중심. prov가 없으면 None."""
    prov = item.prov[0]
    page = doc.pages.get(prov.page_no)
    if page is None:
        return None
    bb = prov.bbox.to_top_left_origin(page.size.height)
    return bb.t, (bb.t + bb.b) / 2


def _reorder_formulas(entries):
    """한 페이지의 [(item, top, center)]에서 수식을 세로 위치에 맞는 자리로 옮긴다.

    DNV는 수식 본체를 그림/벡터로 넣어 Docling이 수식 bbox를 오른쪽 꼬리 텍스트("for condition B.")로
    줄이는데, 읽기 순서(ReadingOrderPredictor)가 이 줄어든 bbox로 정해져 본문 단과 가로로 겹치지 않는 수식이
    페이지 본문 맨 뒤로 밀린다 (예: Pt.3 Ch.1 Sec.2 p.13 [5.1] 수식 3개가 [5.3.1] 끝으로 이동).
    DNV 본문은 1단이므로 각 수식을 같은 페이지에서 세로 중심보다 아래에서 시작하는 첫 항목 앞에 넣는다.
    본문 항목끼리의 순서는 바꾸지 않는다."""
    others = [e for e in entries if e[0].label != DocItemLabel.FORMULA]
    formulas = [e for e in entries if e[0].label == DocItemLabel.FORMULA]
    if not formulas or not others:
        return entries
    slots = {}
    for f in formulas:
        idx = next((i for i, o in enumerate(others) if o[1] > f[2]), len(others))
        slots.setdefault(idx, []).append(f)
    out = []
    for i in range(len(others) + 1):
        out += sorted(slots.get(i, []), key=lambda e: e[1])
        if i < len(others):
            out.append(others[i])
    return out


def _page_runs(doc):
    """iterate_items()를 페이지별 묶음으로 나눠, 각 묶음 안에서 수식 위치를 바로잡아 순서대로 반환."""
    run, run_page = [], None

    def flush():
        yield from (e[0] for e in _reorder_formulas(run))

    for item, _ in doc.iterate_items():
        pos = _top(doc, item) if getattr(item, "prov", None) else None
        page = item.prov[0].page_no if pos is not None else None
        if page is None or page != run_page:
            yield from flush()
            run, run_page = [], page
        if page is None:
            yield item
        else:
            run.append((item, *pos))
    yield from flush()


def _items(doc, page_info, offset=0):
    """Docling 본문을 읽기 순서대로 ("line", page, text) / ("table", page, TableItem) /
    ("picture", page, (PictureItem, (l, t, r, b)))로 반환. bbox는 페이지 왼쪽 위 기준 pt.
    수식은 페이지 안에서 세로 위치에 맞게 다시 배치한다 (_reorder_formulas)."""
    for item in _page_runs(doc):
        page = item.prov[0].page_no + offset if getattr(item, "prov", None) else None
        if page is not None and not page_info(page)[2]:  # 표지·Contents·Changes 페이지
            continue
        if isinstance(item, TableItem):
            yield "table", page, item
        elif isinstance(item, PictureItem):
            bb = item.prov[0].bbox.to_top_left_origin(doc.pages[item.prov[0].page_no].size.height)
            if bb.width >= MIN_FIGURE_PT or bb.height >= MIN_FIGURE_PT:  # 아이콘·장식 제외
                yield "picture", page, (item, (bb.l, bb.t, bb.r, bb.b))
        elif item.label == DocItemLabel.FORMULA:
            latex = (item.text or item.orig).strip()
            if latex and not EQ_NUMBER_RE.match(latex) and not _is_junk_latex(latex):
                yield "line", page, f"$$ {latex} $$"
        elif item.label in TEXT_LABELS:
            if item.label == DocItemLabel.CAPTION and item.parent and item.parent.cref.startswith("#/tables/"):
                continue  # 표에 연결된 캡션은 표 레코드에 들어가므로 본문 줄로 다시 내보내지 않음
            text = GUIDANCE_END_RE.sub("", item.text.replace("\xa0", " ")).strip()
            if NOISE_RE.match(text):
                continue
            # Docling은 목록 기호(a), i))를 marker 필드로 분리하므로 다시 붙여서 계층 인식에 사용
            if isinstance(item, ListItem) and item.marker and not text.startswith(item.marker):
                text = f"{item.marker} {text}"
            if text:
                yield "line", page, text


def parse_docling_json(parts, pdf_path, article_depth=2):
    """PDF 하나의 구간별 Docling JSON(convert_pdf 반환값 [(json_path, start_page)])을 순서대로 파싱해
    (articles, tables, figures) 반환. 구간 경계를 넘어 Section·번호 상태가 이어집니다. pdf_path는 책갈피(Chapter) 읽기용.

    articles: [{"doc", "title_path", "article_no", "root": Node}]
    tables:   [{"id", "doc", "section", "index", "caption", "html", "page", "article_no", "node"}]
    figures:  [{"id", "doc", "section", "index", "caption", "page", "bbox", "article_no", "node", "file"}]
              id는 "Pt.3 Ch.1 Sec.2 Table 1" 형식 (본문 참조 표기와 같음), article_no/node는 놓인 위치
              bbox는 페이지 왼쪽 위 기준 pt, 이미지는 figures.render_figures가 PDF 페이지를 잘라 file에 저장
    """
    doc_name = Path(pdf_path).stem
    articles, tables, figures = [], [], []

    section = ""            # "Pt.2 Ch.1 Section 1 GENERAL" (chunk/표 key로 사용)
    section_title = ""      # "Section 1 GENERAL"
    chapter_title = ""      # "DNV-RU-SHIP Pt.2 Ch.1 General requirements ..."
    page_info = _page_map(pdf_path)
    num_stack = []          # 현재 번호 스택, 예: [1, 1, 2]
    headings = {}           # depth -> "1 General" (article_depth 미만 제목)
    article = None          # 현재 article dict
    node_stack = []         # 현재 article 내부 노드 스택

    def open_article(label, line, page):
        nonlocal article, node_stack
        title_path = [doc_name, chapter_title, section_title]
        depth = len(label.split("."))
        title_path += [headings[d] for d in sorted(headings) if d < depth]
        title_path.append(line)
        root = Node(label, 0, line, page)
        article = {"doc": doc_name, "section": section, "title_path": title_path,
                   "article_no": label, "root": root}
        articles.append(article)
        node_stack = [root]

    def push(node):
        while node_stack[-1].level >= node.level:
            node_stack.pop()
        node_stack[-1].children.append(node)
        node_stack.append(node)

    def push_list(base, level, line, page):
        """a)/i) 목록 항목. DNV 원문에는 한 항목 아래에 a)~e) 목록이 두 번 나오는 경우가 있어
        (예: Pt.3 Ch.11 Sec.2 [5.7.3]) 같은 상위 아래에서 a)/i)로 다시 시작하면 두 번째 목록부터 "a(2)"처럼
        목록 순번을 붙여 node(=ref_id)가 겹치지 않게 함."""
        while node_stack[-1].level >= level:
            node_stack.pop()
        sibs = [c for c in node_stack[-1].children if c.level == level]
        list_no = sibs[-1].list_no + (base == LIST_START[level]) if sibs else 1
        node = Node(base if list_no == 1 else f"{base}({list_no})", level, line, page)
        node.base, node.list_no = base, list_no
        push(node)

    def all_items():
        for json_path, start in parts:
            d = DoclingDocument.load_from_json(json_path)
            for kind, pno, obj in _items(d, page_info, _page_offset(d, start)):
                yield d, kind, pno, obj

    def current_node():
        """그림이 놓인 위치의 chunk node 이름 (to_chunks의 node 규칙과 같음: 번호 노드는 전체 번호, a)/i)는 이어붙임)."""
        path = []
        for n in node_stack:
            path = [n.label] if n.label[0].isdigit() else path + [n.label]
        return ".".join(path)

    prev_kind, prev_line, prev_cols = None, "", None
    fig_waiting = None      # 캡션을 아직 못 찾은 직전 그림 (DNV 캡션은 그림 아래에 오는 경우가 많음)
    for doc, kind, pno, obj in all_items():
        if kind == "picture":
            item, bbox = obj
            # 캡션: Docling 연결 → 바로 앞 줄 "Figure N ..." → (아래 line 처리에서) 바로 다음 줄
            caption = item.caption_text(doc) or (prev_line if prev_kind == "line" and FIGURE_CAPTION_RE.match(prev_line) else "")
            k = sum(1 for f in figures if f["page"] == pno) + 1
            fig = {"doc": doc_name, "section": section, "index": "", "caption": caption, "page": pno,
                   "bbox": [round(v, 1) for v in bbox],
                   "article_no": article["article_no"] if article else "", "node": current_node(),
                   "file": f"{doc_name}/p{pno:04d}_{k:02d}.png"}
            figures.append(fig)
            m = FIGURE_CAPTION_RE.match(caption)
            fig["index"] = f"Figure {m.group(1)}" if m else f"p{pno}-figure{k}"
            fig["id"] = f"{section_id(section)} {fig['index']}"
            fig_waiting = None if m else fig
            # prev_kind/prev_line은 바꾸지 않음: 페이지를 넘어가는 표 사이에 그림이 끼어도 표 이어붙이기가 유지되도록
            # (그림은 본문·표 처리에 영향을 주지 않음)
            continue
        if kind == "line" and fig_waiting is not None and (m := FIGURE_CAPTION_RE.match(obj)):
            fig_waiting["caption"], fig_waiting["index"] = obj, f"Figure {m.group(1)}"
            fig_waiting["id"] = f"{section_id(fig_waiting['section'])} {fig_waiting['index']}"
        fig_waiting = None

        if kind == "table":
            html, cols = obj.export_to_html(doc), obj.data.num_cols
            # Docling이 "Table N ..." 캡션을 section_header로 분류해 표에 연결하지 못하는 경우가 많음
            # → 표 바로 앞 줄이 "Table N ..."이면 그것을 캡션으로 사용
            caption = obj.caption_text(doc) or (prev_line if prev_kind == "line" and CAPTION_RE.match(prev_line) else "")
            m = CAPTION_RE.match(caption)
            if not m and prev_kind == "table" and tables and tables[-1]["section"] == section and cols == prev_cols:
                tables[-1]["html"] += "\n" + html  # 캡션 없이 표 바로 뒤에 이어지는 표 = 다음 페이지로 넘어간 부분
            elif not m and article is not None:
                # 원문에도 캡션이 없는 표(예: Guidance note 속 작은 표)는 "Table N"으로 참조될 수 없어
                # 별도 저장하면 내용이 사라지므로, 해당 조항 본문에 Markdown 표로 넣음
                node_stack[-1].lines.append("\n" + obj.export_to_markdown(doc) + "\n")
                prev_kind = "line"
                continue
            else:
                index = f"Table {m.group(1)}" if m else f"p{pno}-table"
                tables.append({"id": f"{section_id(section)} {index}", "doc": doc_name, "section": section,
                               "index": index, "caption": caption, "html": html, "page": pno,
                               # 표가 놓인 위치: 본문에서 "Table N"으로 참조되지 않는 표도 조항에 연결되도록
                               "article_no": article["article_no"] if article else "", "node": current_node()})
            prev_kind, prev_cols = "table", cols
            continue

        line = obj
        prev_kind, prev_line = "line", line
        if m := SECTION_RE.match(line):
            if pno is not None:
                chapter, chapter_title, _ = page_info(pno)
            section_title = f"Section {m.group(1)} {m.group(2)}".strip()
            section = f"{chapter} {section_title}".strip()
            num_stack, headings, article, node_stack = [], {}, None, []
            continue

        if (b := BARE_NUM_RE.match(line)) and article is not None:
            # 번호만 있는 줄은 Paragraph 이하(depth > article_depth)이고 올바른 다음 번호일 때만 인정 (표 값 등 오탐 방지).
            # 본문은 다음 줄부터 이 노드에 붙음
            parts = [int(p) for p in b.group(1).split(".")]
            if len(parts) > article_depth and _is_successor(num_stack, parts):
                num_stack = parts
                push(Node(b.group(1), len(parts) - article_depth, b.group(1), pno))
                continue

        if m := NUM_RE.match(line):
            parts = [int(p) for p in m.group(1).split(".")]
            depth = len(parts)
            # Article 이상 제목은 번호 패턴으로만 판정: 올바른 다음 번호(_is_successor) + 대문자 시작
            # (본문 속 "6 to 10", "3 m above" 오탐 방지). 길이 제한은 두지 않음: DNV에는 100자가 넘는 제목이 있고
            # (예: Pt.2 Ch.4 Sec.5 "9 Additional considerations ...", 113자) 하나를 놓치면 이후 번호가
            # Section 끝까지 연쇄로 무너짐. 회귀는 checks.check_articles의 책갈피 대조로 잡음
            is_title = depth > article_depth or m.group(2)[0].isupper()
            if is_title and _is_successor(num_stack, parts):
                num_stack = parts
                label = m.group(1)
                if depth <= article_depth:
                    headings = {d: t for d, t in headings.items() if d < depth}
                    headings[depth] = line
                    open_article(label, line, pno)
                else:
                    if article is None:  # 상위 article 없이 하위 번호가 시작된 경우
                        open_article(".".join(m.group(1).split(".")[:article_depth]), "", pno)
                    push(Node(label, depth - article_depth, line, pno))
                continue

        if article is not None:
            cur_letter = next((n for n in reversed(node_stack) if n.level == LETTER_LEVEL), None)
            if (m := ROMAN_RE.match(line)) and cur_letter is not None and not (
                    m.group(1) == "i" and cur_letter.base == "h"
                    or m.group(1) == "v" and cur_letter.base == "u"
                    or m.group(1) == "x" and cur_letter.base == "w"):
                push_list(m.group(1), ROMAN_LEVEL, line, pno)
                continue
            if m := LETTER_RE.match(line):
                push_list(m.group(1), LETTER_LEVEL, line, pno)
                continue
            node_stack[-1].lines.append(line)

    # 내용 없는 제목뿐인 article 제거 (예: "1 General" 바로 아래 "1.1"이 오는 경우)
    articles = [a for a in articles if len(a["root"].lines) > 1 or a["root"].children]
    return articles, tables, figures


def to_chunks(articles):
    """Article chunk + 모든 하위 노드의 nested chunk 생성 (논문 Fig.3~5)."""
    chunks = []

    def walk(node, label_path, article):
        chunks.append({
            "doc": article["doc"],
            "section": article["section"],
            "article_no": article["article_no"],
            "node": ".".join(label_path),
            "title": " > ".join(t for t in article["title_path"] if t),
            "content": node.subtree_text(),
            "page": node.page,
        })
        for c in node.children:  # 번호 노드는 전체 번호("1.1.1"), a)/i)는 이어붙임("1.1.1.a.i")
            walk(c, [c.label] if c.label[0].isdigit() else label_path + [c.label], article)

    for a in articles:
        walk(a["root"], [a["article_no"]], a)
    for i, c in enumerate(chunks):
        c["chunk_id"] = i
    return chunks


def _section_of(lookup, doc, prefix):
    """'Pt.2 Ch.1 Section 1 ' 로 시작하는 section 키 (section 키에는 제목까지 붙어 있음)."""
    return next((s for d, s, _ in lookup if d == doc and s.startswith(prefix)), None)


def _referenced(chunk, lookup, ref_re, kind):
    """chunk 본문의 'Table N' / 'Figure N' 참조에 해당하는 레코드 (같은 문서 안에서).
    'X N' = 같은 Section, 'Sec.2 X N' = 같은 Chapter의 Section 2, 'Ch.2 Sec.1 X N' = 같은 Part의 Ch.2 Section 1"""
    found = []
    here = SECTION_KEY_RE.match(chunk["section"])
    for ch, sec, n in dict.fromkeys(ref_re.findall(chunk["content"])):
        section = chunk["section"]
        if ch and not sec:  # "Ch.2 Table 1"처럼 Section 없이 Chapter만 있는 참조는 대상을 특정할 수 없음
            continue
        if sec and here:
            pt, cur_ch, _ = here.groups()
            ch = ch.replace(" ", "") if ch else cur_ch
            section = _section_of(lookup, chunk["doc"], f"{pt} {ch} Section {sec} ")
        t = lookup.get((chunk["doc"], section, f"{kind} {n}"))
        if isinstance(t, list):  # 그림 lookup은 조각 목록
            found += t
        elif t:
            found.append(t)
    return found


def referenced_tables(chunk, table_lookup):
    """chunk 본문의 표 참조에 해당하는 표를 반환 (같은 문서 안에서)."""
    return _referenced(chunk, table_lookup, TABLE_REF_RE, "Table")


def _inside(chunk, rec):
    """표·그림 레코드가 chunk 범위(자기 node와 그 하위, Article chunk면 Article 전체) 안에 놓였는지."""
    node = chunk["node"]
    return (rec["doc"] == chunk["doc"] and rec["section"] == chunk["section"]
            and rec["article_no"] == chunk["article_no"]
            and (node == chunk["article_no"] or rec["node"] == node or rec["node"].startswith(node + ".")))


def chunk_figures(chunk, figure_index):
    """chunk에 보여줄 그림: chunk 범위 안에 놓인 그림(페이지 순) + 본문에서 'Figure N'으로 참조한 그림.
    figure_index = utils.load_figure_index (lookup: (doc, section, index) -> [그림], by_article: (doc, section, article_no) -> [그림])"""
    inside = [f for f in figure_index["by_article"].get((chunk["doc"], chunk["section"], chunk["article_no"]), [])
              if _inside(chunk, f)]
    out = {f["file"]: f for f in inside}
    for f in _referenced(chunk, figure_index["lookup"], FIGURE_REF_RE, "Figure"):
        out.setdefault(f["file"], f)
    return list(out.values())


# ── ⑤ 조항 ID / ⑥ 참조 관계 / ⑦ 수식·표·그림 연결 ──────────────────────────────

def section_id(section):
    """'Pt.3 Ch.1 Section 4 SYMBOLS AND DEFINITIONS' -> 'Pt.3 Ch.1 Sec.4' (DNV 본문 참조 표기와 같은 형식)."""
    m = SECTION_KEY_RE.match(section)
    return f"{m[1].replace(' ', '')} {m[2].replace(' ', '')} Sec.{m[3]}" if m else section


def ref_id(chunk):
    """정규화 조항 ID: 'Pt.3 Ch.1 Sec.4 [5.1]', 하위 항목은 '[5.1.2.a]', 같은 항목 안 두 번째 a) 목록은 '[5.7.3.a(2)]'."""
    return f"{section_id(chunk['section'])} [{chunk['node']}]"


_N = r"\s?(\d+)"
# 앞쪽 접두(Pt/Ch/Sec)가 붙은 대상(조항·표·그림·부록)을 먼저, 대상 없이 Section/Chapter/Part만 쓴 참조를 나중에 시도
REF_SCAN_RE = re.compile(
    rf"(?:\bPt\.{_N}\s+)?(?:\bCh\.{_N}\s+)?(?:\bSec\.{_N}\s+)?"
    r"(?:\[(\d+(?:\.\d+)*)\]|\bTable\s+(\d+)\b|\bFigure\s+(\d+)\b|\bApp\.\s?([A-Z])\b)"
    rf"|(?:\bPt\.{_N}\s+)?(?:\bCh\.{_N}\s+)?\bSec\.{_N}\b"
    rf"|(?:\bPt\.{_N}\s+)?\bCh\.{_N}\b"
    rf"|\bPt\.{_N}\b")
# 외부 문서: DNV 다른 문서, IMO 협약·코드, IACS, 산업 표준
EXTERNAL_DOC = (r"(?:\bDNV-(?:CG|ST|OS|RP|CP|SE|RU)-[A-Z]?\d+\b"
                r"|\b(?:SOLAS|MARPOL|COLREG|STCW|ICLL)\b"
                r"|\b(?:IGC|IBC|IGF|IMSBC|IMDG|FSS|FTP|LSA|ISM|ISPS|HSC|MODU|BCH|GC|EGC)[\s-]?[Cc]ode\b"
                r"|\bIACS\s+(?:UR|UI|Rec\.?)\s?[A-Z]?\d+\b"
                r"|\b(?:ISO|EN|ASTM|IEC)\s?[A-Z]?\d{2,}(?:-\d+)*\b"
                r"|\b(?:MSC|MEPC)\.\d+(?:/Circ\.\d+)?(?:\(\d+\))?)")
# 외부 문서명 + 그 문서 안의 위치 ("DNV-CG-0128 Sec.3 [3.2.3]", "DNV-CG-0128 App.A [1.6]", "FSS Code Ch. 15",
# "IGC Code Ch. 11.3.1", "SOLAS Reg. II-2/4.5"). 이 꼬리를 먼저 잘라내지 않으면 "Sec.3 [3.2.3]", "Ch.16"이
# 현재 Part·Chapter의 조항·Chapter로 잘못 해석됨
EXTERNAL_RE = re.compile(
    EXTERNAL_DOC
    + r"(?:,?\s+(?:Ch\.\s?\d+(?:\.\d+)*|Chapter\s+[\dIVX]+(?:-\d)?|Sec\.\s?\d+|App\.\s?[A-Z]|Annex\s+[\dIVX]+"
      r"|Reg\.?\s?[\dIVX]+(?:-\d+)?(?:/[\d.]+)?|[Rr]egulation\s+[\dIVX]+(?:-\d+)?(?:/[\d.]+)?))*"
    r"(?:\s+(?:\[\d+(?:\.\d+)*\]|Table\s+\d+\b|Figure\s+\d+\b))?")
FORMULA_RE = re.compile(r"\$\$ (.*?) \$\$")
CLAUSE_UNIT = {1: "subsection", 2: "article"}  # 그 이상은 paragraph


def _clause_exists(index, sec, num):
    """'[14]'처럼 한 자리 번호는 Article이 아니라 그 아래 조항들의 묶음(subsection)이므로 하위 존재로 판정."""
    return f"{sec} [{num}]" in index["ref_ids"] or ("." not in num and (sec, num) in index["subsections"])


def extract_refs(chunk, index):
    """chunk 본문의 참조를 대상 단위별로 정규화.
    [{"type": part|chapter|section|subsection|article|paragraph|table|figure|appendix|external,
      "text": 원문 표기, "target": 정규화 ID (특정 불가면 None),
      "in_corpus": 대상 Part가 코퍼스(파싱한 PDF)에 있는지, "resolved": 코퍼스 안에서 대상을 찾았는지}]
    in_corpus인데 resolved가 아니면 파싱 누락이나 원문 오기를 의심할 수 있음."""
    here = SECTION_KEY_RE.match(chunk["section"])
    pt0, ch0, sec0 = here[1].replace(" ", ""), here[2].replace(" ", ""), here[3]
    out, seen = [], set()
    # 수식($$…$$)은 제외: 참조가 들어갈 자리가 아니고, 텍스트가 수식으로 잘못 인식된 경우
    # ("$$ ( I G C \, \text {Code Ch. 16.7.1.3}) $$") "Ch. 16"이 현재 Part의 Chapter로 잘못 해석됨
    content = FORMULA_RE.sub(lambda m: " " * len(m.group(0)), chunk["content"])

    def add(kind, text, target, resolved):
        if (kind, target or text) not in seen:
            seen.add((kind, target or text))
            in_corpus = kind not in ("external", "appendix") and target is not None \
                and target.split(" ")[0] in index["parts"]
            out.append({"type": kind, "text": text, "target": target, "in_corpus": in_corpus, "resolved": resolved})

    # 1) 외부 문서 (+ 그 안의 위치)를 먼저 뽑고 본문에서 지워 아래 조항 해석에 섞이지 않게 함
    for m in EXTERNAL_RE.finditer(content):
        text = " ".join(m.group(0).split())
        add("external", text, text, False)
    content = EXTERNAL_RE.sub(lambda m: " " * len(m.group(0)), content)

    # 2) 코퍼스 체계(Pt/Ch/Sec/[조항]/Table/Figure) 참조
    for m in REF_SCAN_RE.finditer(content):
        g = m.groups()
        text = " ".join(m.group(0).split())
        if any(g[3:7]):  # 조항·표·그림·부록
            pt, ch, sec = g[0:3]
            kind = ("table" if g[4] else "figure" if g[5] else "appendix" if g[6]
                    else CLAUSE_UNIT.get(len(g[3].split(".")), "paragraph"))
            # 생략된 상위는 현재 위치로 채움. 단 위가 주어지고 중간이 빠진 경우("Pt.4 [1.2]", "Ch.2 Table 1",
            # "Pt.4 Sec.2 ...")는 대상을 특정할 수 없음 (부록은 Section이 없으므로 Chapter까지만 필요)
            if (pt and not ch) or (ch and not sec and kind != "appendix"):
                add(kind, text, None, False)
                continue
            pt = f"Pt.{pt}" if pt else pt0
            ch = f"Ch.{ch}" if ch else ch0
            if kind == "appendix":  # 부록은 본문(Section 책갈피)에 포함하지 않으므로 코퍼스에 없음
                add(kind, text, f"{pt} {ch} App.{g[6]}", False)
                continue
            sec_key = f"{pt} {ch} Sec.{sec or sec0}"
            if g[3]:
                add(kind, text, f"{sec_key} [{g[3]}]", _clause_exists(index, sec_key, g[3]))
            elif g[4]:
                tid = f"{sec_key} Table {g[4]}"
                add(kind, text, tid, tid in index["table_ids"])
            else:
                fid = f"{sec_key} Figure {g[5]}"
                add(kind, text, fid, fid in index["figure_ids"])
        elif g[9]:  # [Pt] [Ch] Sec
            pt = f"Pt.{g[7]}" if g[7] else pt0
            ch = f"Ch.{g[8]}" if g[8] else (ch0 if not g[7] else None)
            target = f"{pt} {ch} Sec.{g[9]}" if ch else None
            add("section", text, target, target in index["sections"])
        elif g[11]:  # [Pt] Ch
            target = f"Pt.{g[10]}" if g[10] else pt0
            target = f"{target} Ch.{g[11]}"
            add("chapter", text, target, target in index["chapters"])
        elif g[12]:
            target = f"Pt.{g[12]}"
            add("part", text, target, target in index["parts"])
    return out


def annotate_chunks(chunks, tables, figures):
    """chunk마다 ⑤ ref_id, ⑥ refs, ⑦ formulas/table_ids/figure_ids를 붙인다 (chunk 수·순서는 그대로).
    - table_ids/figure_ids = chunk 범위 안에 놓인 표·그림 + 본문 참조로 해석된 표·그림 (id 순서 유지, 중복 제거)
    - 다른 Part 참조는 그 Part가 코퍼스에 있으면 그대로 해석됨 (ID에 Part가 들어 있으므로)"""
    for c in chunks:
        c["ref_id"] = ref_id(c)
    index = {
        "ref_ids": {c["ref_id"] for c in chunks},
        "subsections": {(section_id(c["section"]), c["node"].split(".")[0]) for c in chunks},
        "sections": {section_id(c["section"]) for c in chunks},
        "chapters": {section_id(c["section"]).rsplit(" ", 1)[0] for c in chunks},
        "parts": {section_id(c["section"]).split(" ")[0] for c in chunks},
        "table_ids": {t["id"] for t in tables},
        "figure_ids": {f["id"] for f in figures},
    }
    by_article = collections.defaultdict(lambda: ([], []))
    for t in tables:
        by_article[(t["doc"], t["section"], t["article_no"])][0].append(t)
    for f in figures:
        by_article[(f["doc"], f["section"], f["article_no"])][1].append(f)

    for c in chunks:
        c["refs"] = extract_refs(c, index)
        c["formulas"] = FORMULA_RE.findall(c["content"])
        tabs, figs = by_article.get((c["doc"], c["section"], c["article_no"]), ([], []))
        ref_targets = [r["target"] for r in c["refs"] if r["resolved"]]
        c["table_ids"] = list(dict.fromkeys(
            [t["id"] for t in tabs if _inside(c, t)] + [t for t in ref_targets if t in index["table_ids"]]))
        c["figure_ids"] = list(dict.fromkeys(
            [f["id"] for f in figs if _inside(c, f)] + [f for f in ref_targets if f in index["figure_ids"]]))
    return chunks


class _TableText(HTMLParser):
    """표 HTML → 행마다 "셀 | 셀 | ..." 평문 (여러 페이지로 합쳐진 <table> 여러 개도 그대로 이어서 처리)."""

    def __init__(self):
        super().__init__()
        self.rows, self.cell = [], None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.rows.append([])
        elif tag in ("td", "th"):
            self.cell = []

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None:
            text = " ".join("".join(self.cell).split())
            if text and self.rows:
                self.rows[-1].append(text)
            self.cell = None

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)


def table_to_text(html):
    parser = _TableText()
    parser.feed(html)
    return "\n".join(" | ".join(r) for r in parser.rows if r)


def embed_text(chunk, table_lookup):
    """임베딩용 텍스트 = content + 참조된 표의 평문.
    DNV는 답(계수·최소값 등)이 표 행에만 있는 경우가 많아, content만 임베딩하면 그 질문에서 정답 chunk가 밀림."""
    parts = [chunk["content"]]
    for t in referenced_tables(chunk, table_lookup):
        body = table_to_text(t["html"])
        parts.append(body if t["caption"] and t["caption"] in chunk["content"] else f"{t['caption']}\n{body}")
    return "\n".join(parts)


def chunk_texts(chunks, table_lookup=None):
    """검색·GPL·학습에 쓰는 chunk 텍스트. table_lookup이 None이면 논문대로 content만 사용."""
    if table_lookup is None:
        return [c["content"] for c in chunks]
    return [embed_text(c, table_lookup) for c in chunks]
