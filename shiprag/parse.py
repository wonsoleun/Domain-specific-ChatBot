"""Docling JSON -> 계층형 nested chunks + 표(HTML).

입력은 convert.py가 만든 Docling JSON입니다 (수식은 LaTeX, 표는 셀 구조 포함, 머리글/바닥글 제외).

논문 2.2.1 재현:
  - Article 단위 chunk (Title = 상위 계층 제목 연결, Content = 본문)
  - Article 하위 계층(Paragraph, Sub-paragraph, ...)마다 nested chunk 추가
  - 표는 HTML로 변환해 별도 저장, 본문에서 "Table N"으로 참조되면 생성 단계에서 첨부

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
import re
from pathlib import Path

import pypdfium2 as pdfium
from docling_core.types.doc import DocItemLabel, DoclingDocument, ListItem, TableItem

SECTION_RE = re.compile(r"^SECTION\s+(\d+)\s*(.*)$")
NUM_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(\S.*)$")
LETTER_RE = re.compile(r"^([a-z])\)\s*(.*)$")
ROMAN_RE = re.compile(r"^(i|ii|iii|iv|v|vi|vii|viii|ix|x)\)\s*(.*)$")
CAPTION_RE = re.compile(r"^Table\s+(\d+)\b")
TABLE_REF_RE = re.compile(r"\bTable\s+(\d+)\b")
CHAPTER_RE = re.compile(r"\b(Pt\.\s*\d+\s+Ch\.\s*\d+)")
# 실제 DNV PDF의 매 페이지 반복 텍스트: 다운로드 워터마크, 우측 세로 표기, Guidance note 끝 표시
NOISE_RE = re.compile(r"^(This copy of the document is intended for use by"
                      r"|-+e-n-d-+o-f-+[a-z-]*-+$"
                      r"|Part \d+ Chapter \d+ Section \d+$)")

LETTER_LEVEL, ROMAN_LEVEL = 100, 101


class Node:
    def __init__(self, label, level, first_line, page):
        self.label = label          # "1.1", "1.1.1", "a", "ii"
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


def _page_offset(doc, start):
    """구간 변환(page_range) JSON의 page_no가 구간 내 상대 번호면 PDF 절대 번호로 바꿀 offset."""
    return start - 1 if doc.pages and min(doc.pages) < start else 0


def _items(doc, page_info, offset=0):
    """Docling 본문을 읽기 순서대로 ("line", page, text) / ("table", page, TableItem)로 반환."""
    for item, _ in doc.iterate_items():
        page = item.prov[0].page_no + offset if getattr(item, "prov", None) else None
        if page is not None and not page_info(page)[2]:  # 표지·Contents·Changes 페이지
            continue
        if isinstance(item, TableItem):
            yield "table", page, item
        elif item.label == DocItemLabel.FORMULA:
            latex = (item.text or item.orig).strip()
            if latex:
                yield "line", page, f"$$ {latex} $$"
        elif item.label in TEXT_LABELS:
            text = item.text.replace("\xa0", " ").strip()
            if NOISE_RE.match(text):
                continue
            # Docling은 목록 기호(a), i))를 marker 필드로 분리하므로 다시 붙여서 계층 인식에 사용
            if isinstance(item, ListItem) and item.marker and not text.startswith(item.marker):
                text = f"{item.marker} {text}"
            if text:
                yield "line", page, text


def parse_docling_json(parts, pdf_path, article_depth=2):
    """PDF 하나의 구간별 Docling JSON(convert_pdf 반환값 [(json_path, start_page)])을 순서대로 파싱해
    (articles, tables) 반환. 구간 경계를 넘어 Section·번호 상태가 이어집니다. pdf_path는 책갈피(Chapter) 읽기용.

    articles: [{"doc", "title_path", "article_no", "root": Node}]
    tables:   [{"doc", "section", "index", "caption", "html", "page"}]
    """
    doc_name = Path(pdf_path).stem
    articles, tables = [], []

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

    def all_items():
        for json_path, start in parts:
            d = DoclingDocument.load_from_json(json_path)
            for kind, pno, obj in _items(d, page_info, _page_offset(d, start)):
                yield d, kind, pno, obj

    for doc, kind, pno, obj in all_items():
        if kind == "table":
            caption = obj.caption_text(doc)
            m = CAPTION_RE.match(caption)
            tables.append({"doc": doc_name, "section": section,
                           "index": f"Table {m.group(1)}" if m else f"p{pno}-table",
                           "caption": caption, "html": obj.export_to_html(doc), "page": pno})
            continue

        line = obj
        if m := SECTION_RE.match(line):
            if pno is not None:
                chapter, chapter_title, _ = page_info(pno)
            section_title = f"Section {m.group(1)} {m.group(2)}".strip()
            section = f"{chapter} {section_title}".strip()
            num_stack, headings, article, node_stack = [], {}, None, []
            continue

        if m := NUM_RE.match(line):
            parts = [int(p) for p in m.group(1).split(".")]
            depth = len(parts)
            # Article 이상 제목은 짧고 대문자로 시작 (표/본문 속 "6 to 10", "3 m above" 오탐 방지)
            is_title = depth > article_depth or (len(line) <= 100 and m.group(2)[0].isupper())
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
                    m.group(1) == "i" and cur_letter.label == "h"
                    or m.group(1) == "v" and cur_letter.label == "u"
                    or m.group(1) == "x" and cur_letter.label == "w"):
                push(Node(m.group(1), ROMAN_LEVEL, line, pno))
                continue
            if m := LETTER_RE.match(line):
                push(Node(m.group(1), LETTER_LEVEL, line, pno))
                continue
            node_stack[-1].lines.append(line)

    # 내용 없는 제목뿐인 article 제거 (예: "1 General" 바로 아래 "1.1"이 오는 경우)
    articles = [a for a in articles if len(a["root"].lines) > 1 or a["root"].children]
    return articles, tables


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


def referenced_tables(chunk, table_lookup):
    """chunk 본문에 'Table N'이 있으면 같은 문서·같은 Section의 표를 반환."""
    found = []
    for n in dict.fromkeys(TABLE_REF_RE.findall(chunk["content"])):
        t = table_lookup.get((chunk["doc"], chunk["section"], f"Table {n}"))
        if t:
            found.append(t)
    return found
