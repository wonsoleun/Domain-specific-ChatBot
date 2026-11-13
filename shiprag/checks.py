"""01 결과 고정 검사: 파서 변경 후 조항 누락·오탐과 ID 중복을 잡는다.

1) 책갈피 대조: PDF 책갈피(Chapter > Section > "1 ..." > "1.1 ...")의 Article이 chunks에 모두 있고,
   책갈피에 없는 Article(번호에 '.'이 있는 것)이 chunks에 생기지 않았는지.
   (한 자리 번호 "1 General"에 본문이 바로 붙어 Article로 잡힌 것은 책갈피 Article이 아니므로 비교에서 제외)
2) ID 유일성: chunk의 ref_id가 모두 다른지 (질문 속 조항 번호로 바로 조회하는 경로의 전제).
"""
import collections
import re
from pathlib import Path

import pypdfium2 as pdfium

CHAPTER_RE = re.compile(r"\b(Pt\.\s*\d+)\s+(Ch\.\s*\d+)")
SECTION_BM_RE = re.compile(r"^Section\s+(\d+)\b")
ARTICLE_BM_RE = re.compile(r"^(\d+\.\d+)\s")
SECTION_KEY_RE = re.compile(r"^(Pt\.\d+) (Ch\.\d+) Section (\d+)\b")


def bookmark_articles(pdf_path):
    """PDF 책갈피의 Article 집합 {("Pt.2 Ch.1", "3", "1.1")}.
    DNV 책갈피 구조: level 0 = Chapter 제목 또는 "Section N ...", level 1 = "1 General", level 2 = "1.1 Scope"."""
    out, chapter, section = set(), None, None
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        for b in pdf.get_toc():
            title = b.get_title().replace("\xa0", " ").strip()
            if b.level == 0:
                if m := CHAPTER_RE.search(title):
                    chapter, section = f"{m[1].replace(' ', '')} {m[2].replace(' ', '')}", None
                else:
                    m = SECTION_BM_RE.match(title)
                    section = m[1] if m else None
            elif b.level == 2 and chapter and section and (m := ARTICLE_BM_RE.match(title)):
                out.add((chapter, section, m[1]))
    finally:
        pdf.close()
    return out


def check_articles(chunks, pdf_paths):
    """PDF별 책갈피 대조. {doc: {"bookmark": n, "missing": [...], "extra": [...]}}"""
    parsed = collections.defaultdict(set)
    for c in chunks:
        if m := SECTION_KEY_RE.match(c["section"]):
            parsed[c["doc"]].add((f"{m[1]} {m[2]}", m[3], c["article_no"]))
    report = {}
    for pdf in pdf_paths:
        doc = Path(pdf).stem
        bm = bookmark_articles(pdf)
        if not bm:  # 책갈피가 없는 PDF는 비교 불가
            continue
        mine = {a for a in parsed[doc] if "." in a[2]}
        report[doc] = {"bookmark": len(bm), "missing": sorted(bm - mine), "extra": sorted(mine - bm)}
    return report


def check_unique_ids(chunks):
    """중복된 ref_id 목록 [(ref_id, [chunk_id...])]."""
    ids = collections.defaultdict(list)
    for c in chunks:
        ids[c["ref_id"]].append(c["chunk_id"])
    return [(k, v) for k, v in ids.items() if len(v) > 1]


def run_checks(chunks, pdf_paths, show=10):
    """검사 결과를 출력하고 통과 여부 반환."""
    ok = True
    for doc, r in check_articles(chunks, pdf_paths).items():
        bad = r["missing"] or r["extra"]
        ok &= not bad
        print(f"[check] {doc}: 책갈피 Article {r['bookmark']}개, 누락 {len(r['missing'])}, 책갈피에 없음 {len(r['extra'])}"
              + ("" if bad else " → OK"))
        for kind in ("missing", "extra"):
            for ch, sec, art in r[kind][:show]:
                print(f"    {kind}: {ch} Sec.{sec} [{art}]")
    dup = check_unique_ids(chunks)
    ok &= not dup
    print(f"[check] ref_id 중복 {len(dup)}개" + ("" if dup else " → OK"))
    for k, v in dup[:show]:
        print(f"    {k}: chunk_id {v}")
    return ok
