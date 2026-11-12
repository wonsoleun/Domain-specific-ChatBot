"""수식 영역 보정(fix_formula_bbox) 전후 비교: 지정한 쪽만 보정 끄고/켜고 두 번 변환해
수식별 LaTeX를 출력하고, Docling 이미지 내보내기(item.get_image)로 수식 영역을 잘라
<out>/<off|on>/pNNNN_kk.png로 저장합니다. CodeFormula는 이 영역을 사방 18%씩 더 넓혀 봅니다.
  python scripts/diag_formula.py --pdf "data/raw/DNV Rule SHIP Part 3 Hull - 2026_S.pdf" --pages 12 13 66
"""
import argparse
from pathlib import Path

from docling_core.types.doc import DocItemLabel

from shiprag.convert import make_converter

ap = argparse.ArgumentParser()
ap.add_argument("--pdf", required=True)
ap.add_argument("--pages", type=int, nargs="+", default=[12, 13])
ap.add_argument("--out", default="data/diag_formula")
args = ap.parse_args()

results = {}
for name, fix in (("off", False), ("on", True)):
    out_dir = Path(args.out) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    converter = make_converter(fix_formula_bbox=fix, page_images_scale=2.0)
    rows = []
    for page_no in args.pages:
        doc = converter.convert(args.pdf, page_range=(page_no, page_no)).document
        for k, item in enumerate(t for t in doc.texts if t.label == DocItemLabel.FORMULA):
            bb = item.prov[0].bbox
            png = out_dir / f"p{page_no:04d}_{k:02d}.png"
            img = item.get_image(doc)
            if img is not None:
                img.save(png)
            rows.append((page_no, f"x[{bb.l:.0f}-{bb.r:.0f}] y[{bb.b:.0f}-{bb.t:.0f}]", item.text, png.name))
    results[name] = rows

for name, rows in results.items():
    print(f"\n=== fix_formula_bbox={name} ({len(rows)} formulas) ===")
    for page_no, box, latex, png in rows:
        print(f"p.{page_no} {box} {png}: {latex}")
