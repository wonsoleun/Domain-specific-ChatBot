"""Docling 변환 RAM 진단: 설정을 하나씩 켜며 몇 쪽만 변환하고 메모리를 3초마다 출력.
각 설정은 별도 프로세스로 돌리므로 한 설정이 RAM 부족으로 죽어도 어디서 죽었는지 로그에 남습니다.
  python scripts/diag_convert.py                          # data/raw 첫 PDF의 27~31쪽
  python scripts/diag_convert.py --pdf "data/raw/xxx.pdf" --pages 27 31
"""
import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

CONFIGS = {
    "base":            dict(do_table_structure=False, do_formula_enrichment=False),
    "+table":          dict(do_table_structure=True, do_formula_enrichment=False),
    "+table+formula":  dict(do_table_structure=True, do_formula_enrichment=True),
}


def child(pdf, start, end, name):
    import torch
    from shiprag.convert import make_converter

    proc = psutil.Process()
    peak = [0]

    def monitor():
        t0 = time.time()
        while True:
            rss = proc.memory_info().rss / 2**30
            peak[0] = max(peak[0], rss)
            gpu = f"  GPU {torch.cuda.memory_allocated() / 2**30:.1f}GB" if torch.cuda.is_available() else ""
            print(f"    [{name}] {time.time() - t0:5.0f}s  RSS {rss:.1f}GB  "
                  f"free {psutil.virtual_memory().available / 2**30:.1f}GB{gpu}", flush=True)
            time.sleep(3)

    threading.Thread(target=monitor, daemon=True).start()
    converter = make_converter(**CONFIGS[name])
    print(f"    [{name}] models loaded", flush=True)
    t0 = time.time()
    doc = converter.convert(pdf, page_range=(start, end)).document
    print(f"  RESULT {name}: OK  {time.time() - t0:.0f}s for {end - start + 1} pages, "
          f"peak RSS {peak[0]:.1f}GB, {len(doc.tables)} tables, "
          f"{sum(1 for t in doc.texts if t.label == 'formula')} formulas", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=None)
    ap.add_argument("--pages", type=int, nargs=2, default=[27, 31])
    ap.add_argument("--child", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    pdf = args.pdf or str(sorted(Path("data/raw").glob("*.pdf"))[0])

    if args.child:
        child(pdf, *args.pages, args.child)
        sys.exit()

    import torch
    print(f"total RAM {psutil.virtual_memory().total / 2**30:.1f}GB, "
          f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE (CPU runtime)'}")
    print(f"PDF {pdf}, pages {args.pages[0]}-{args.pages[1]}")
    for name in CONFIGS:
        print(f"\n=== {name} ===", flush=True)
        r = subprocess.run([sys.executable, __file__, "--pdf", pdf, "--pages", *map(str, args.pages),
                            "--child", name])
        if r.returncode != 0:
            print(f"  RESULT {name}: FAILED (exit code {r.returncode}; -9 = RAM 부족으로 강제 종료)")
