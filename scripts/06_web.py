"""6단계: 웹 UI 실행.
  python scripts/06_web.py                     # 검색만 (CPU로 충분), http://127.0.0.1:8000
  python scripts/06_web.py --generate          # LLaMA 답변 생성까지 (GPU 필요)
  python scripts/06_web.py --host 0.0.0.0 --port 8080
02_build_index.py로 base 또는 adapted 인덱스를 먼저 만들어야 합니다.
"""
import argparse

import uvicorn

from shiprag.utils import load_config
from shiprag.web import create_app

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config.yaml")
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--generate", action="store_true", help="LLaMA 답변 생성 활성화 (GPU 필요)")
args = ap.parse_args()

app = create_app(load_config(args.config), generate=args.generate)
uvicorn.run(app, host=args.host, port=args.port)
