"""웹 UI: 질의 → bi-encoder 검색(Top-k) → 조항 + 참조 표·그림 표시, (선택) LLaMA 답변.

FastAPI 백엔드 + 정적 HTML 한 장(shiprag/static/index.html). 실행은 scripts/06_web.py.
- 인덱스(base/adapted)는 처음 요청될 때 로드하고 메모리에 유지 (모델 로드는 한 번만).
- 답변 생성은 generate=True로 앱을 만들 때만 LLaMA를 로드 (GPU 필요). 기본은 검색만.
"""
import threading
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .parse import chunk_figures, referenced_tables
from .retrieve import Retriever
from .utils import load_figure_index, load_table_lookup, read_jsonl

STATIC_DIR = Path(__file__).parent / "static"


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    index: str = "base"
    k: int = Field(default=5, ge=1, le=20)
    generate: bool = False


def create_app(cfg, generate=False):
    p = cfg["paths"]
    index_dirs = {"base": p["index_base"], "adapted": p["index_adapted"]}
    table_lookup = load_table_lookup(p["tables"])
    figure_index = load_figure_index(p.get("figures", "data/processed/figures.jsonl"))
    figure_dir = Path(p.get("figure_dir", "data/processed/figures"))
    tables_by_id, figures_by_id = defaultdict(list), defaultdict(list)  # 그림 하나가 picture 여러 조각일 수 있어 리스트
    for t in read_jsonl(p["tables"]):  # table_lookup은 키가 겹치는 표를 덮어쓰므로 파일에서 직접
        tables_by_id[t.get("id")].append(t)
    for fs in figure_index["by_article"].values():
        for f in fs:
            figures_by_id[f.get("id")].append(f)

    def hit_tables(h):
        """01이 저장한 table_ids(놓인 위치 + 본문 참조) 사용. 예전 chunks.jsonl(필드 없음)이면 본문 참조만."""
        if "table_ids" not in h:
            return referenced_tables(h, table_lookup)
        return [t for tid in h["table_ids"] for t in tables_by_id.get(tid, [])]

    def hit_figures(h):
        if "figure_ids" not in h:
            return chunk_figures(h, figure_index)
        return [f for fid in h["figure_ids"] for f in figures_by_id.get(fid, [])]

    retrievers, load_lock = {}, threading.Lock()
    gen, gen_lock = None, threading.Lock()

    if generate:
        from .generate import Generator
        g = cfg["generator"]
        gen = Generator(g["model"], g["load_in_4bit"], g["max_new_tokens"])

    def available():
        return {name: (Path(d) / "meta.json").exists() for name, d in index_dirs.items()}

    def get_retriever(name):
        if name not in index_dirs:
            raise HTTPException(400, f"알 수 없는 인덱스: {name}")
        if not available()[name]:
            raise HTTPException(404, f"{name} 인덱스가 없습니다. "
                                     f"python scripts/02_build_index.py --model {name} 명령을 먼저 실행하세요.")
        with load_lock:
            if name not in retrievers:
                retrievers[name] = Retriever(index_dirs[name])
        return retrievers[name]

    app = FastAPI(title="ship-reg-rag")
    # 01이 저장한 그림 PNG (PDF 페이지를 잘라 만든 것). 폴더가 없어도 앱은 뜨고 그림만 404
    app.mount("/figures", StaticFiles(directory=figure_dir, check_dir=False), name="figures")

    @app.on_event("startup")
    def preload():
        # 첫 검색이 느리지 않도록 사용 가능한 인덱스 하나를 미리 로드 (adapted 우선)
        for name in ("adapted", "base"):
            if available()[name]:
                get_retriever(name)
                break

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    n_chunks = sum(1 for _ in open(p["chunks"], encoding="utf-8"))
    docs = sorted({d for d, _, _ in table_lookup})

    @app.get("/api/status")
    def status():
        return {"indexes": available(), "generate": gen is not None,
                "top_k": cfg["retriever"]["top_k"],
                "n_chunks": n_chunks, "n_tables": len(table_lookup),
                "n_figures": sum(map(len, figure_index["by_article"].values())), "docs": docs}

    @app.post("/api/search")
    def search(req: SearchRequest):  # def(동기) → FastAPI가 스레드풀에서 실행
        query = req.query.strip()
        if not query:
            raise HTTPException(400, "질문을 입력하세요.")
        retriever = get_retriever(req.index)
        hits = retriever.search(query, k=req.k)
        answer = None
        if req.generate:
            if gen is None:
                raise HTTPException(400, "답변 생성이 꺼져 있습니다 (06_web.py --generate, GPU 필요).")
            with gen_lock:  # LLaMA는 한 번에 한 요청만
                answer = gen.answer(query, hits, table_lookup)
        results = [{
            "rank": h["rank"], "score": h["score"], "title": h["title"], "doc": h["doc"],
            "section": h["section"], "article_no": h["article_no"], "node": h["node"],
            "ref_id": h.get("ref_id", ""), "page": h["page"], "content": h["content"],
            "tables": [{"caption": t["caption"], "html": t["html"], "page": t["page"]} for t in hit_tables(h)],
            "figures": [{"caption": f["caption"], "page": f["page"], "url": f"/figures/{quote(f['file'])}"}
                        for f in hit_figures(h)],
        } for h in hits]
        return {"query": query, "index": req.index, "results": results, "answer": answer}

    return app
