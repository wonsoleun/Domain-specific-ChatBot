"""웹 UI: 질의 → bi-encoder 검색(Top-k) → 조항 + 참조 표 표시, (선택) LLaMA 답변.

FastAPI 백엔드 + 정적 HTML 한 장(shiprag/static/index.html). 실행은 scripts/06_web.py.
- 인덱스(base/adapted)는 처음 요청될 때 로드하고 메모리에 유지 (모델 로드는 한 번만).
- 답변 생성은 generate=True로 앱을 만들 때만 LLaMA를 로드 (GPU 필요). 기본은 검색만.
"""
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .parse import referenced_tables
from .retrieve import Retriever
from .utils import load_table_lookup

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
                "n_chunks": n_chunks, "n_tables": len(table_lookup), "docs": docs}

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
            "page": h["page"], "content": h["content"],
            "tables": [{"caption": t["caption"], "html": t["html"], "page": t["page"]}
                       for t in referenced_tables(h, table_lookup)],
        } for h in hits]
        return {"query": query, "index": req.index, "results": results, "answer": answer}

    return app
