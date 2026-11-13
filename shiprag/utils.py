import json
from pathlib import Path

import yaml


def load_config(path="config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_table_lookup(path):
    return {(t["doc"], t["section"], t["index"]): t for t in read_jsonl(path)}


def load_figure_index(path):
    """figures.jsonl → parse.chunk_figures용 조회 구조. 파일이 없으면(01을 그림 추가 전에 돌린 경우) 빈 구조."""
    from .figures import build_figure_index
    return build_figure_index(read_jsonl(path) if Path(path).exists() else [])
