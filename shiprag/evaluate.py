"""검색 정확도 (논문 4.2.1, 식 4~5).

S_i = 100 - 10·(R_i - 1)  (1 ≤ R_i ≤ 10),  그 외 0
R_i = 정답 Article의 순위. nested chunk 때문에 같은 Article이 여러 번 나올 수 있으므로
     검색 결과를 Article 단위로 중복 제거한 뒤 순위를 셉니다.
     (DNV는 Section마다 번호가 1부터 다시 시작하므로 Article = (doc, section, article_no))
"""
import numpy as np


def article_rank(hits, doc, section, article_no):
    seen = []
    for h in hits:
        key = (h["doc"], h["section"], h["article_no"])
        if key not in seen:
            seen.append(key)
            if key == (doc, section, article_no):
                return len(seen)
    return None


def evaluate(retriever, testset, search_k=200):
    ranks = []
    for t in testset:
        hits = retriever.search(t["query"], k=search_k)
        ranks.append(article_rank(hits, t["doc"], t["section"], str(t["article"])))
    scores = [100 - 10 * (r - 1) if r and r <= 10 else 0 for r in ranks]
    return {
        "accuracy": float(np.mean(scores)),
        "recall@1": float(np.mean([bool(r and r <= 1) for r in ranks])),
        "recall@5": float(np.mean([bool(r and r <= 5) for r in ranks])),
        "ranks": ranks,
    }
