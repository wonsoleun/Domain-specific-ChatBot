# ship-reg-rag

Han et al. (2026), *"A RAG-based Q&A system for ship regulations applying domain adaptation"* (IJNAOE 18, 100735) 구조를 DNV 규정 PDF에 적용한 baseline입니다.

```
DNV PDF ─▶ Docling JSON(수식 LaTeX·표 구조) ─▶ ① 계층형 DB 구축 ─▶ ② base 인덱스 ─▶ ③ GPL 학습데이터 ─▶ ④ Domain adaptation
                                                                       │
질의 ─▶ adapted bi-encoder 검색(Top-k) ─▶ 참조 표(HTML) 첨부 ─▶ LLaMA 3.1 8B (4-bit) ─▶ 답변 + 출처
```

| 단계 | 논문 | 이 repo | 모델 |
|---|---|---|---|
| ① DB 구축 | 2.2.1 | `shiprag/convert.py`, `shiprag/parse.py` | Docling (TableFormer, CodeFormula 수식→LaTeX) |
| ② 검색 | 2.2.2 | `shiprag/retrieve.py` | `msmarco-bert-base-dot-v5` (BERT + MS MARCO, dot product) |
| ③ GPL 데이터 | 3.2 | `shiprag/gpl.py` | GPT-4o (질문 생성), `ms-marco-MiniLM-L-6-v2` (cross-encoder) |
| ④ 학습 | 3.1, 4.2 | `shiprag/train.py` | MarginMSE, AdamW, lr 1e-5, batch 64 |
| ⑤ 생성 | 2.2.3, 4.3 | `shiprag/generate.py` | `Llama-3.1-8B-Instruct` 4-bit |
| 평가 | 4.2.1 | `shiprag/evaluate.py` | 식 (4)~(5) 순위 점수 |

## 설치

```bash
pip install -r requirements.txt
pip install -e .
```

- LLaMA 3.1은 gated 모델입니다. Hugging Face에서 라이선스에 동의한 뒤 `huggingface-cli login`(Colab은 `HF_TOKEN` secret)이 필요합니다.
- 3단계는 `OPENAI_API_KEY` 환경변수가 필요합니다.

## 실행 순서

DNV PDF를 `data/raw/`에 넣고:

```bash
python scripts/01_build_db.py                          # PDF → Docling JSON/MD → chunks.jsonl, tables.jsonl
python scripts/02_build_index.py --model base          # base 모델 인덱스
python scripts/03_generate_gpl_data.py --step queries  # GPT-4o 질문 생성 (중단 후 재실행 시 이어서)
python scripts/03_generate_gpl_data.py --step label    # negative mining + cross-encoder 라벨
python scripts/04_train.py                             # domain adaptation
python scripts/02_build_index.py --model adapted       # adapted 모델 인덱스
python scripts/05_ask.py --query "What is the minimum thickness of bottom plating?"
```

검색만 확인할 때는 `--no-generate`, base 모델과 비교할 때는 `--index base`를 붙입니다. `--query`를 생략하면 대화형 모드로 실행되어 모델을 한 번만 로드합니다.

### 검색 성능 비교 (선택)

직접 만든 테스트셋(JSONL)으로 base vs adapted를 비교합니다.

```json
{"query": "...", "doc": "DNV-RU-SHIP-Pt2", "section": "Pt.2 Ch.1 Section 3 TESTING PROCEDURES", "article": "3.1"}
```

`doc` / `section` / `article` 값은 `chunks.jsonl`의 `doc` / `section` / `article_no`를 그대로 복사하세요 (DNV는 Section마다 번호가 다시 시작하므로 section까지 있어야 Article이 구분됩니다).

```bash
python scripts/eval_retrieval.py --testset data/test.jsonl
```

### Colab

```python
!git clone https://github.com/<you>/ship-reg-rag.git
%cd ship-reg-rag
!pip install -q -r requirements.txt && pip install -q -e .
# PDF 업로드 → data/raw/
import os
from google.colab import userdata
os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
!python scripts/01_build_db.py
...
```

Colab 세션은 끊길 수 있으니 `data/`, `models/`를 Google Drive에 두는 것을 권장합니다 (`config.yaml`의 `paths` 수정).

## 출력 데이터 형식

`data/processed/docling/<PDF이름>/p0001-0050.json` / `.md`: Docling 변환 결과 (`convert.batch_pages`쪽 단위). JSON은 파싱 입력이고, **Markdown은 변환이 잘 됐는지 눈으로 확인하는 용도**입니다(수식은 `$$ ... $$`, 표는 Markdown 표로 보입니다). 수백 쪽을 한 번에 변환하면 Colab RAM이 부족해 프로세스가 죽기 때문에(출력 끝에 `^C`) 구간마다 저장하며, 이미 JSON이 있는 구간은 다시 변환하지 않습니다. 변환 설정을 바꿨다면 해당 폴더를 지우고 다시 실행하세요.


`chunks.jsonl` (한 줄 = 한 chunk)

```json
{"chunk_id": 120, "doc": "DNV-RU-SHIP-Pt2", "section": "Pt.2 Ch.1 Section 3 TESTING PROCEDURES",
 "article_no": "3.1", "node": "3.1.7",
 "title": "DNV-RU-SHIP-Pt2 > DNV-RU-SHIP Pt.2 Ch.1 General requirements ... > Section 3 TESTING PROCEDURES > 3 Test methods > 3.1 Tensile testing at ambient temperature",
 "content": "3.1.7 Reduction of area The reduction of area at fracture Z (%) ... $$ Z = ... $$ ...", "page": 30}
```

- `node == article_no`이면 Article 전체 chunk, 그 외는 nested chunk (논문 Fig.3~5)
- `title`은 출처로 표시, `content`만 임베딩

`tables.jsonl`: `{"doc", "section", "index": "Table 1", "caption", "html", "page"}`. 검색된 chunk 본문에 `Table N`이 있으면 같은 문서·같은 Section의 표 HTML이 LLM 입력에 첨부됩니다 (논문 Fig.6).

## 논문과 다른 점 / 제약

1. **Negative mining**: 논문은 생성 질문 × DB 전체를 cross-encoder로 채점합니다(약 6만 × 15만 = 90억 쌍). 이는 현실적으로 불가능하므로, 하위 30% 구간은 **base bi-encoder 점수**로 정하고 cross-encoder는 실제 학습 쌍의 라벨링에만 씁니다. 또 같은 Article의 chunk는 negative에서 제외합니다(정답 텍스트를 포함할 수 있음).
2. **수식 → LaTeX**: 논문은 변환 방법을 밝히지 않아, Docling의 formula enrichment(CodeFormula 모델)를 사용합니다. 본문 수식은 `$$ LaTeX $$` 형태로 chunk에 들어갑니다. 단, **표 셀 안의 수식은 LaTeX로 변환되지 않고** 셀 텍스트 그대로 HTML에 들어갑니다 (논문 Fig.7과 다른 점). 수식 모델은 이미지에서 LaTeX를 생성하므로 틀린 수식이 나올 수 있어, `.md` 파일로 샘플을 확인하는 것을 권장합니다.
3. **파싱 규칙은 DNV-RU-SHIP 체계 기준**입니다 (Pt.2 Materials and welding 553쪽으로 확인).
   - PDF 하나에 여러 Chapter가 합쳐져 있어, **PDF 책갈피**로 페이지별 Chapter를 정하고 `Section ...` 책갈피 구간만 본문으로 씁니다 (표지·Changes·Contents 제외). 책갈피가 없는 PDF는 전체를 본문으로 봅니다.
   - Section 번호가 Chapter마다 다시 시작하므로 `section` 값은 `Pt.2 Ch.1 Section 3 TESTING PROCEDURES`처럼 Chapter를 포함합니다.
   - 계층: `SECTION N` → `1` → `1.1`(Article) → `1.1.1` → `1.1.1.1` / `a)`. `—` 글머리 목록과 Guidance note는 해당 조항 본문에 포함됩니다.
   - 매 페이지 워터마크(`This copy of the document is intended for use by …`), 우측 세로 표기, `---e-n-d---o-f---…` 표시는 제거합니다.
   - 확인 결과: 책갈피의 Article(`1.1` 수준) 642개를 모두 같은 번호·Section·Chapter로 복원했습니다. 단 이 검증은 Docling 대신 PDF 텍스트 라인으로 한 것이므로, **실제 01 실행 후 `chunks.jsonl`과 `.md`를 직접 확인하세요.**
4. 다른 Section/문서의 표를 참조하는 경우(`Sec.2 Table 3`)와 조항 간 상호참조는 처리하지 않습니다 (논문 future work).
5. Top-k에 같은 Article의 상위·하위 chunk가 함께 들어올 수 있습니다 (논문에 처리 방법 언급 없음).

## 비용·시간 참고 (논문 설정 기준)

- 1단계 변환: 처음 실행 시 Docling 모델을 Hugging Face에서 내려받습니다. 레이아웃·표·수식 모델을 모두 돌리므로 **GPU에서 실행**하세요 (CPU에서는 수천 페이지 변환에 매우 오래 걸립니다).
- 3단계 질문 생성: chunk 20,000개 → GPT-4o 호출 20,000회
- 3단계 라벨링: 300만 쌍 cross-encoder 추론 (T4 기준 수십 분 수준)
- 4단계 학습: 논문은 RTX A6000에서 약 12시간. Colab T4에서는 더 오래 걸리므로 처음에는 `n_contents`, `max_steps`를 줄여서 돌려보세요.

## 구조

```
ship-reg-rag/
├── config.yaml
├── shiprag/
│   ├── convert.py    # PDF → Docling JSON/MD (수식 LaTeX)
│   ├── parse.py      # Docling JSON → 계층 chunk + 표 HTML
│   ├── retrieve.py   # bi-encoder 인덱스/검색
│   ├── gpl.py        # GPL 데이터 생성
│   ├── train.py      # MarginMSE domain adaptation
│   ├── generate.py   # LLaMA 답변 생성
│   ├── evaluate.py   # 검색 정확도
│   └── utils.py
├── scripts/          # 01 ~ 05 실행 스크립트
└── data/raw/         # DNV PDF (git에 올리지 않음)
```

## Reference

Han, I.-S., Roh, M.-I., Kong, M.-C. (2026). A RAG-based Q&A system for ship regulations applying domain adaptation. *International Journal of Naval Architecture and Ocean Engineering*, 18, 100735. https://doi.org/10.1016/j.ijnaoe.2025.100735
