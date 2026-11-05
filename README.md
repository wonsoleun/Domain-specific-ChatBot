# Sea-Ya

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
| ⑥ 웹 UI (Sea-Ya) | - | `shiprag/web.py`, `shiprag/static/index.html` | ② 검색 모델 그대로 사용 (CPU), ⑤는 선택 |

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

### 웹 UI (Sea-Ya)

검색 결과를 브라우저에서 채팅처럼 확인하는 웹사이트입니다. 서비스 이름은 **Sea-Ya**이고, 로고는 바다 물결(`~`)을 형상화했습니다. **GPU와 OpenAI 없이 노트북 CPU만으로 동작**하며(답변 생성은 선택), 인터넷이 없어도 검색은 됩니다.

#### 실행

GPU 없는 Windows PC 기준입니다 (프로젝트 폴더의 가상환경 `.venv` 사용).

```powershell
# 처음 한 번: 가상환경 + CPU용 torch + 의존성 설치
python -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -e .

# 처음 한 번: base 인덱스 생성 (CPU로 10,574개 chunk 기준 수십 분)
.venv\Scripts\python scripts\02_build_index.py --model base

# 서버 실행 → 브라우저에서 http://127.0.0.1:8000
.venv\Scripts\python scripts\06_web.py
```

| 옵션 | 설명 |
|---|---|
| (없음) | 검색만. CPU로 충분 |
| `--generate` | LLaMA 3.1 8B 답변 생성까지 (GPU 필요, Colab 등) |
| `--host 0.0.0.0` | 같은 네트워크(와이파이)의 다른 기기에서 `http://<PC IP>:8000`으로 접속 허용 |
| `--port 8080` | 포트 변경 (기본 8000) |

- 서버를 켜 둔 터미널 창은 사이트를 쓰는 동안 열어 두고, 끌 때는 **Ctrl+C**를 누릅니다.
- 기본 주소 `127.0.0.1`은 "이 PC 자신"이라서, 어느 와이파이에서든 서버를 켠 **그 PC의 브라우저**에서는 접속됩니다 (학교 발표용으로 노트북만 있으면 됨).
- 인터넷이 없으면 시작 시 Hugging Face 온라인 확인 때문에 늦어질 수 있으니 `$env:HF_HUB_OFFLINE = "1"`을 먼저 설정하세요. 이때 수식(KaTeX)과 글꼴(Google Fonts)은 CDN에서 불러오지 못해, 수식이 `$$ ... $$` 원문으로 보이고 기본 글꼴로 표시됩니다.

#### 사용한 모델·기술

| 구분 | 사용 | 비고 |
|---|---|---|
| 검색 모델 (base) | `sentence-transformers/msmarco-bert-base-dot-v5` | ②와 동일한 bi-encoder. 질문을 768차원 벡터로 임베딩 후 dot product Top-k |
| 검색 모델 (adapted) | `models/bi-encoder-adapted` | ③·④로 domain adaptation한 모델. 폴더가 있을 때만 선택 가능 |
| 검색 인덱스 | `data/index/{base,adapted}/embeddings.npy` | 10,574 × 768 (`embed_tables: true`, 참조 표 평문 포함) |
| 답변 생성 (선택) | `meta-llama/Llama-3.1-8B-Instruct` 4-bit | `--generate`일 때만 로드. 기본은 꺼짐 |
| 백엔드 | FastAPI + Uvicorn | `shiprag/web.py`. 기존 `Retriever`, `referenced_tables`를 그대로 호출 |
| 프론트엔드 | HTML + CSS + 바닐라 JavaScript (빌드 도구 없음) | `shiprag/static/index.html` 한 파일 |
| 수식 렌더링 | KaTeX 0.16.9 (cdnjs) | chunk 본문의 `$$ LaTeX $$`를 렌더링 |
| 글꼴 | Roboto, Noto Sans KR (Google Fonts) | |
| 대화 기록 저장 | 브라우저 `localStorage` | 서버에는 저장하지 않음. 최근 30개 대화 |

#### 파일 구성

| 파일 | 역할 |
|---|---|
| `scripts/06_web.py` | 실행 스크립트. `config.yaml`을 읽어 앱을 만들고 Uvicorn으로 실행 |
| `shiprag/web.py` | FastAPI 앱 (`create_app`). 인덱스 존재 확인, 처음 요청 시 `Retriever` 로드 후 메모리에 유지, 검색 결과에 참조 표 첨부 |
| `shiprag/static/index.html` | 화면 전체 (CSS·JS 포함). `pyproject.toml`의 package-data로 패키지에 포함 |

#### 화면 구성

| 영역 | 내용 |
|---|---|
| 왼쪽 사이드바 | Sea-Ya 로고, 새 대화, 최근 대화 목록(삭제 가능), 테마(시스템/라이트/다크), 전체 삭제. 접기 가능 |
| 가운데 대화 | 첫 화면은 인사말 + 예시 질문 + 입력창. 질문하면 오른쪽 말풍선(질문)과 Top-k 조항 카드(제목, Section, 미리보기, 노드 번호, 페이지, 점수, 참조 표 개수) |
| 입력창 | 검색 모델 **base / adapted** 선택(인덱스가 없는 쪽은 비활성화), 결과 수 3·5·10개, `--generate`일 때만 "답변 생성" 체크박스. Enter 보내기, Shift+Enter 줄바꿈 |
| 오른쪽 상세 패널 | 카드를 누르면 열림. 출처 경로, 노드·페이지·점수, 조항 전문(수식 렌더링, 본문에 인라인된 Markdown 표는 표로 변환), 참조된 표(HTML) |

- 색감은 Google 스타일(흰 배경, 파랑 `#1a73e8`, 연한 회청색 `#f0f4f9`)이고, 다크 모드와 좁은 화면(사이드바·상세 패널이 겹쳐 열림)을 지원합니다.
- `http://127.0.0.1:8000/?q=질문`으로 열면 바로 검색합니다. `&open=1`을 붙이면 첫 결과의 상세 패널까지 엽니다 (시연용).
- 표 HTML은 허용된 태그(`table`, `tr`, `td` 등)와 `colspan`/`rowspan`만 남기고 삽입합니다.

#### API

| 요청 | 입력 | 응답 |
|---|---|---|
| `GET /` | - | 화면 (`index.html`) |
| `GET /api/status` | - | `{"indexes": {"base": true, "adapted": false}, "generate": false, "top_k": 5, "n_chunks": 10574, ...}` |
| `POST /api/search` | `{"query": "...", "index": "base", "k": 5, "generate": false}` | `{"query", "index", "answer", "results": [{"rank", "score", "title", "doc", "section", "article_no", "node", "page", "content", "tables": [{"caption", "html", "page"}]}]}` |

없는 인덱스를 고르면 404, 잘못된 인덱스 이름은 400, 서버가 `--generate` 없이 실행됐는데 `generate: true`면 400을 돌려줍니다.

#### adapted 모델을 웹에 연결하려면

학습(③·④)은 GPU가 필요하므로 Colab에서 하고, 웹은 노트북 CPU에서 그대로 돌립니다.

1. Colab에서 `03 --step queries` → `03 --step label` → `04_train.py` → `02_build_index.py --model adapted`
2. `models/bi-encoder-adapted/`(모델)와 `data/index/adapted/`(인덱스)를 **둘 다** 내려받아 노트북의 같은 경로에 넣기 (인덱스만 있으면 질문을 임베딩할 모델이 없어 오류)
3. 서버를 다시 켜면 입력창의 adapted 버튼이 활성화되어 base와 같은 질문으로 비교 가능

현재 03 질문 생성은 GPT-4o(OpenAI API)를 사용합니다.

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
- `title`은 출처로 표시. 임베딩되는 것은 `content` + 참조된 표의 평문입니다 (아래 [논문과 다른 점](#논문과-다른-점) 1, `retriever.embed_tables: false`면 논문대로 `content`만)

`tables.jsonl`: `{"doc", "section", "index": "Table 1", "caption", "html", "page"}`. 검색된 chunk 본문에 `Table N`이 있으면 같은 Section의 표, `Sec.2 Table N` / `Ch.2 Sec.1 Table N`이면 해당 Section의 표 HTML이 LLM 입력에 첨부됩니다 (논문 Fig.6). 여러 페이지에 걸친 표는 하나로 합쳐지며(`html`에 `<table>`이 여러 개), Docling이 캡션을 제목으로 잘못 분류한 경우 표 바로 앞 줄의 `Table N ...`을 캡션으로 씁니다. 원문에도 캡션이 없는 표(Guidance note 속 작은 표 등)는 참조될 수 없으므로 `tables.jsonl` 대신 해당 조항 `content`에 Markdown 표로 들어갑니다.

## 논문과 다른 점

1. **표 내용도 임베딩** (`config.yaml`의 `retriever.embed_tables`, 기본 `true`): 논문은 `content`만 임베딩하고 표는 답변 생성 때만 LLM에 첨부합니다. 그런데 DNV 규칙은 답(계수·최소 두께 등)이 표 행에만 있는 경우가 많아서, 예를 들어 "minimum thickness of bottom shell plating"을 검색하면 정답인 `Pt.3 Ch.6 Section 3 [1.1.1]`이 Top-5에 들지 못하고 본문에 같은 단어가 있는 선종별 규칙이 대신 나옵니다 (실제 답 "Keel, bottom, bilge … a = 4.5, b = 0.035"는 Table 1 안에만 있음). 그래서 chunk가 참조하는 표(`referenced_tables`와 같은 규칙)를 행마다 `셀 | 셀 | ...` 평문으로 펼쳐 `content` 뒤에 붙인 텍스트로 임베딩합니다(`parse.embed_text`). 이 텍스트는 2단계 인덱스, 3단계 GPT-4o 질문 생성·cross-encoder 채점, 4단계 학습에 똑같이 쓰입니다.
   - `chunks.jsonl`과 `chunk_id`는 바뀌지 않습니다(1단계를 다시 돌릴 필요 없음). 생성 단계의 LLM 입력도 그대로 `content` + 표 HTML입니다.
   - 현재 DB 기준 10,574개 chunk 중 1,170개에 표가 붙고, 640개 표 중 611개가 어떤 chunk에서든 참조됩니다.
   - 표가 붙으면서 512 token을 넘는 chunk가 약 530개 → 870개로 늘어납니다(단어 수 기준 추정). 넘는 부분(주로 표 뒷부분)은 잘립니다.
   - 표 셀 구분은 Docling 결과를 따르므로 값이 옆 셀 텍스트에 붙어 나오기도 합니다 (예: `Keel, bottom, bilge, and sea chest boundaries 4.5`).
   - 설정을 바꾸면 02(base) → 03 → 04 → 02(adapted)를 다시 실행해야 합니다. 인덱스 `meta.json`에 `embed_tables`가 기록되어, 설정과 다른 base 인덱스로 03 label을 돌리면 에러가 납니다. 이미 만든 `queries.jsonl`은 검사하지 않으므로 직접 지우세요.
2. **Negative mining**: 논문은 생성 질문 × DB 전체를 cross-encoder로 채점합니다(약 6만 × 15만 = 90억 쌍). 이는 현실적으로 불가능하므로, 하위 30% 구간은 **base bi-encoder 점수**로 정하고 cross-encoder는 실제 학습 쌍의 라벨링에만 씁니다. 또 같은 Article의 chunk는 negative에서 제외합니다(정답 텍스트를 포함할 수 있음).
3. **질문 생성 지시문 추가**: 논문의 지시문 3개에 4번 "규정 원문을 읽지 않은 사용자처럼 묻고, 조항·섹션·표 번호(`3.2.2`, `[2.1.1]`, `Section 5`, `Table 1` 등)를 쓰거나 수식·기호를 인용하지 말 것"을 추가했습니다 (`shiprag/gpl.py`의 `QUERY_INSTRUCTIONS`). 논문 지시문만으로 500개 chunk를 시험 생성했을 때 질문 1,500개 중 287개(19.1%)가 "according to rule 3.2.2", "as referenced in section [5.7.2]"처럼 번호나 수식을 포함했습니다. 실제 사용자는 번호를 모르고 묻는데, chunk `content`가 `3.2.2 ...`로 시작하므로 이런 질문은 의미가 아니라 번호 일치만으로 정답이 찾아져 학습 신호가 약해집니다. 지시문을 바꾸면 기존 `queries.jsonl`은 지우고 다시 생성해야 합니다.
4. **수식 → LaTeX**: 논문은 변환 방법을 밝히지 않아, Docling의 formula enrichment(CodeFormula 모델)를 사용합니다. 본문 수식은 `$$ LaTeX $$` 형태로 chunk에 들어갑니다. 단, **표 셀 안의 수식은 LaTeX로 변환되지 않고** 셀 텍스트 그대로 HTML에 들어갑니다 (논문 Fig.7과 다른 점). 수식 모델은 이미지에서 LaTeX를 생성하므로 틀린 수식이 나올 수 있어, `.md` 파일로 샘플을 확인하는 것을 권장합니다.

## 제약

1. **파싱 규칙은 DNV-RU-SHIP 체계 기준**입니다 (Pt.2 Materials and welding 553쪽으로 확인).
   - PDF 하나에 여러 Chapter가 합쳐져 있어, **PDF 책갈피**로 페이지별 Chapter를 정하고 `Section ...` 책갈피 구간만 본문으로 씁니다 (표지·Changes·Contents 제외). 책갈피가 없는 PDF는 전체를 본문으로 봅니다.
   - Section 번호가 Chapter마다 다시 시작하므로 `section` 값은 `Pt.2 Ch.1 Section 3 TESTING PROCEDURES`처럼 Chapter를 포함합니다.
   - 계층: `SECTION N` → `1` → `1.1`(Article) → `1.1.1` → `1.1.1.1` / `a)`. `—` 글머리 목록과 Guidance note는 해당 조항 본문에 포함됩니다.
   - 매 페이지 워터마크(`This copy of the document is intended for use by …`), 우측 세로 표기, `---e-n-d---o-f---…` 표시는 제거합니다. 이 표시가 수식으로 잘못 인식된 LaTeX와 수식 번호만 있는 `(1)` 수식도 버립니다.
   - 확인 결과: 책갈피의 Article(`1.1` 수준) 642개를 모두 같은 번호·Section·Chapter로 복원했습니다. 단 이 검증은 Docling 대신 PDF 텍스트 라인으로 한 것이므로, **실제 01 실행 후 `chunks.jsonl`과 `.md`를 직접 확인하세요.**
2. 다른 Part(PDF)의 표 참조와 조항 간 상호참조(`see [2.3.4]`)는 처리하지 않습니다 (논문 future work).
3. Top-k에 같은 Article의 상위·하위 chunk가 함께 들어올 수 있습니다 (논문에 처리 방법 언급 없음).

## 비용·시간 참고 (논문 설정 기준)

- 1단계 변환: 처음 실행 시 Docling 모델을 Hugging Face에서 내려받습니다. 레이아웃·표·수식 모델을 모두 돌리므로 **GPU에서 실행**하세요 (CPU에서는 수천 페이지 변환에 매우 오래 걸립니다).
- 3단계 질문 생성: chunk 20,000개 → GPT-4o 호출 20,000회
- 3단계 라벨링: 300만 쌍 cross-encoder 추론 (T4 기준 수십 분 수준)
- 4단계 학습: 논문은 RTX A6000에서 약 12시간. Colab T4에서는 더 오래 걸리므로 처음에는 `n_contents`, `max_steps`를 줄여서 돌려보세요.
  - T4(15GB)에서 batch 64를 한 번에 올리면 CUDA OOM이 나므로, `train.micro_batch_size`(기본 16) 단위로 나눠 gradient를 누적합니다. MarginMSE는 샘플별 독립 loss라 결과는 batch 64와 같습니다. 그래도 OOM이면 8로 줄이세요.

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
│   ├── web.py        # 웹 UI(Sea-Ya) 백엔드 (FastAPI)
│   ├── static/       # 웹 UI 화면 (index.html 한 파일)
│   └── utils.py
├── scripts/          # 01 ~ 06 실행 스크립트
└── data/raw/         # DNV PDF (git에 올리지 않음)
```

## Reference

Han, I.-S., Roh, M.-I., Kong, M.-C. (2026). A RAG-based Q&A system for ship regulations applying domain adaptation. *International Journal of Naval Architecture and Ocean Engineering*, 18, 100735. https://doi.org/10.1016/j.ijnaoe.2025.100735
