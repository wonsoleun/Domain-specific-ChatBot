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


## 실행 순서

DNV PDF를 `data/raw/`에 넣고:

```bash
python scripts/01_build_db.py                          # PDF → Docling JSON/MD → chunks.jsonl, tables.jsonl, figures.jsonl + 그림 PNG (끝에 책갈피 대조·ID 유일성 검사)
python scripts/02_build_index.py --model base          # base 모델 인덱스
python scripts/03_generate_gpl_data.py --step queries  # GPT-4o 질문 생성 (중단 후 재실행 시 이어서)
python scripts/03_generate_gpl_data.py --step label    # negative mining + cross-encoder 라벨
python scripts/04_train.py                             # domain adaptation
python scripts/02_build_index.py --model adapted       # adapted 모델 인덱스
python scripts/05_ask.py --query "What is the minimum thickness of bottom plating?"
```

검색만 확인할 때는 `--no-generate`, base 모델과 비교할 때는 `--index base`를 붙입니다. `--query`를 생략하면 대화형 모드로 실행되어 모델을 한 번만 로드합니다.

#### 사용한 모델·기술

| 구분 | 사용 | 비고 |
|---|---|---|
| 검색 모델 (base) | `sentence-transformers/msmarco-bert-base-dot-v5` | ②와 동일한 bi-encoder. 질문을 768차원 벡터로 임베딩 후 dot product Top-k |
| 검색 모델 (adapted) | `models/bi-encoder-adapted` | ③·④로 domain adaptation한 모델. 폴더가 있을 때만 선택 가능 |
| 검색 인덱스 | `data/index/{base,adapted}/embeddings.npy` | chunk 수 × 768 (현재 11,574개, `embed_tables: true`, 참조 표 평문 포함) |
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
| `shiprag/web.py` | FastAPI 앱 (`create_app`). 인덱스 존재 확인, 처음 요청 시 `Retriever` 로드 후 메모리에 유지, 검색 결과에 참조 표와 그림(`figures: [{caption, page, url}]`) 첨부. 그림 PNG는 `/figures/...`로 제공 (`paths.figure_dir`) |
| `shiprag/static/index.html` | 화면 전체 (CSS·JS 포함). `pyproject.toml`의 package-data로 패키지에 포함 |


### 검색 성능 비교 (선택)

직접 만든 테스트셋(JSONL)으로 base vs adapted를 비교합니다.

```json
{"query": "...", "doc": "DNV-RU-SHIP-Pt2", "section": "Pt.2 Ch.1 Section 3 TESTING PROCEDURES", "article": "3.1"}
```

`doc` / `section` / `article` 값은 `chunks.jsonl`의 `doc` / `section` / `article_no`를 그대로 복사하세요 (DNV는 Section마다 번호가 다시 시작하므로 section까지 있어야 Article이 구분됩니다).

```bash
python scripts/eval_retrieval.py --testset data/test.jsonl
```

### 수식 보정 확인 (선택)

지정한 쪽만 `fix_formula_bbox` 끄고/켜고 두 번 변환해, 수식별 LaTeX를 출력하고 Docling 이미지 내보내기(`get_image()`)로 수식 영역을 `data/diag_formula/off|on/pNNNN_kk.png`에 저장합니다. CodeFormula는 이 영역을 사방 18%씩 더 넓혀 봅니다.

```bash
python scripts/diag_formula.py --pdf "data/raw/DNV Rule SHIP Part 3 Hull - 2026_S.pdf" --pages 12 13 66
```

이 설정을 바꾸거나 Docling을 업데이트한 뒤에는 `data/processed/docling/<PDF이름>/` 폴더를 지우고 01부터 다시 실행해야 합니다. 이때 chunk가 달라지므로 02 인덱스도 다시 만들고, chunk 수가 바뀌었으면 `queries.jsonl`도 지운 뒤 03/04를 다시 돌리세요.

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
 "content": "3.1.7 Reduction of area The reduction of area at fracture Z (%) ... $$ Z = ... $$ ...", "page": 30,
 "ref_id": "Pt.2 Ch.1 Sec.3 [3.1.7]",
 "refs": [{"type": "article", "text": "[3.1.2]", "target": "Pt.2 Ch.1 Sec.3 [3.1.2]", "in_corpus": true, "resolved": true},
          {"type": "table", "text": "Table 2", "target": "Pt.2 Ch.1 Sec.3 Table 2", "in_corpus": true, "resolved": true},
          {"type": "external", "text": "ISO 6892-1", "target": "ISO 6892-1", "in_corpus": false, "resolved": false}],
 "formulas": ["Z = ..."], "table_ids": ["Pt.2 Ch.1 Sec.3 Table 2"], "figure_ids": []}
```

- `node == article_no`이면 Article 전체 chunk, 그 외는 nested chunk (논문 Fig.3~5)
- `title`은 출처로 표시. 임베딩되는 것은 `content` + 참조된 표의 평문입니다 (아래 [논문과 다른 점](#논문과-다른-점) 1, `retriever.embed_tables: false`면 논문대로 `content`만)
- **`ref_id`** (`parse.ref_id`): 정규화 조항 ID `Pt.N Ch.N Sec.N [번호]`. DNV 본문 참조 표기와 같은 형식이라, 질문에 조항 번호가 있으면 검색 없이 바로 찾을 수 있습니다. 모든 chunk에서 유일합니다(01의 고정 검사). 하위 항목은 `[3.1.7.a]`, 한 항목 안에 a) 목록이 두 번 나오면(원문 그대로, 예: Pt.3 Ch.11 Sec.2 [5.7.3]) 두 번째 목록은 `[5.7.3.a(2)]`입니다. 질문에 `[5.1]`만 있으면 Section마다 있으므로 Pt/Ch/Sec 없이는 하나로 정해지지 않습니다.
- **`refs`** (`parse.extract_refs`): 본문의 참조를 대상 단위별로 정규화한 목록. `type`은 `part` / `chapter` / `section` / `subsection`(`[14]`처럼 한 자리 번호) / `article`(`[3.1]`) / `paragraph`(`[3.1.2]` 이하) / `table` / `figure` / `appendix` / `external`입니다. 생략된 Pt·Ch·Sec는 현재 위치로 채우고, 위가 주어지고 중간이 빠진 참조(`Ch.2 Table 1`)는 `target: null`입니다. 외부 문서(DNV-CG/ST/OS, SOLAS·MARPOL·IGC Code 등 IMO 규범, IACS UR/UI, ISO/EN 등)는 그 안의 위치까지 묶어(`DNV-CG-0128 Sec.3 [3.2.3]`, `IGC Code Ch. 16.7.1.1`) `external` 하나로 저장합니다. 이렇게 먼저 잘라내지 않으면 `Sec.3 [3.2.3]`이 현재 Chapter의 조항으로 잘못 해석됩니다. 수식(`$$…$$`) 안은 보지 않습니다. `in_corpus`는 대상 Part가 파싱한 PDF에 있는지(Pt.1/4/6/7 참조는 false), `resolved`는 실제 chunk·표·그림을 찾았는지입니다. 현재 Article chunk 기준 코퍼스 안 참조의 해석률은 article 99%, paragraph 99.6%, table 98%, figure 94%(캡션을 못 찾은 그림)입니다.
- **`formulas` / `table_ids` / `figure_ids`** (`parse.annotate_chunks`): 이 chunk에 딸린 수식(LaTeX), 표·그림 ID(`tables.jsonl`·`figures.jsonl`의 `id`). 표·그림은 ① chunk 범위 안에 놓인 것(`node`가 같거나 하위, Article chunk면 Article 전체) + ② `refs`에서 해석된 것입니다. 웹 UI는 이 목록으로 표·그림을 보여 줍니다. 임베딩과 LLM 입력은 논문대로 본문 참조 표(`referenced_tables`)만 씁니다.

`tables.jsonl`: `{"id": "Pt.2 Ch.1 Sec.3 Table 1", "doc", "section", "index": "Table 1", "caption", "html", "page", "article_no", "node"}`. `article_no`/`node`는 표가 놓인 위치입니다(Section 첫 Article 앞의 기호표처럼 조항 밖이면 빈 값). 검색된 chunk 본문에 `Table N`이 있으면 같은 Section의 표, `Sec.2 Table N` / `Ch.2 Sec.1 Table N`이면 해당 Section의 표 HTML이 LLM 입력에 첨부됩니다 (논문 Fig.6). 여러 페이지에 걸친 표는 하나로 합쳐지며(`html`에 `<table>`이 여러 개), Docling이 캡션을 제목으로 잘못 분류한 경우 표 바로 앞 줄의 `Table N ...`을 캡션으로 씁니다. 원문에도 캡션이 없는 표(Guidance note 속 작은 표 등)는 참조될 수 없으므로 `tables.jsonl` 대신 해당 조항 `content`에 Markdown 표로 들어갑니다.

`figures.jsonl`: `{"id", "doc", "section", "index": "Figure 1", "caption", "page", "bbox", "article_no", "node", "file"}`와 PNG 파일(`data/processed/figures/<PDF이름>/pNNNN_kk.png`). 웹 UI 표시용이며 임베딩·LLM 입력에는 쓰지 않습니다 (그림 캡션 문장은 원래대로 `content`에 들어 있음).
- **이미지는 PDF 페이지를 잘라서 만듭니다** (`shiprag/figures.py`의 `render_figures`, `parse.figure_scale` 배율). DNV 그림은 벡터 path로 그린 것과 비트맵이 섞여 있어(497개 중 벡터 283개) PDF에 붙은 이미지 객체를 꺼내면 절반 넘게 빠지기 때문입니다. Docling JSON의 picture bbox만 쓰므로 재변환이 필요 없고, 01을 다시 실행하면 그림만 새로 만들어집니다.
- 가로·세로 모두 30pt 미만인 picture와 표지·Contents 페이지의 picture는 제외합니다 (Part 2/3/5 합계 455개).
- 캡션은 Docling이 연결한 캡션 → 그림 바로 앞 줄 → 바로 다음 줄의 `Figure N ...` 순으로 찾습니다. 못 찾으면 `index`가 `p34-figure2` 같은 이름이 되어 `Figure N` 참조로는 찾을 수 없고, 아래의 위치 기준으로만 표시됩니다.
- chunk에 붙는 그림 (`parse.chunk_figures`): ① 그 chunk 범위 안에 놓인 그림(`node`가 같거나 하위, Article chunk면 Article 전체), ② 본문의 `Figure N` / `Sec.N Figure N` / `Ch.N Sec.N Figure N` 참조(표 참조와 같은 규칙). 그림 하나가 picture 여러 개(좌우 조각, 다음 페이지로 이어진 부분)로 나뉘어 같은 캡션을 공유하면 모두 표시합니다.

## 논문과 다른 점

1. **표 내용도 임베딩** (`config.yaml`의 `retriever.embed_tables`, 기본 `true`): 논문은 `content`만 임베딩하고 표는 답변 생성 때만 LLM에 첨부합니다. 그런데 DNV 규칙은 답(계수·최소 두께 등)이 표 행에만 있는 경우가 많아서, 예를 들어 "minimum thickness of bottom shell plating"을 검색하면 정답인 `Pt.3 Ch.6 Section 3 [1.1.1]`이 Top-5에 들지 못하고 본문에 같은 단어가 있는 선종별 규칙이 대신 나옵니다 (실제 답 "Keel, bottom, bilge … a = 4.5, b = 0.035"는 Table 1 안에만 있음). 그래서 chunk가 참조하는 표(`referenced_tables`와 같은 규칙)를 행마다 `셀 | 셀 | ...` 평문으로 펼쳐 `content` 뒤에 붙인 텍스트로 임베딩합니다(`parse.embed_text`). 이 텍스트는 2단계 인덱스, 3단계 GPT-4o 질문 생성·cross-encoder 채점, 4단계 학습에 똑같이 쓰입니다.
   - `chunks.jsonl`과 `chunk_id`는 바뀌지 않습니다(1단계를 다시 돌릴 필요 없음). 생성 단계의 LLM 입력도 그대로 `content` + 표 HTML입니다.
   - 현재 DB 기준 11,574개 chunk 중 1,216개에 표가 붙고, 642개 표 중 616개가 어떤 chunk에서든 참조됩니다. (웹 UI의 `table_ids`는 놓인 위치로도 연결하므로 Section 첫 Article 앞의 기호표 9개를 빼고 모두 연결됩니다.)
   - 표가 붙으면서 512 token을 넘는 chunk가 약 530개 → 870개로 늘어납니다(단어 수 기준 추정). 넘는 부분(주로 표 뒷부분)은 잘립니다.
   - 표 셀 구분은 Docling 결과를 따르므로 값이 옆 셀 텍스트에 붙어 나오기도 합니다 (예: `Keel, bottom, bilge, and sea chest boundaries 4.5`).
   - 설정을 바꾸면 02(base) → 03 → 04 → 02(adapted)를 다시 실행해야 합니다. 인덱스 `meta.json`에 `embed_tables`가 기록되어, 설정과 다른 base 인덱스로 03 label을 돌리면 에러가 납니다. 이미 만든 `queries.jsonl`은 검사하지 않으므로 직접 지우세요.
2. **Negative mining**: 논문은 생성 질문 × DB 전체를 cross-encoder로 채점합니다(약 6만 × 15만 = 90억 쌍). 이는 현실적으로 불가능하므로, 하위 30% 구간은 **base bi-encoder 점수**로 정하고 cross-encoder는 실제 학습 쌍의 라벨링에만 씁니다. 또 같은 Article의 chunk는 negative에서 제외합니다(정답 텍스트를 포함할 수 있음).
3. **질문 생성 지시문 추가**: 논문의 지시문 3개에 4번 "규정 원문을 읽지 않은 사용자처럼 묻고, 조항·섹션·표 번호(`3.2.2`, `[2.1.1]`, `Section 5`, `Table 1` 등)를 쓰거나 수식·기호를 인용하지 말 것"을 추가했습니다 (`shiprag/gpl.py`의 `QUERY_INSTRUCTIONS`). 논문 지시문만으로 500개 chunk를 시험 생성했을 때 질문 1,500개 중 287개(19.1%)가 "according to rule 3.2.2", "as referenced in section [5.7.2]"처럼 번호나 수식을 포함했습니다. 실제 사용자는 번호를 모르고 묻는데, chunk `content`가 `3.2.2 ...`로 시작하므로 이런 질문은 의미가 아니라 번호 일치만으로 정답이 찾아져 학습 신호가 약해집니다. 지시문을 바꾸면 기존 `queries.jsonl`은 지우고 다시 생성해야 합니다.
4. **수식 → LaTeX**: 논문은 변환 방법을 밝히지 않아, Docling의 formula enrichment(CodeFormula 모델)를 사용합니다. 본문 수식은 `$$ LaTeX $$` 형태로 chunk에 들어갑니다. 단, **표 셀 안의 수식은 LaTeX로 변환되지 않고** 셀 텍스트 그대로 HTML에 들어갑니다 (논문 Fig.7과 다른 점). 수식 모델은 이미지에서 LaTeX를 생성하므로 틀린 수식이 나올 수 있어, `.md` 파일로 샘플을 확인하는 것을 권장합니다.
5. **수식 영역 보정** (`convert.fix_formula_bbox`, 기본 켜짐): DNV PDF는 수식 본체를 글자가 아닌 그림(이미지 객체 또는 벡터 path)으로 넣는 경우가 많습니다. 예: Pt.3 Ch.1 Sec.2 [5.1]의 WSD 수식 `W_stat ≤ η_1 R`(p.12), Pt.3 Ch.2의 bow height `F_b` 수식(p.66). Docling layout 후처리(`layout_postprocessor._adjust_cluster_bboxes`)는 수식 bbox를 그 안의 PDF 텍스트 셀 크기로 줄입니다. 그래서 본체가 잘리고 꼬리 텍스트 `for condition A.`만 CodeFormula에 들어가 `$$ R \text { for condition } A . $$`처럼 수식 본체가 사라집니다. Docling의 처리 흐름(레이아웃 검출·후처리·읽기 순서)은 그대로 두고, `shiprag/convert.py`의 `ShipRagPdfPipeline`이 CodeFormula 모델을 `FormulaBBoxFix`로 감싸 수식 이미지를 자르기 직전(`prepare_element`)에 bbox를 보정합니다.
   1. **후보 수집**: 줄어든 bbox 중심 위아래 80pt 띠 안에서 PDF 이미지 객체(`get_bitmap_rects()`)와, 페이지 이미지(144 dpi)의 잉크 연결 요소를 모읍니다. 잉크 기준 경계는 Shah et al.(2021), Mali et al.(2020)의 후처리 방식을 참고했고, 벡터 path로 그린 수식도 여기서 잡힙니다.
   2. **걸러내기**: 높이 60pt 초과, 페이지 폭 80% 초과, 띠 밖으로 이어지는 그래픽(실제 그림·다른 줄)과 다른 항목의 텍스트 셀에 겹치는 잉크는 뺍니다. 인접한 줄의 수식이 합쳐지는 것이 두 선행 연구의 주요 오류였으므로 기준을 보수적으로 잡았습니다.
   3. **합치기**: 꼬리 텍스트와 같은 줄에서 가로 15pt 안으로 이어지는 후보, 분수처럼 바로 위·아래 1.5pt 안에 붙은 후보를 하나의 수식 블록으로 합칩니다 (Hu et al. 2024).
   4. **자르기**: 넓힌 영역을 `prov` bbox에 기록하고(JSON에도 저장), 기존 CodeFormula의 `prepare_element`로 페이지 이미지를 잘라 넣습니다. `parse.py`는 바뀌지 않습니다.
   5. **반복 꼬리 정리**: CodeFormula가 정상 수식 뒤에 같은 내용을 조금씩 바꿔 가며 생성 한도(2,048토큰)까지 반복하는 경우가 있습니다. 예: F_b 수식(p.66) 뒤에 `\stackrel { \infty } { \infty } \cdots`가 4,000자 넘게 이어짐. Docling의 `TailRepetitionStopper`도 켜져 있지만, 64자 이하 단위가 16번 이상 *똑같이* 반복될 때만 잡아서 이 경우(반복 단위 ~400자)는 놓칩니다. 그래서 `FormulaBBoxFix`가 모델 출력을 받은 뒤 `strip_formula_loop`로 한 번 더 정리합니다. 2,000자 이상인 출력에서 24토큰 구간이 출력 끝까지 3번 이상 반복되면, 모든 반복에 공통인 범위까지 자릅니다. 그다음 닫히지 않은 그룹, 인자가 모자란 명령, 끝의 `\quad` 같은 공백 명령을 걷어냅니다. 기존 chunk의 수식 1,251개에 적용해 보니 정상 수식은 바뀌지 않았고(길이 조건이 없으면 Hog/Sag처럼 같은 구조가 되풀이되는 463·805자 정상 수식을 잘못 자름), 루프 2건만 잘렸습니다. 단, 정상 수식 바로 뒤에 반복되지 않는 쓰레기가 먼저 오고 짧은 쓰레기가 반복되는 형태는 그 앞부분 쓰레기가 남습니다.

   보정 전후는 `scripts/diag_formula.py`로 비교합니다 (위 [수식 보정 확인](#수식-보정-확인-선택)).
6. **수식 읽기 순서 보정** (`shiprag/parse.py`의 `_reorder_formulas`): Docling의 읽기 순서(`ReadingOrderPredictor`)는 위 5의 보정 *전*, 즉 꼬리 텍스트(`for condition B.`)로 줄어든 수식 bbox로 정해집니다. 이 bbox는 줄 오른쪽에 치우쳐 위아래 본문과 가로로 겹치지 않으므로, 수식이 별도 단처럼 취급되어 **페이지 본문 맨 뒤로 밀립니다**. 예: Pt.3 Ch.1 Sec.2 p.13 맨 위의 [5.1] 수식 3개(WSD condition B, PSF condition A·B)가 같은 페이지 끝의 [5.3.1]에 붙음. Pt.5 Ch.10 Sec.10 p.94의 프로펠러 하중 F_b 수식이 [10.2.1]이 아니라 [10.5] Steering system에 붙음. 5의 보정을 읽기 순서 앞으로 옮겨도 이 bbox가 본문 단 왼쪽 끝까지 닿지 않으면 페이지 맨 위 수식은 여전히 어긋나므로(Docling 규칙 기반 읽기 순서로 재현해 확인), 재변환 없이 파싱 단계에서 바로잡습니다. DNV 본문은 1단이므로 각 수식을 같은 페이지에서 수식 세로 중심보다 아래에서 시작하는 첫 항목 바로 앞에 넣고, 수식이 아닌 항목끼리의 순서는 바꾸지 않습니다. 기존 JSON에 적용한 결과 chunk 수와 `(doc, node)`는 그대로이고, 수식 1,342개 중 30개가 다른 조항(모두 같은 Section 안의 앞 조항)으로 옮겨졌으며, chunk 217개의 content가 바뀌었습니다.

## 제약

1. **파싱 규칙은 DNV-RU-SHIP 체계 기준**입니다 (Pt.2 Materials and welding 553쪽으로 확인).
   - PDF 하나에 여러 Chapter가 합쳐져 있어, **PDF 책갈피**로 페이지별 Chapter를 정하고 `Section ...` 책갈피 구간만 본문으로 씁니다 (표지·Changes·Contents 제외). 책갈피가 없는 PDF는 전체를 본문으로 봅니다.
   - Section 번호가 Chapter마다 다시 시작하므로 `section` 값은 `Pt.2 Ch.1 Section 3 TESTING PROCEDURES`처럼 Chapter를 포함합니다.
   - 계층: `SECTION N` → `1` → `1.1`(Article) → `1.1.1` → `1.1.1.1` / `a)`. `—` 글머리 목록과 Guidance note는 해당 조항 본문에 포함됩니다.
   - 매 페이지 워터마크(`This copy of the document is intended for use by …`), 우측 세로 표기, `---e-n-d---o-f---…` 표시는 제거합니다. 이 표시가 수식으로 잘못 인식된 LaTeX와 수식 번호만 있는 `(1)` 수식도 버립니다.
   - 번호 제목은 **번호 패턴으로만** 판정합니다: 현재 번호의 올바른 다음 번호(`_is_successor`)이고 번호 뒤가 대문자로 시작하면 제목입니다. 길이 제한은 두지 않습니다. 예전에는 100자 이하만 인정해서, 113자인 Pt.2 Ch.4 Sec.5 "9 Additional considerations …" 하나를 놓치자 이후 9.1~15.3이 Section 끝까지 [8.3.2] 밑으로 들어갔습니다(Part 2·5 합계 Article 48개 누락).
   - IGC 기반 장(Pt.5 Ch.7 등)은 Docling이 run-in 번호 `3.1.1`을 본문과 떼어 번호만 있는 줄로 냅니다. Paragraph 이하 깊이이고 올바른 다음 번호이면 그 줄로 노드를 열고 다음 줄부터 본문을 붙입니다 (약 830개 Paragraph 복원).
   - **고정 검사** (`shiprag/checks.py`, 01 끝에서 자동 실행): ① PDF 책갈피의 Article(`1.1` 수준, Part 2/3/5 합계 3,004개)이 chunks에 모두 있고 책갈피에 없는 Article이 생기지 않았는지, ② `ref_id`가 모두 유일한지. 실패하면 목록을 출력하고 종료 코드 1로 끝납니다(결과 파일은 확인용으로 남음). 파서를 고친 뒤에는 이 검사가 통과하는지 먼저 보세요.
2. 다른 Part(PDF)의 표 참조와 조항 간 상호참조(`see [2.3.4]`)는 처리하지 않습니다 (논문 future work).
3. Top-k에 같은 Article의 상위·하위 chunk가 함께 들어올 수 있습니다 (논문에 처리 방법 언급 없음).

## 비용·시간 참고 (논문 설정 기준)

- 1단계 변환: 처음 실행 시 Docling 모델을 Hugging Face에서 내려받습니다. 레이아웃·표·수식 모델을 모두 돌리므로 **GPU에서 실행**하세요 (CPU에서는 수천 페이지 변환에 매우 오래 걸립니다).
  - 수식 인식은 `convert.formula_batch_size`개(기본 2, Docling 기본은 5)씩 묶어 생성합니다. 묶음 중 하나가 반복 루프에 빠지면 묶음 전체가 2,048토큰까지 생성해, T4에서 `CUDACachingAllocator ... OOM` 경고가 자주 납니다. 그래도 잦으면 1로 줄이세요. 생성은 greedy라 묶음 크기를 바꿔도 결과가 거의 같으므로(패딩 때문에 아주 드물게 다를 수 있음), 이미 변환한 구간을 다시 돌릴 필요는 없습니다.
- 3단계 질문 생성: chunk 20,000개 → GPT-4o 호출 20,000회
- 3단계 라벨링: 300만 쌍 cross-encoder 추론 (T4 기준 수십 분 수준)
- 4단계 학습: 논문은 RTX A6000에서 약 12시간. Colab T4에서는 더 오래 걸리므로 처음에는 `n_contents`, `max_steps`를 줄여서 돌려보세요.
  - T4(15GB)에서 batch 64를 한 번에 올리면 CUDA OOM이 나므로, `train.micro_batch_size`(기본 16) 단위로 나눠 gradient를 누적합니다. MarginMSE는 샘플별 독립 loss라 결과는 batch 64와 같습니다. 그래도 OOM이면 8로 줄이세요.

## 구조

```
ship-reg-rag/
├── config.yaml
├── shiprag/
│   ├── convert.py    # PDF → Docling JSON/MD (수식 LaTeX, 수식 영역 보정)
│   ├── parse.py      # Docling JSON → 계층 chunk + 표 HTML + 그림 목록
│   ├── figures.py    # 그림 PNG 저장 (PDF 페이지를 bbox대로 잘라냄)
│   ├── checks.py     # 01 고정 검사 (책갈피 대조, ref_id 유일성)
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
