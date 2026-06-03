# 💊 의약품 RAG QnA 시스템

> 증상이나 약 이름을 입력하면, 사용자 상황(나이·임신·복용약 등)을 역질문으로 확인한 뒤
> **안전한 상비약을 추천**하는 RAG 기반 챗봇 + **응답 신뢰성 평가 파이프라인**

RaGoon 팀 프로젝트 · 식약처 e약은요 데이터 기반

---

## 📑 목차

1. [프로젝트 소개](#1-프로젝트-소개)
2. [구현 방법](#2-구현-방법)
3. [사용 방법](#3-사용-방법)
4. [시연](#4-시연)
5. [응답 신뢰성 평가](#5-응답-신뢰성-평가)
6. [부록 · 스터디 기록](#6-부록--스터디-기록)

---

## 1. 프로젝트 소개

### 배경 & 목적

전문가의 도움을 즉시 받기 어려운 상황에서, **상비약 선택을 돕는 보조 수단**이 필요합니다.
단, 의약품은 안전성이 핵심이므로 "그럴듯한 답"이 아니라 **사용자 상황에 맞는 안전한 답**을 주는 것,
그리고 그 답이 **신뢰할 수 있는지 정량적으로 검증**하는 것을 목표로 했습니다.

> ⚠️ 본 시스템은 약사·의사를 대체하지 않으며, 정보 제공·보조 목적의 프로토타입입니다.

### 핵심 특징

- **상황 기반 역질문**: 나이·임신 여부·복용 중인 약 등을 되물어 위험을 거른 뒤 추천
- **2가지 대화 흐름**: 사용자 입력 형태에 따라 자동 분기
  - **흐름 A (프롬프트 기반)** — "두통이 있어요"처럼 **증상만 입력** → 벡터 유사도 검색으로 약 탐색 → LLM이 규칙에 따라 자율 역질문
  - **흐름 B (코드 기반)** — "타이레놀 먹어도 돼?"처럼 **약 이름 명시** → 해당 약 문서에서 금기 슬롯을 동적 생성 → 슬롯별 역질문
- **금기·안전 경고 자동 안내**: 임산부·소아·음주·상호작용 등 위험 상황에서 복용 불가/상담 안내
- **응답 신뢰성 평가 파이프라인**: 시나리오 자동 실행 → RAGAS 9개 메트릭 + 결정론적 금기 검증

---

## 2. 구현 방법

### 기술 스택

| 구분 | 사용 기술 |
| --- | --- |
| 챗봇 LLM | OpenAI **gpt-5-mini** |
| 검색 임베딩 | **BGE-M3** (BAAI, 한국어·다국어) |
| 벡터 DB | **ChromaDB** (로컬·영구 저장) |
| 오케스트레이션 | **LangChain** |
| 백엔드 | **FastAPI** (Python) |
| 프론트엔드 | **React + TypeScript** |
| 평가 | **RAGAS** + 평가 LLM gpt-4o-mini |
| 데이터 | 식약처 **e약은요** API (경구약 82개 품목) |

### 디렉토리 구조

```
project-docs/
├─ develop/
│  ├─ code/                 # 챗봇 핵심 로직
│  │  ├─ rag_qna_multi.py   #   메인 챗봇 (MedicalChatbot) · 흐름 A/B 분기
│  │  ├─ drug_detector.py   #   입력에서 약 이름 탐지
│  │  ├─ caution_parser.py  #   약 문서 → 금기 슬롯 동적 생성 (흐름 B)
│  │  ├─ dialogue.py        #   대화 상태(DialogueState)
│  │  ├─ validator.py       #   복용 가부 판정
│  │  ├─ api_server.py      #   FastAPI 서버 (포트 8000)
│  │  └─ prompts/           #   시스템 프롬프트
│  ├─ front/                # React + TS 웹 UI (포트 3000)
│  ├─ dataset/              # 약 문서 데이터 (drug_documents.json 등)
│  ├─ evaluation/           # 신뢰성 평가 파이프라인
│  │  ├─ scenarios.py       #   평가 시나리오 정의
│  │  ├─ run_chatbot.py     #   시나리오 자동 실행 → 챗봇 출력 수집
│  │  ├─ run_ragas_single.py#   RAGAS 평가
│  │  └─ REPORT.md          #   평가 결과 보고서
│  └─ ARCHITECTURE.md       # 코드 흐름 상세 문서
└─ chroma_db/               # 벡터 DB 저장소
```

### 동작 흐름

```
사용자 입력
   │
   ├─ 약 이름 탐지 → 의도 분석(증상만 / 약 복용 가부 / 약 비교) → 상황 추출
   │
   ├─ [흐름 A] 증상만 입력
   │      → 벡터 유사도 검색(BGE-M3 + ChromaDB)으로 후보 약 탐색
   │      → 프롬프트 규칙에 따라 LLM이 나이·임신·복용약 역질문
   │      → 상황 반영해 안전한 약 추천
   │
   └─ [흐름 B] 약 이름 명시
          → 해당 약 문서에서 금기 슬롯 동적 생성 (caution_parser)
          → 위험도 우선순위로 슬롯별 역질문
          → 금기 판정 후 복용 가부 안내 (가능 시 추천, 불가 시 거부/상담)
```

자세한 내부 흐름은 [`develop/ARCHITECTURE.md`](./develop/ARCHITECTURE.md) 참고.

---

## 3. 사용 방법

### 사전 준비

- Python 3.10+, Node.js 18+
- 프로젝트 루트에 `.env` 파일 생성 후 OpenAI 키 설정
  ```
  OPENAI_API_KEY=sk-...
  ```
- 벡터 DB(`chroma_db/`)가 포함되어 있습니다. (재구축이 필요하면 `develop/code/build_documents.py` 참고)

### ① 웹으로 실행 (권장)

프론트와 API 서버를 한 번에 실행합니다.

```bash
cd develop/front
npm install
npm start
```

- 웹 UI: http://localhost:3000
- API 서버: http://localhost:8000 (FastAPI, `/api/chat`)
- `prestart`가 백엔드 의존성(`requirements-api.txt`)을 자동 설치합니다.

증상이나 약 이름을 입력하면 대화가 시작되고, 역질문이 나오면 **예/아니요 버튼**으로 답할 수 있습니다.

### ② CLI로 실행

```bash
cd develop/code
python rag_qna_multi.py
```

### ③ API 서버만 실행

```bash
cd develop/code
pip install -r requirements-api.txt
python api_server.py        # http://localhost:8000
```

---

## 4. 시연

> 📸 시연 이미지/영상을 `docs/` 폴더에 넣고 아래 경로를 연결하세요.

### 웹 화면

<img width="550" height="700" alt="image" src="https://github.com/user-attachments/assets/fada384b-a31f-438c-81aa-3512035c1cb9" />

### 대화 예시 (역질문 → 추천)

https://github.com/user-attachments/assets/ca170456-f318-4642-9bbe-0213f520598f


**예시 시나리오**

```
🙋 두통이 있어요
🤖 나이가 어떻게 되세요?
🙋 30
🤖 임신 중이신가요?
🙋 아니요
🤖 현재 복용 중인 다른 약이 있으신가요?
🙋 아니요
🤖 💊 추천 약: 이지엔6이브 / 부루펜 …
   📌 이유 · ⚠️ 주의사항 · 🏥 병원 권장 안내
```

---

## 5. 응답 신뢰성 평가

시나리오 기반으로 챗봇 응답의 신뢰성을 정량 측정합니다.

```bash
cd develop/evaluation

# 1) 시나리오 자동 실행 → 챗봇 응답·검색문서 수집
python run_chatbot.py

# 2) RAGAS 평가 (9개 메트릭)
python run_ragas_single.py
```

- **메트릭(9개)**: RAG 표준 4개(Faithfulness·Answer Relevancy·Context Precision·Context Recall)
  + 추가 2개(Factual Correctness·Noise Sensitivity)
  + 커스텀 Aspect Critic 3개(약 추천 정확성·안전 경고 포함·금기 약 회피)
- **결정론적 검증**: 금기 회피처럼 정답이 명확한 안전 항목은 LLM 채점 대신 규칙 기반으로 재확인
- 결과·분석은 [`develop/evaluation/REPORT.md`](./develop/evaluation/REPORT.md) 참고

---

## 6. 부록 · 스터디 기록

주차별 팀 스터디·개발 기록입니다. 각 주차 폴더에 팀원별 실습 노트(`*.ipynb`)와 코드가 정리되어 있습니다.

| 주차 | 주제 | 주요 내용 |
| --- | --- | --- |
| [week06](./study/week06/) | 기초 학습 & 시나리오 설계 | 임베딩·벡터DB 실습, LangChain·LLM 오케스트레이션 테스트, OpenAI API 실습, 의약품 시나리오 초안 작성 |
| [week07](./study/week07/) | QnA 시스템 1차 구현 & 데이터 구축 | e약은요 API로 약 데이터 수집·문서화(`fetch_drug_data`, `build_documents`), RAG QnA 구현, 프롬프트·시나리오 보강 |
| [week10](./study/week10/) | 모듈화 & 역질문 고도화 | 기능별 모듈 분리(`drug_detector`·`caution_parser`·`clarifier`·`validator`·`dialogue` 등), 역질문 흐름 고도화, 흐름 A/B 통합 |

> 각 주차 폴더는 팀원별 작업 공간(지민·진우·하나·한림·현주)으로 나뉘어 있으며, 통합 결과물은 [`develop/`](./develop/)에 반영되어 있습니다.
