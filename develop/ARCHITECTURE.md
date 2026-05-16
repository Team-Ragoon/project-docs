# 의약품 RAG QnA 시스템 구조 문서

> 코드의 전반적인 기능 및 흐름, 각 file의 역할 및 연관성을 작성해놓은 file입니다.<br> 개발하시면서 수정되는 흐름 , 내용 /  문서로 남겨 놓으면 좋을 내용(추가 기능, prompt 수정 사항)은 추가해주세요!

---

# 현재 확인용 출력을 위한 코드가 있는 위치
```
(rag_qna_multi.py)
- retriever_multi()에서 유사도 순 정렬한 5개의 문서에 대한 확인 출력
- MedicalChatbot -> chat()에서 caution_slots의 현 상황 출력(역질문), analysis 출력, drug_names(탐지한 약 이름들) 출력,
  query 입력 후에 caution_slots, extra_context 출력.

(caution_parser.py)
- parse_contraindications_for_drugs()에서 역질문 목록 출력.
```

# 현재 코드에서 수정을 고려해야 하는 요소
```
- 클래스 DialogueState에서 history를 사용할 것인지. 
(지금 코드에서는 역질문에 대한 답을 따로 저장하고 반영 -> history 필요 없음)
(다만, 필수 역질문 등을 구현할 때 사용될 여지가 있으므로 삭제는 하지 않음.)

```

# 현재 코드에서 구현해야 하는 기능
```
- 흐름 A에서 필수 역질문(약 종류에 따라 조금씩 달라질 듯. "임산부" 정보만 고정해두면 될 듯). 
- 흐름 B에서 약이 다양할 때 그 중 한 개를 추천하기.(사용자의 caution_slots 및 상황에 따라서)
```


# 전체 흐름

## 흐름 A — 증상만 입력
{05.14 - 증상만 입력 -> 약 추천으로 진행. (시나리오 테스트 필요, 필수 역질문하여 추천하도록 기능 추가해야 함.)}
```
사용자 입력
  -> 의도 분석 : analyzer.py
  -> 답변 생성 : rag.py · build_rag_chain
```

## 흐름 B — 약 이름 포함
{05.14 - 약 이름이 포함되면 금기 사항 기반 역질문 진행 => 약 복용 가능 여부를 판단.}
{05.16 - query에 금기 사항이 존재 -> 역질문 진행 X, 바로 복용 불가능함을 안내함}
{05.16 - 금기 사항 안에서 우선순위 결정 -> 우선순위별로 최대 3개의 역질문 진행하도록 함. 이때, 질문하지 않은 금기 사항에 대해 최종 답변에서 언급하여 주의할 수 있도록 함.}
```
사용자 입력
  -> 의도 분석 : analyzer.py
  -> query 내에 포함된 약 이름을 DB에서 찾기: drug_detector.py
  -> 찾은 약들의 금기 사항 parsing해서 역질문 목록 만들기 및 우선 순위 결정하기 : caution_parser.py
  -> 역질문 횟수 확인 후 역질문 진행:  validator.py
      ├── 슬롯(금기 사항) 미완료 → 해당 slot에 대한 자연스러운 문장의 역질문 생성 : clarifier.py
      │                → answer_parser.py (예/아니오 파싱) → 반복
      └── 슬롯(금기 사항) 모두 완료   → rag.py · build_recommend_final_chain / build_cannot_recommend_chain
  → 최종 답변 출력
```

---

## 파일별 역할

| 파일 | 역할 | 비고 |
|------|------|------|
| `dialogue.py` | 대화 상태 전체 관리 (`DialogueState`) | 모든 파일이 의존 |
| `analyzer.py` | query의도 분석 + 상황 사전 추출 | `query_type`, `symptom` 추출 |
| `drug_detector.py` | query 내의 약 이름을 DB에서 탐지 (퍼지 매칭) | `medication_loader` 의존 |
| `medication_loader.py` | JSON DB 로드 + 캐싱 | `lru_cache`로 중복 I/O 방지 |
| `build_documents.py` | 오프라인 전처리 스크립트 | 런타임 무관, 1회 실행용 |
| `caution_parser.py` | 금기 문장 추출 + LLM 구조화 | 내부 `_cache`로 재파싱 방지 |
| `validator.py` | 슬롯 검증 및 역질문 여부 판단 | `MAX_CLARIFY = 5` |
| `clarifier.py` | 역질문 문장 생성 | `validator` → `clarifier` 순서 |
| `answer_parser.py` | 역질문에 대한 사용자의 응답에 대해 긍정/부정 판단 | 1차 키워드 → 2차 LLM |
| `rag.py` | 답변 생성 체인 모음 | 여러 체인 포함 (하단 참고) |
| `system_prompt.py` | 프롬프트 템플릿 및 few-shot 예시 |

---

## rag.py 체인 목록

| 함수 | 용도 | 상태 |
|------|------|------|
| `build_rag_chain` | 증상 입력 → 바로 RAG 추천 | 사용 중 |
| `build_recommend_final_chain` | 슬롯 완료 후 최종 추천 | 사용 중 |
| `build_cannot_recommend_chain` | 복용 불가 시 이유 설명 | 사용 중 |
| `build_summarize_chain` | 역질문 후 상황 요약 | 사용 중 |

---

## 절대 경로 → 상대 경로 변환 필요 파일

| 파일 | 변수 | 현재 경로 |
|------|------|-----------|
| `medication_loader.py` | `DB_PATH` | `C:\Team-Ragoon\...\drug_data.json` |
| `build_documents.py` | (하드코딩) | `C:/Team-Ragoon\...\drug_data.json` 외 2개 |


---

## 파일 구조

```
code/
├── prompt/
│   └── system_prompt.py    
├── analyzer.py
├── answer_parser.py
├── caution_parser.py
├── clarifier.py
├── dialogue.py
├── drug_detector.py
├── medication_loader.py
├── rag.py
├── validator.py
└── build_documents.py
```
