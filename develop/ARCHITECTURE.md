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

 ~~클래스 DialogueState에서 history를 사용할 것인지.
(지금 코드에서는 역질문에 대한 답을 따로 저장하고 반영 -> history 필요 없음)
(다만, 필수 역질문 등을 구현할 때 사용될 여지가 있으므로 삭제는 하지 않음.)~~

-> 흐름 A에서 사용.



# 현재 코드에서 구현해야 하는 기능

- ~~흐름 A에서 필수 역질문(약 종류에 따라 조금씩 달라질 듯. "임산부" 정보만 고정해두면 될 듯).~~ -> 구현 완료
- 흐름 B에서 약이 다양할 때 그 중 한 개를 추천하기.(사용자의 caution_slots 및 상황에 따라서)
  <br>-> 약 별로 caution_slots 구분해야 함.



# 전체 흐름

## chat() 진입 시 분기
```
사용자 입력
    │
    ├─ 흐름 A 역질문 진행 중? (_in_flow_a_clarify = True)
    │      → rag_chain으로 바로 응답 (컨텍스트 재사용)
    │
    ├─ 흐름 B 역질문 대기 중? (_pending_subject 존재)
    │      → 역질문 응답 처리
    │
    └─ 새 쿼리
           → 약 탐지 → 의도 분석 → 상황 추출 → 흐름 A 또는 B


(새 query 처리 과정)
1. 약 이름 탐지
  -> query 내에 DB 내에 저장되어 있는 약 이름이 포함되어 있는지 확인.
  -> 위치 : rag_qna_multi.py의 새 query를 위한 응답 초기화 직후
  -> 함수 : detect_drugs_in_text() (drug_detector.py에 정의되어 있음.)
  -> 입력을 token화 -> 조사 제거 진행 -> DB 내의 약과 이름 비교 -> matching된 약 이름 목록과 사용자의 원문 keyword 모두 반환.

2. query 의도 분석
  -> query의 의도 종류 : symptom_only(증상만 언급), medication(특정 약 복용 가능 여부), comparison(두 약 비교)
  -> symptom_only / comparison => 흐름 A
  -> medication => 흐름 B

3. 상황 사전 추출(query의 의도가 medication인 경우만)
  -> self.situation_extractor 사용해서 SITUATION_EXTRACT_SYSTEM_PROMPT 기반으로 query에서 사용자 상황을 미리 추출해 역질문을 줄임.
  -> query 내에 역질문이 필요한 금기 사항에 해당하는 내용이 있으면 역질문 진행하지 않도록 미리 처리.
  -> 금기 사항이 아닌 사용자 상황들은 extra_context에 저장.
  -> 위치 : rag_qna_multi.py에서 query의 의도가 medication으로 분석된 다음.
  -> 함수 : analyzer.py의 build_situation_extract_chain() 사용.
  -> 추출한 내용을 금기 목록 존재 여부를 판단하는 함수 : dialogue.py의 apply_extracted_situation()
```

## 흐름 A — 증상만 입력
{05.14 - 증상만 입력 -> 약 추천으로 진행. (시나리오 테스트 필요, 필수 역질문하여 추천하도록 기능 추가해야 함.)}<br>
{05.24 - 필수 역질문 구현 완료.}
```
1. ChromaDB에서 관련 문서 검색
2. query expansion (expand_chain) -> 쿼리 3가지 표현으로 확장
3. 각 확장 쿼리로 검색 -> 유사도 기준 상위 5개 문서 선택
4. context를 _cached_context에 저장. (_in_flow_a_clarify = True)
5. SYSTEM_PROMPT 기반으로 LLM이 자율 역질문 진행
6. 최종 답변
```

## 흐름 B — 약 이름 포함
{05.14 - 약 이름이 포함되면 금기 사항 기반 역질문 진행 => 약 복용 가능 여부를 판단.}<br>
{05.16 - query에 금기 사항이 존재 -> 역질문 진행 X, 바로 복용 불가능함을 안내함}<br>
{05.16 - 금기 사항 안에서 우선순위 결정 -> 우선순위별로 최대 3개의 역질문 진행하도록 함. 이때, 질문하지 않은 금기 사항에 대해 최종 답변에서 언급하여 주의할 수 있도록 함.}<br>
```
1. 역질문 목록 생성
  - caution_parser.py 내의 함수들이 진행.
  - cache 적용해서 동일 약에 대해서는 reparsing 진행하지 않음.
  - 모든 후보 약의 주의사항 text 수집. (atphQestim, intrcQestim)
    -> CONTRAINDICATION_PATTERNS로 정해놓은 pattern에 해당하는 금기 문장만 추출. 
    -> LLM이 JSON으로 parsing (CAUTION_PARSER_SYSTEM_PROMPT 기반)
    -> subject 정규화 (중복 통합 / 임산부 및 수유부는 1개의 subject로 합침, 유당불내증 관련 질환들도 1개의 subject로 합침)
    -> PRIORITY_RULES 기반으로 우선순위 정렬.
    -> applicable_drugs 필드 사용해서 각 약의 원본 text에 해당 subject가 있는지 mapping 생성.
  - 아세트아미노펜 성분 약이 있으면 validator.py에서 우선순위 3위로 삽입함.

2. 남은 질문 slot 판별
  - validator.py -> get_missing_slots() -> should_clarify() / get_priority_slot()
  - 이미 답변된 slot 제외 -> skip 조건 확인 -> 아직 질문 안 한 slot 목록 반환
  - rag_qna_multi.py에서 _next_clarify_or_answer()에서 can_drugs(금기 제외 후 남은 약)을 먼저 계산한 뒤, 그 약들 기준으로 should_clarify() 호출 -> True면 get_priority_slot()으로 다음 질문 1개 꺼냄.

3. 역질문 생성
  - clarifier.py -> build_clarifier_chain()
  - slot의 subject, reason, question 받아 자연스러운 한국어 질문 문장으로 가공.
  - rag_qna_multi.py에서는 결과 문장을 _pending_question에 저장, _pending_subject도 같이 저장해 다음 턴에 어떤 slot의 답변인지 추적.

4. 사용자 답변 parsing
  - answer_pasrser.py -> parse_yes_no() / parse_age()
  - parse_yes_no() : 규칙 기반(긍정/부정 키워드) -> 모호하면 LLM 판단. (나이 외의 모든 cuation_slot에 사용)
  - parse_age() : LLM이 text에서 나이를 숫자로 추출. 불가능하면 None("몰라요"와 같은 답변)
  - rag_qna_multi.py에서 _pending_subject가 있으면 역질문 항목이 있다는 의미이므로 이때, 진입.
    -> _pending_subject가 "나이"일 때 parse_age()
    -> _pending_subject가 그 외라면 parse_yes_no()
  - 호출 결과는 caution_slots / exta_context에 저장/

5. 추천 가능 약 계산
  - rag_qna_multi.py의 _get_remaining_candidates()
  - caution_slots에서 True인 주제 모아, 각 slot의 applicable_drugs 확인 -> 해당하는 약 제외.
  - 나이 slot은 age_thresholds를 약별로 비교해 개별 제외. (나이에 대한 금기사항이 다를 수 있기 때문)
  - 최종적으로 금기 없는 약 목록(can_drugs) 반환.
  - 남은 질문 여부를 can_drugs 기준으로 판단. (+ 역질문 횟수)

6. 최종 답변 생성
  - build_recommend_final_chain() / build_cannot_recommend_chain()
    -> can_drugs 존재 => build_recommend_final_chain() : 추천 가능한 약과 복용 안내 생성.
    -> can_drugs 존재 X => build_cannot_recommend_chain() : 추천 불가 이유 설명.
  - rag_qna_multi.py에서 _generate_final_answer()에서 can_drugs 여부에 따라 분기하여 진행.
  

7. (추가 사항) 나이 처리 방식
  - 기준 추출 : caution_parser.py의 _extract_age_threshold()
    -> 약마다 금기사항 기준인 나이가 다를 수 있음.
    -> 약마다 text에서 기준 표현만 추출 -> 약별로 금기사항 기준이 되는 나이 저장.
  
  - 사전 파악 : analyzer.py
    -> query에 나이가 명시되 경우, extra_context["나이"]에 숫자로 저장.
  
  - 자동 처리: validator.py의 get_missing_slots()
    -> extra_context["나이"]가 있으면 age_thresholds와 비교 -> slot 자동 채움, 역질문 생략
  
  - 역질문 : caution_parser.py
    -> 나이 slot의 question은 정확히 나이를 묻는 형태로 고정.
  
  - 답변 parsing : answer_parser.py의 parse_age()
  
  - 약별 제외 : rag_qna_multi.py의 _get_remaining_candidates()
    -> 약별 age_thresholds 비교 
    -> 기준 미만인 약들은 제외.
```
```
<금기 사항 parsing 및 matching 방식의 변경 사항.>
1. 기존 방식
  candidates인 약들의 주의사항을 모두 합산. (약별 caution_slot 따로 저장 X.) -> 최종 답변 생성 시 각 약의 원본 금기 text 다시 읽어 True인 subject keyword가 있는지 text matching으로 판별.

2. 기존 방식의 문제점
  - 여러 약을 합산 parsing할 때, subject가 어떤 약의 금기사항인지 정보가 버려짐. => 각 약의 text를 다시 읽어야 함.
  - text에 있는 다양한 표현 때문에 matching이 깨질 수 있음.

3. 현재 방식
  - parsing 시점에 applicable_drugs 필드를 같이 생성.
  - 합산 parsing은 그대로 유지하되, 의미 있는 정규화 패턴을 이용해 각 약의 원본 텍스트에 해당 subject가 있는지 mapping 추가 생성.
  - 금기 사항에 대해 해당 금기 사항을 가지는 약들이 applicable_drugs : ["약A", "약B" ..]와 같이 저장되는 방식.
  - 최종 답변 생성 시 text 전체가 없이 해당 field만 다시 확인하면 됨.
```
```
<나이 관련 금기사항 역질문 방식의 변경 사항.>
1. 기존 방식
  금기사항에 있는 기준 그대로 예/아니오 형태로 답할 수 있도록 명확한 나이를 언급하는 역질문.

2. 기존 방식의 문제점
  - 약 별로 나이에 관련된 금기사항이 다를 수 있음. (7세 미만, 12세 미만 ..)
  - 이럴 땐, 나이 관련 금기사항 1개만 뽑아서 질문함. (모든 약에 대한 확인을 할 수 없음)

3. 현재 방식
  - 나이 관련 금기사항이 존재하는 경우에는 나이를 묻는 역질문을 진행.
  - 만약, 정확한 연령이 아닌 어린이와 같은 범주형 답변이라면 llm이 알아서 대략적인 나이를 유추하도록 함.
  - 초등학교 1학년과 같이 숫자로 표현하지 않았지만 유추 가능한 표현도 llm이 변환하도록 함.
  - 사용자의 나이를 직접 입력 받아서 그걸로 caution_slots의 True/False 여부를 판단함.
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

| 함수 | 용도 | 상태 | 흐름 |
|------|------|------|
| `build_rag_chain` | 증상 입력 → 바로 RAG 추천 | 사용 중 | A |
| `build_recommend_final_chain` | 슬롯 완료 후 최종 추천 | 사용 중 | B |
| `build_cannot_recommend_chain` | 복용 불가 시 이유 설명 | 사용 중 | B |
| `build_summarize_chain` | 역질문 후 상황 요약 | 사용 중 | B |


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

## 리액트 실행

Python 패키지 설치(`pip install`)와 API 서버(`api_server.py`)는 `npm start`가 자동으로 처리합니다.

```bash
cd develop/front
npm install   # 최초 1회만 (Node 패키지)
npm start
```

- API http://localhost:8000 · React http://localhost:3000
