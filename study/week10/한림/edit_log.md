- Reverse_qa 1차 수정 (2026/05/13)
  **문제**
  "나 머리아파"처럼 짧은 질문 입력 시, LLM이 주의사항 텍스트에서 복용금지 키워드를 추출하지 못해 options가 빈 배열로 반환됨. options가 비면 역질문을 아예 스킵하고 바로 답변으로 넘어가는 구조였음.

**수정 내용**
reverse_qa.py Step 2에서 LLM이 복용금지 키워드 추출에 실패해 options가 비는 경우, 아래 4개 항목을 기본값으로 fallback 처리하도록 추가.

---

## 자세한 내용

- 1차 수정
  수정 내용

reverse_qa.py Step 2에서 LLM이 복용금지 키워드 추출에 실패해 options가 비는 경우, 아래 4개 항목을 기본값으로 fallback 처리하도록 추가.

- 임신 중이거나 임신 가능성 있음
- 평소 음주를 즐기거나 최근 음주함
- 만 15세 미만 어린이
- 간 질환 보유
  기존엔 options가 비면 역질문을 아예 스킵하고 바로 답변으로 넘어갔는데, 이제 최소한 위 4개는 항상 물어보게 됨.

---

역질문에 '질문'으로 할 내용을 넣고 싶으면, CONTRAINDICATION_LABEL_MAP에 키워드 추가하거나, fallback options 리스트에 직접 항목 추가하면 됨.

LLM이 자동 추출하는 경우 → reverse_prompt.py의 CONTRAINDICATION_LABEL_MAP에 추가

무조건 물어보고 싶은 경우 → reverse_qa.py의 fallback options 리스트에 직접 추가

해당 내용 반영한 colab : https://colab.research.google.com/drive/16Qo-q7lesTSDr5E1KvjpMX19BLhVCdCi#scrollTo=9NLslEAL5BjS

## 해당 셀 8 - reverse_qa (역질문 로직)
