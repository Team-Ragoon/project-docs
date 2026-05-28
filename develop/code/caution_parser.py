import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from medication_loader import get_drug_all_caution_texts, has_acetaminophen
from prompts.system_prompt import CAUTION_PARSE_SYSTEM_PROMPT

CONTRAINDICATION_PATTERNS = [
    r"[^.\n]*복용하지\s*마십시오[^.\n]*[.\n]",
    r"[^.\n]*투여하지\s*마십시오[^.\n]*[.\n]",
    r"[^.\n]*사용하지\s*마십시오[^.\n]*[.\n]",
    r"[^.\n]*복용을\s*피하십시오[^.\n]*[.\n]",
    r"[^.\n]*함께\s*복용하지\s*마십시오[^.\n]*[.\n]",
    r"[^.\n]*함께\s*사용하지\s*마십시오[^.\n]*[.\n]",
    r"[^.\n]*음주하지\s*마십시오[^.\n]*[.\n]",
    r"[^.\n]*금기[^.\n]*[.\n]",
]

# 역질문 시 우선순위 판별 키워드 (숫자가 낮을수록 높은 우선순위)
PRIORITY_RULES = [
    # 우선순위 1: 나이 (N세/개월 미만·이하)
    (1, re.compile(r"(만\s*)?\d+\s*(개월|세)\s*(미만|이하)"
                   r"|소아|영아|유아|젖먹이|신생아|영유아|어린이"
                   r"|\b나이\b")),
    # 우선순위 2: 임산부
    (2, re.compile(r"임부|임신\s*가능성|임산부")),
    # 우선순위 3: 수유부
    (3, re.compile(r"수유부|수유\s*중")),
    # 우선순위 4: 알코올 (감기약·설사약 공통)
    (4, re.compile(r"알코올|음주|\b술\b")),
    # 우선순위 5: 불내성·결핍증·흡수장애 (설사약 핵심 금기 - 장엔폴·타라부틴)
    (5, re.compile(r"불내성|결핍증|흡수장애")),
    # 우선순위 6: 해열진통제 복용자 (감기약 특유)
    (6, re.compile(r"해열진통제")),
    # 우선순위 7: MAO 억제제 (감기약 특유)
    (7, re.compile(r"MAO\s*억제제|모노아민\s*산화효소|항우울제|항정신병")),
    # 우선순위 8: 위장진통·진경제 (설사약 특유 - 장엔폴 상호작용 금기)
    (8, re.compile(r"위장진통|진경제")),
    # 우선순위 9: 간장애
    (9, re.compile(r"간장애")),
    # 우선순위 10: 신장 관련
    (10, re.compile(r"신장|신부전|신장애")),
    # 우선순위 11: 심장/혈압 관련
    (11, re.compile(r"심장|고혈압|혈압|심부전")),
    # 우선순위 12: 위장 관련
    (12, re.compile(r"위장|궤양|출혈")),
    # 우선순위 13: 항암 관련
    (13, re.compile(r"항암|메토트렉세이트")),
    # 우선순위 14: 수술 관련
    (14, re.compile(r"수술|우회로")),
]

# 아세트아미노펜 최대 용량 슬롯 (validator에서 동적 삽입)
ACETAMINOPHEN_SLOT = {
    "subject": "아세트아미노펜 최대 용량",
    "question": "현재 아세트아미노펜(타이레놀 등) 성분의 다른 약을 함께 복용하고 계신가요?",
    "reason": "아세트아미노펜을 하루 최대 용량(4,000mg) 이상 복용하면 간 손상 위험이 있습니다.",
}

AGE_PATTERN = re.compile(r"(만\s*)?\d+\s*(개월|세)\s*(미만|이하)|소아|영아|유아|젖먹이|신생아|영유아|어린이|^나이$")

PREG_PATTERN = re.compile(r"임부|임신 가능성|임산부")
LACTATION_PATTERN = re.compile(r"수유부|수유 중")

LACTOSE_PATTERN = re.compile(r"갈락토오스\s*불내성|유당\s*분해효소\s*결핍|젖당\s*분해효소\s*결핍|포도당.갈락토오스\s*흡수장애|유당불내증|젖당불내증")

HYPERSENSITIVITY_PATTERN = re.compile(r"과민증|과민반응")

PREG_SLOT = {
    "subject": "임산부",
    "question": "혹시 임산부이시거나 임신 가능성이 있으신가요?",
    "reason": "태아에 영향을 줄 수 있음",
}

LACTATION_SLOT = {
    "subject": "수유부",
    "question": "혹시 수유 중이신가요?",
    "reason": "모유를 통해 아기에게 영향을 줄 수 있음",
}

LACTOSE_MERGED_SLOT = {
    "subject": "유당불내증",
    "question": "혹시 유당불내증(우유나 유제품을 먹으면 복통·설사가 생기는 경우)이 있으신가요?",
    "reason": "유당(젖당) 관련 유전적 문제가 있는 환자는 복용이 금지됩니다.",
}

# 금기 텍스트에서 "N세/개월 미만·이하" 기준 나이(세) 추출.
# "65세 이상 고령자" 등 미만/이하가 없는 표현은 의도적으로 무시.
# 반환값: {"age": float, "inclusive": bool} (이하=True, 미만=False) 또는 None
def _extract_age_threshold(text: str) -> dict | None:
    year_match = re.search(r"(\d+)\s*세\s*(미만|이하)", text)
    if year_match:
        return {"age": float(year_match.group(1)), "inclusive": year_match.group(2) == "이하"}
    month_match = re.search(r"(\d+)\s*개월\s*(미만|이하)", text)
    if month_match:
        return {"age": int(month_match.group(1)) / 12, "inclusive": month_match.group(2) == "이하"}
    return None


def _normalize_subject(subject: str, item: dict) -> tuple[str, dict]:
    if AGE_PATTERN.search(subject):
        return "나이", item
    elif PREG_PATTERN.search(subject):
        return "임산부", {**item, **PREG_SLOT}
    elif LACTATION_PATTERN.search(subject):
        return "수유부", {**item, **LACTATION_SLOT}
    elif LACTOSE_PATTERN.search(subject):
        return "유당불내증", {**item, **LACTOSE_MERGED_SLOT}
    return subject, item


# 전체 문장에서 금기 사항에 해당하는 문장만 추출.
def _extract_contraindication_sentencese(text: str) -> list[str]:
    sentences = []
    for pattern in CONTRAINDICATION_PATTERNS:
        found = re.findall(pattern, text)
        sentences.extend(found)
    return list(dict.fromkeys(s.strip() for s in sentences))


# 역질문의 우선순위
# reason이 모호할 수 있으므로 subject만으로 먼저 판별
def _get_priority(item: dict) -> int:
    subject = item.get('subject', '')
    for priority, pattern in PRIORITY_RULES:
        if pattern.search(subject):
            #print(f"[우선순위] '{subject}' → {priority}순위(패턴: {pattern.pattern})")
            return priority
    
    # reason까지 포함해서 재시도
    text = f"{subject}{item.get('reason', '')}"
    for priority, pattern in PRIORITY_RULES:
        if pattern.search(text):
            return priority
    
    #print(f"[우선순위] '{subject}' → 99순위")
    return 99  # 해당 없으면 맨 뒤


# 금기 대상을 json 형식으로 변환.
def build_caution_parser_chain(llm: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages([
        ("system", CAUTION_PARSE_SYSTEM_PROMPT),
        ("human", "금기 문장:\n{sentences}"),
    ])
    return prompt | llm | JsonOutputParser()


_cache: dict[str, tuple[dict, ...]] = {}


# 정규화된 subject에 맞는 약 텍스트 매칭 패턴 반환.
# 기존 AGE_PATTERN / PREG_PATTERN / LACTOSE_PATTERN 우선 사용,
# 그 외는 PRIORITY_RULES에서 탐색, 없으면 subject 자체로 검색.
def _get_subject_pattern(subject: str) -> re.Pattern:
    if subject == "나이":
        return AGE_PATTERN
    if subject == "임산부":
        return PREG_PATTERN
    if subject == "수유부":
        return LACTATION_PATTERN
    if subject == "유당불내증":
        return LACTOSE_PATTERN
    if subject == "과민증 환자":
        return HYPERSENSITIVITY_PATTERN
    for _, pattern in PRIORITY_RULES:
        if pattern.search(subject):
            return pattern
    return re.compile(re.escape(subject))


# 여러 약의 주의사항을 합산 및 parsing해서 역질문 목록 만드는 함수.
# 결과를 _cache에 저장해서 동일 약 reparsing 방지. (동일 약들 => 역질문 목록 다시 만들지 않음.)
def parse_contraindications_for_drugs(drug_names: list[str], llm: ChatOpenAI) -> tuple[dict, ...]:

    # cache key : 약 이름 목록을 정렬해서 문자열로
    cache_key = "|".join(sorted(drug_names))
    if cache_key in _cache:
        return _cache[cache_key]

    # 후보 약 전체 주의사항 합산 + 약별 텍스트 보관
    drug_texts: dict[str, str] = {}
    # applicable_drugs 매핑용: 금기 문장만 추출한 텍스트 (전체 텍스트 사용 시 "상의하십시오" 절의
    # 키워드도 hit되어 실제 금기가 아닌 약이 포함되는 오탐 방지)
    drug_contraindication_texts: dict[str, str] = {}
    combined_texts = []
    for drug_name in drug_names:
        texts = get_drug_all_caution_texts(drug_name)
        full_text = texts["atpnQesitm"] + "\n" + texts["intrcQesitm"]
        drug_texts[drug_name] = full_text
        drug_contraindication_texts[drug_name] = " ".join(
            _extract_contraindication_sentencese(full_text)
        )
        combined_texts.append(texts["atpnQesitm"])
        combined_texts.append(texts["intrcQesitm"])

    combined = "\n".join(t for t in combined_texts if t)
    sentences = _extract_contraindication_sentencese(combined)

    if not sentences:
        _cache[cache_key] = ()
        return ()


    chain = build_caution_parser_chain(llm)
    raw: list[dict] = chain.invoke({"sentences" : "\n".join(sentences)})

    # subject 중복 제거(정규화 없이 원본 subject 기준)
    seen = set()
    deduped = []
    for item in raw:
        normalized = _normalize_subject(item["subject"], item)[0]
        if normalized not in seen:
            seen.add(normalized)
            deduped.append(item)

    # 우선순위 정렬
    sorted_items = sorted(deduped, key = _get_priority)
    #print(f"[정렬 후 순서] {[item['subject'] for item in sorted_items]}")

    # 정규화 + applicable_drugs + (나이 슬롯은 age_thresholds) 추가
    final_items = []
    for item in sorted_items:
        normalized, updated_item = _normalize_subject(item["subject"], item)
        updated_item["subject"] = normalized
        pattern = _get_subject_pattern(normalized)
        updated_item["applicable_drugs"] = [
            drug for drug, text in drug_contraindication_texts.items()
            if pattern.search(text)
        ] or list(drug_names)  # 매칭 없으면 모든 약에 적용 (안전 fallback)

        if normalized == "나이":
            # 약별 나이 기준(세) 추출 → 최종 답변 시 약별 정밀 판단에 사용
            age_thresholds = {}
            for drug in updated_item["applicable_drugs"]:
                drug_sentences = _extract_contraindication_sentencese(drug_texts.get(drug, ""))
                threshold = _extract_age_threshold(" ".join(drug_sentences))
                if threshold is not None:
                    age_thresholds[drug] = threshold
            updated_item["age_thresholds"] = age_thresholds
            # 정확한 나이를 직접 묻는 질문으로 고정 (yes/no 형태 방지)
            updated_item["question"] = "복용하실 분의 정확한 나이를 알려주세요. (예: 7세, 30세)"

        final_items.append(updated_item)

    _cache[cache_key] = tuple(final_items)

    # 확인용 출력 ----------------------------------------
    # print("\n[역질문 목록]")
    # for i, item in enumerate(_cache[cache_key], 1):
    #     print(f"  {i}. subject: {item['subject']}")
    #     print(f"     applicable_drugs: {item['applicable_drugs']}")
    # ----------------------------------------------------

    return _cache[cache_key]
