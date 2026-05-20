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
    r"[^.\n]*금기[^.\n]*[.\n]",
]

# 역질문 시 우선순위 판별 키워드 (숫자가 낮을수록 높은 우선순위)
PRIORITY_RULES = [
    # 우선순위 1: 나이 (N세/개월 미만·이하)
    (1, re.compile(r"(만\s*)?\d+\s*(개월|세)\s*(미만|이하)|\b나이\b")),
    # 우선순위 2: 임부/임산부/수유부 (알코올보다 우선)
    (2, re.compile(r"임부|임신 가능성|임산부|수유부|수유 중|임산부/수유부")),
    # 우선순위 3: 알코올
    (3, re.compile(r"알코올|음주|\b술\b")),
    # 우선순위 4: MAO 억제제
    (4, re.compile(r"MAO\s*억제제|모노아민\s*산화효소|항우울제|항정신병")),    
    # 우선순위 5: 불내성·결핍증·흡수장애 
    (5, re.compile(r"불내성|결핍증|흡수장애")),
    # 우선순위 6: 간장애
    (6, re.compile(r"간장애")),
    # 우선순위 7: 신장 관련
    (7, re.compile(r"신장|신부전|신장애")),
    # 우선순위 8: 심장/혈압 관련
    (8, re.compile(r"심장|고혈압|혈압|심부전")),
    # 우선순위 9: 위장 관련
    (9, re.compile(r"위장|궤양|출혈")),
    # 우선순위 10: 항암 관련
    (10, re.compile(r"항암|메토트렉세이트")),
    # 우선순위 11: 수술 관련
    (11, re.compile(r"수술|우회로")),
]

# 아세트아미노펜 최대 용량 슬롯 (validator에서 동적 삽입)
ACETAMINOPHEN_SLOT = {
    "subject": "아세트아미노펜 최대 용량",
    "question": "현재 아세트아미노펜(타이레놀 등) 성분의 다른 약을 함께 복용하고 계신가요?",
    "reason": "아세트아미노펜을 하루 최대 용량(4,000mg) 이상 복용하면 간 손상 위험이 있습니다.",
}

AGE_PATTERN = re.compile(r"(만\s*)?\d+\s*(개월|세)\s*(미만|이하)|소아|영아|유아|젖먹이|신생아|영유아|어린이|^나이$")

PREG_PATTERN = re.compile(r"임부|임신 가능성|임산부|수유부|수유 중")

LACTOSE_PATTERN = re.compile(r"갈락토오스\s*불내성|유당\s*분해효소\s*결핍|젖당\s*분해효소\s*결핍|포도당.갈락토오스\s*흡수장애|유당불내증|젖당불내증")


PREG_MERGED_SLOT = {
    "subject": "임산부/수유부",
    "question": "혹시 임산부이시거나 수유 중이신가요?",
    "reason": "태아 또는 모유를 통해 아기에게 영향을 줄 수 있음",
}

LACTOSE_MERGED_SLOT = {
    "subject": "유당불내증",
    "question": "혹시 유당불내증(우유나 유제품을 먹으면 복통·설사가 생기는 경우)이 있으신가요?",
    "reason": "유당(젖당) 관련 유전적 문제가 있는 환자는 복용이 금지됩니다.",
}

def _normalize_subject(subject: str, item: dict) -> tuple[str, dict]:
    if AGE_PATTERN.search(subject):
        return "나이", item
    elif PREG_PATTERN.search(subject):
        return "임산부/수유부", {**item, **PREG_MERGED_SLOT}
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


# 여러 약의 주의사항을 합산 및 parsing해서 역질문 목록 만드는 함수.
# 결과를 _cache에 저장해서 동일 약 reparsing 방지. (동일 약들 => 역질문 목록 다시 만들지 않음.)
def parse_contraindications_for_drugs(drug_names: list[str], llm: ChatOpenAI) -> tuple[dict, ...]:
    
    # cache key : 약 이름 목록을 정렬해서 문자열로
    cache_key = "|".join(sorted(drug_names))
    if cache_key in _cache:
        return _cache[cache_key]
    
    # 후보 약 전체 주의사항 합산
    combined_texts = []
    for drug_name in drug_names:
        texts = get_drug_all_caution_texts(drug_name)
        combined_texts.append(texts["atpnQesitm"])
        combined_texts.append(texts["intrcQesitm"])
        #combined_texts.append(texts["atpnWarnQesitm"])
    
    combined = "\n".join(t for t in combined_texts if t)
    #print(f"[combined 길이] {len(combined)}자")
    sentences = _extract_contraindication_sentencese(combined)
    # print(f"[추출된 금기 문장 수] {len(sentences)}개")
    # print(f"[추출된 문장들]\n" + "\n".join(sentences))


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

    # 정렬 후 정규화 적용
    final_items = []
    for item in sorted_items:
        normalized, updated_item = _normalize_subject(item["subject"], item)
        updated_item["subject"] = normalized
        final_items.append(updated_item)

    _cache[cache_key] = tuple(final_items)

    # 확인용 출력 ----------------------------------------
    print("\n[역질문 목록]")
    for i, item in enumerate(_cache[cache_key], 1):
        print(f"  {i}. subject: {item['subject']}")
        print(f"     question: {item['question']}")
        print(f"     reason:   {item['reason']}")
    # ---------------------------------------------------- 

    return _cache[cache_key]
