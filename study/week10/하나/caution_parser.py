import re
from functools import lru_cache
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from medication_loader import get_drug_all_caution_texts
from system_prompt import CAUTION_PARSE_SYSTEM_PROMPT

CONTRAINDICATION_PATTERNS = [
    r"[^.\n]*복용하지\s*마십시오[^.\n]*[.\n]",
    r"[^.\n]*투여하지\s*마십시오[^.\n]*[.\n]",
    r"[^.\n]*사용하지\s*마십시오[^.\n]*[.\n]",
    r"[^.\n]*복용을\s*피하십시오[^.\n]*[.\n]",
    r"[^.\n]*함께\s*복용하지\s*마십시오[^.\n]*[.\n]",
    r"[^.\n]*함께\s*사용하지\s*마십시오[^.\n]*[.\n]",
    r"[^.\n]*금기[^.\n]*[.\n]",
]

def _extract_contraindication_sentencese(text: str) -> list[str]:
    sentences = []
    for pattern in CONTRAINDICATION_PATTERNS:
        found = re.findall(pattern, text)
        sentences.extend(found)
    return list(dict.fromkeys(s.strip() for s in sentences))


def build_caution_parser_chain(llm: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages([
        ("system", CAUTION_PARSE_SYSTEM_PROMPT),
        ("human", "금기 문장:\n{sentences}"),
    ])
    return prompt | llm | JsonOutputParser()


_cache: dict[str, tuple[dict, ...]] = {}


def parse_contraindications_for_drugs(drug_names: list[str], llm: ChatOpenAI) -> tuple[dict, ...]:
    # lru_cache는 list caching 불가능 -> tuple 반환.
    # 여러 약의 주의사항을 합산해서 parsing 
    # llm이 subject , reason 등의 구조로 정제하는 역할을 함.
    # 결과를 _cache에 저장해 동일 약 재파싱 방지
    
    # cache key : 약 이름 목록을 정렬해서 문자열로
    cache_key = "|".join(sorted(drug_names))
    if cache_key in _cache:
        return _cache[cache_key]
    
    # 후보 약 전체 주의사항 합산
    combined_texts = []
    for drug_name in drug_names:
        print(drug_name)
        texts = get_drug_all_caution_texts(drug_name)
        combined_texts.append(texts["atpnQesitm"])
        combined_texts.append(texts["intrcQesitm"])
        #combined_texts.append(texts["atpnWarnQesitm"])
    
    combined = "\n".join(t for t in combined_texts if t)
    sentences = _extract_contraindication_sentencese(combined)

    if not sentences:
        _cache[cache_key] = ()
        return ()

    
    chain = build_caution_parser_chain(llm)
    result = tuple(chain.invoke({"sentences" : "\n".join(sentences)}))

    # subject 중복 제거
    seen = set()
    deduped = []
    for item in result:
        if item["subject"] not in seen:
            seen.add(item["subject"])
            deduped.append(item)

    _cache[cache_key] = tuple(deduped)

    # ── 확인용 출력 ──────────────────────────────────────
    print("\n[역질문 목록]")
    for i, item in enumerate(_cache[cache_key], 1):
        print(f"  {i}. subject: {item['subject']}")
        print(f"     question: {item['question']}")
        print(f"     reason:   {item['reason']}")
    # ────────────────────────────────────────────────────   
    return _cache[cache_key]