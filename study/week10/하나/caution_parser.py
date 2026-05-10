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


def parse_contraindications(drug_name: str, llm: ChatOpenAI) -> tuple[dict, ...]:
    # lru_cache는 list caching 불가능 -> tuple 반환.
    # 약 이름을 사용해서 금기 대상 구조화하기.
    # llm이 subject , reason 등의 구조로 정제하는 역할을 함.
    # 결과를 _cache에 저장해 동일 약 재파싱 방지
    if drug_name in _cache:
        return _cache[drug_name]
    
    texts = get_drug_all_caution_texts(drug_name)

    combined = texts["atpnQesitm"] + "\n" + texts["intrcQesitm"]
    # 두 field 합쳐서 금기 문장 추출하기
    sentences = _extract_contraindication_sentencese(combined)

    if not sentences:
        _cache[drug_name] = ()
        return ()
    
    chain = build_caution_parser_chain(llm)
    result = tuple(chain.invoke({"sentences" : "\n".join(sentences)}))
    _cache[drug_name] = result
    return result