from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from prompts.system_prompt import AGE_PARSE_SYSTEM_PROMPT

POSITIVE_KEYWORDS = ["예", "네", "맞아", "있어", "있음", "해당", "그렇", "맞습니다", "맞아요", "그래", "응"]
NEGATIVE_KEYWORDS = ["아니", "없어", "없음", "아님", "안 해당", "아닙니다", "아니에요", "아니요"]
UNKNOWN_KEYWORDS = ["모르겠", "잘 모르", "모름", "몰라", "모릅니다", "잘 몰라"]

# "안하고 있어", "하지 않아" 처럼 긍정 키워드를 포함하지만 실제로는 부정인 패턴
# → 이 패턴이 매칭되면 긍정 키워드 무시하고 부정으로 처리
NEGATION_OVERRIDES = ["안하고", "안 하고", "하지 않", "안해요", "안해", "안 해", "안 하", "안먹고", "안 먹고"]


# LLM 기반 긍정/부정/모름 판단 prompt
_PARSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """사용자의 응답이 긍정인지, 부정인지, 또는 모름인지 판단해 JSON으로만 응답하세요.

{{"is_positive": true | false | null}}

- 명확히 긍정 → true
- 명확히 부정 → false
- 모르겠다, 잘 모르겠다, 불확실 → null

예시:
- "네, 맞아요" → true
- "아니요" → false
- "잘 모르겠어요" → null
- "네, 저는 임산부가 아니에요" → false  (문맥상 부정)
- "ㅇㅇ" -> true
- "ㄴㄴ" -> false
- "있긴 한데 잘 모르겠어요" → true"""),
    ("human", "역질문: {question}\n사용자 응답: {user_input}")
])

# LLM 기반 나이 추출 prompt
_AGE_PARSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", AGE_PARSE_SYSTEM_PROMPT),
    ("human", "나이 응답: {user_input}")
])


# 사용자의 응답의 긍정/부정/모름 판단. (1차: 규칙 기반) -> (2차: LLM 판단)
# 반환값: True(긍정), False(부정), None(모름/불확실)
def parse_yes_no(user_input: str, question: str, llm: ChatOpenAI, subject: str = "") -> bool | None:
    # 1차: 규칙 기반 (명확한 경우 LLM 호출 없이 처리)
    has_override = any(k in user_input for k in NEGATION_OVERRIDES)
    has_positive = any(k in user_input for k in POSITIVE_KEYWORDS) and not has_override
    has_negative = any(k in user_input for k in NEGATIVE_KEYWORDS) or has_override
    has_unknown = any(k in user_input for k in UNKNOWN_KEYWORDS)

    if has_negative and not has_positive:
        return False
    if has_positive and not has_negative:
        return True
    if has_unknown and not has_positive and not has_negative:
        return None

    # 2차: 긍정/부정이 섞이거나 모호한 경우 LLM 판단
    chain = _PARSE_PROMPT | llm | JsonOutputParser()
    result = chain.invoke({"question": question, "user_input": user_input})
    is_pos = result.get("is_positive")
    return None if is_pos is None else bool(is_pos)


# 나이 역질문에 대한 사용자 응답에서 나이(세)를 추출.
# "초등학교 1학년", "소아" 등 범주형 표현도 LLM이 숫자로 변환.
# 변환 불가 시 None 반환.
def parse_age(user_input: str, llm: ChatOpenAI) -> float | None:
    chain = _AGE_PARSE_PROMPT | llm | JsonOutputParser()
    result = chain.invoke({"user_input": user_input})
    age = result.get("age")
    return float(age) if age is not None else None
