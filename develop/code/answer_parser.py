from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from prompts.system_prompt import AGE_PARSE_SYSTEM_PROMPT

POSITIVE_KEYWORDS = ["예", "네", "맞아", "있어", "있음", "해당", "그렇", "맞습니다", "맞아요", "그래", "응"]
NEGATIVE_KEYWORDS = ["아니", "없어", "없음", "아님", "안 해당", "아닙니다", "아니에요", "아니요"]


# LLM 기반 긍정/부정 판단 prompt
_PARSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """사용자의 응답이 긍정인지 부정인지 판단해 JSON으로만 응답하세요.

{{"is_positive": true | false}}

예시:
- "네, 맞아요" → true
- "아니요" → false
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


# 사용자의 응답의 긍정/부정 판단. (1차: 규칙 기반) -> (2차: LLM 판단)
def parse_yes_no(user_input: str, question: str, llm: ChatOpenAI, subject: str = "") -> bool:
    # 1차: 규칙 기반 (명확한 경우 LLM 호출 없이 처리)
    has_positive = any(k in user_input for k in POSITIVE_KEYWORDS)
    has_negative = any(k in user_input for k in NEGATIVE_KEYWORDS)

    if has_negative and not has_positive:
        return False
    if has_positive and not has_negative:
        return True

    # 2차: 긍정/부정이 섞이거나 모호한 경우 LLM 판단
    chain = _PARSE_PROMPT | llm | JsonOutputParser()
    result = chain.invoke({"question": question, "user_input": user_input})
    return result.get("is_positive", False)


# 나이 역질문에 대한 사용자 응답에서 나이(세)를 추출.
# "초등학교 1학년", "소아" 등 범주형 표현도 LLM이 숫자로 변환.
# 변환 불가 시 None 반환.
def parse_age(user_input: str, llm: ChatOpenAI) -> float | None:
    chain = _AGE_PARSE_PROMPT | llm | JsonOutputParser()
    result = chain.invoke({"user_input": user_input})
    age = result.get("age")
    return float(age) if age is not None else None
