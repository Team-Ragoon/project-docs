from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# 1차: 규칙 기반 키워드 매칭
POSITIVE_KEYWORDS = ["예", "네", "맞아", "있어", "있음", "해당", "그렇", "맞습니다", "맞아요", "그래"]
NEGATIVE_KEYWORDS = ["아니", "없어", "없음", "아님", "안 해당", "아닙니다", "아니에요", "아니요"]


# 2차: LLM 판단을 위한 prompt
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


# 사용자의 응답의 긍정/부정 판단을 위한 함수. (1차: 규칙 기반) -> (2차: LLM 판단)
def parse_yes_no(user_input: str, question: str, llm: ChatOpenAI) -> bool:
    
    # 1차: 규칙 기반 (명확한 경우 LLM 호출 없이 처리)
    has_positive = any(k in user_input for k in POSITIVE_KEYWORDS)
    has_negative = any(k in user_input for k in NEGATIVE_KEYWORDS)

    if has_negative and not has_positive:
        # 명확한 부정.
        return False 
    if has_positive and not has_negative:
        # 명확한 긍정.
        return True 


    # 2차: 긍정/부정이 섞이거나 모호한 경우 LLM 판단
    # 예: "네, 저는 임산부가 아니에요" → 긍정+부정 혼재 → LLM 판단
    chain = _PARSE_PROMPT | llm | JsonOutputParser()
    result = chain.invoke({
        "question": question,
        "user_input": user_input
    })
    return result.get("is_positive", False)