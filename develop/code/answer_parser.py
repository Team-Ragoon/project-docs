import re
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

# 나이 관련 표현을 "해당함"으로 처리하는 키워드
AGE_RELATED_KEYWORDS = [
    "젖먹이", "영아", "영유아", "갓난", "신생아", "아기", "애기",
    "돌쟁이", "유아", "어린이", "소아",
]

AGE_PATTERN = re.compile(r"(만\s*)?\d+\s*(개월|세)\s*(미만|이하)|소아|영아|유아|젖먹이|신생아|영유아")

# 나이 관련 subject는 "나이"로 통일해서 저장하기. => 역질문 시 사용자에게 보여지는 형태를 변경하기 위한 함수
def record_clarify_answer(self, subject: str, is_positive: bool):
    key = "나이" if AGE_PATTERN.search(subject) else subject
    self.caution_slots[key] = is_positive



# 사용자의 응답의 긍정/부정 판단을 위한 함수. (1차: 규칙 기반) -> (2차: LLM 판단)
def parse_yes_no(user_input: str, question: str, llm: ChatOpenAI, subject: str="") -> bool:
    # 나이 관련 역질문일 때 구어체 표현 처리
    if subject and re.search(r"(개월|세)\s*(미만|이하)", subject):
        if any(k in user_input for k in AGE_RELATED_KEYWORDS):
            return True

    
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