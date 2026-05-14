from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from prompts.system_prompt import CLARIFIER_SYSTEM_PROMPT

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", CLARIFIER_SYSTEM_PROMPT),
    ("human", """현재 파악된 정보: {filled_slots}
     확인할 항목: {subject} ({reason})
     권장 질문 형태 : {question}
     
     -> 자연스럽게 질문해주세요."""),
])

def build_clarifier_chain(llm: ChatOpenAI):
    return _PROMPT | llm | StrOutputParser()