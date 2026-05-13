from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from system_prompt import ANALYZER_SYSTEM_PROMPT, SITUATION_EXTRACT_HUMAN_PROMPT, SITUATION_EXTRACT_SYSTEM_PROMPT


def build_analyzer_chain(llm: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages([
        ("system", ANALYZER_SYSTEM_PROMPT),
        ("human", "{user_input}"),
    ])
    return prompt | llm | JsonOutputParser()

def build_situation_extractor_chain(llm: ChatOpenAI):
    # 사용자 입력에서 상황 정보 사전 추출
    prompt = ChatPromptTemplate.from_messages([
        ("system", SITUATION_EXTRACT_SYSTEM_PROMPT),
        ("human", SITUATION_EXTRACT_HUMAN_PROMPT),
    ])
    return prompt | llm | JsonOutputParser()