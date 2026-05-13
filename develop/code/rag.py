from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from system_prompt import (
    SYSTEM_PROMPT, FEW_SHOT_EXAMPLES, 
    SUMMARIZE_HUMAN_PROMPT, SUMMARIZE_SYSTEM_PROMPT,
    RECOMMEND_FINAL_HUMAN_PROMPT, RECOMMEND_FINAL_SYSTEM_PROMPT,
    CANNOT_RECOMMEND_SYSTEM_PROMPT, CANNOT_RECOMMEND_HUMAN_PROMPT,
    RAG_HUMAN_PROMPT
    )


def build_rag_chain(llm: ChatOpenAI):
    # RAG 기반 QnA chain
    few_shot_messages = []
    for example in FEW_SHOT_EXAMPLES:
        few_shot_messages.append(("human", example["question"]))
        few_shot_messages.append(("ai", example["answer"]))

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        *few_shot_messages,
        ("human", RAG_HUMAN_PROMPT)
    ])

    return prompt | llm | StrOutputParser()


def build_recommend_chain(llm: ChatOpenAI):
    # 증상만 입력 -> 바로 약 추천
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "증상: {symptom}\n\n이 증상에 맞는 약을 추천해주세요."),        
    ])
    return prompt | llm | StrOutputParser()


def build_summarize_chian(llm: ChatOpenAI):
    # 역질문 후 사용자의 상황 요약하기
    prompt = ChatPromptTemplate.from_messages([
        ("system", SUMMARIZE_SYSTEM_PROMPT),
        ("human", SUMMARIZE_HUMAN_PROMPT),
    ])
    return prompt | llm | StrOutputParser()

def build_recommend_final_chain(llm: ChatOpenAI):
    # 사용자 상황 기반 최종 약 추천
    prompt = ChatPromptTemplate.from_messages([
        ("system", RECOMMEND_FINAL_SYSTEM_PROMPT),
        ("human", RECOMMEND_FINAL_HUMAN_PROMPT),
    ])
    return prompt | llm | StrOutputParser()

def build_cannot_recommend_chain(llm: ChatOpenAI):
    # 복용 불가 시 이유 설명
    prompt = ChatPromptTemplate.from_messages([
        ("system", CANNOT_RECOMMEND_SYSTEM_PROMPT),
        ("human", CANNOT_RECOMMEND_HUMAN_PROMPT),
    ])
    return prompt | llm | StrOutputParser()    