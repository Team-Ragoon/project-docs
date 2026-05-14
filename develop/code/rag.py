from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from prompts.system_prompt import (
    SYSTEM_PROMPT, FEW_SHOT_EXAMPLES, 
    SUMMARIZE_HUMAN_PROMPT, SUMMARIZE_SYSTEM_PROMPT,
    RECOMMEND_FINAL_HUMAN_PROMPT, RECOMMEND_FINAL_SYSTEM_PROMPT,
    CANNOT_RECOMMEND_SYSTEM_PROMPT, CANNOT_RECOMMEND_HUMAN_PROMPT,
    RAG_HUMAN_PROMPT
    )


# RAG 기반 QnA chain -> 사용자의 입력에 따라 약을 "추천"할 때 사용하는 chain
def build_rag_chain(llm: ChatOpenAI):
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


# 역질문 후 사용자의 현 상황 및 입력 내역 요약하는 chain -> 역질문 후 최종 답변 진행 전에 사용하는 chain
def build_summarize_chain(llm: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages([
        ("system", SUMMARIZE_SYSTEM_PROMPT),
        ("human", SUMMARIZE_HUMAN_PROMPT),
    ])
    return prompt | llm | StrOutputParser()


# 역질문 진행 후, 사용자 상황 기반에 따라 최종적으로 약을 추천하는 chain. -> 역질문 후, 언급한 약이 사용 가능하다고 판단되면 사용.
def build_recommend_final_chain(llm: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages([
        ("system", RECOMMEND_FINAL_SYSTEM_PROMPT),
        ("human", RECOMMEND_FINAL_HUMAN_PROMPT),
    ])
    return prompt | llm | StrOutputParser()


# 역질문 진행 후, 사용자 상황이 약의 금기 사항에 언급된 경우 사용하는 chain. -> 역질문 후, 언급한 약이 사용 불가능하다고 판단되면 사용.
def build_cannot_recommend_chain(llm: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages([
        ("system", CANNOT_RECOMMEND_SYSTEM_PROMPT),
        ("human", CANNOT_RECOMMEND_HUMAN_PROMPT),
    ])
    return prompt | llm | StrOutputParser()    