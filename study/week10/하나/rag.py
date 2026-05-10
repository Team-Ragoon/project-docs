from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from system_prompt import SYSTEM_PROMPT, FEW_SHOT_EXAMPLES, CAUTION_SYSTEM_PROMPT, CAUTION_HUMAN_PROMPT


def build_rag_chain(llm: ChatOpenAI):
    # RAG 기반 QnA chain
    few_shot_messages = []
    for example in FEW_SHOT_EXAMPLES:
        few_shot_messages.append(("human", example["question"]))
        few_shot_messages.append(("ai", example["answer"]))

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        *few_shot_messages,
        ("human", "[의약품 정보]\n{context}\n\n[사용자 질문]\n{question}")
    ])

    return prompt | llm | StrOutputParser()

def build_recommend_chain(llm: ChatOpenAI):
    # 증상만 입력 -> 바로 약 추천
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "증상: {symptom}\n\n이 증상에 맞는 약을 추천해주세요."),        
    ])
    return prompt | llm | StrOutputParser()


def build_caution_chain(llm: ChatOpenAI):
    # 약까지 입력 -> 주의사항 확인 후 최종 답변
    prompt = ChatPromptTemplate.from_messages([
        ("system", CAUTION_SYSTEM_PROMPT),
        ("human", CAUTION_HUMAN_PROMPT)
    ])
    return prompt | llm | StrOutputParser()
