"""
RAG 기반 의약품 QnA 시스템 - Query Expansion 버전
- 임베딩: BGE-M3
- 벡터 DB: ChromaDB
- LLM: GPT-4o-mini (OpenAI)
- 추가: Query Expansion (LLM으로 쿼리 자동 확장)
"""

import os
import json
import chromadb
from dotenv import load_dotenv
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from prompts.system_prompt import SYSTEM_PROMPT, FEW_SHOT_EXAMPLES

load_dotenv()

# 1. ChromaDB 로드
CHROMA_PATH = "C:/team-ragoon/project-docs/chroma_db"

ef = SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-m3")
client = chromadb.PersistentClient(path=CHROMA_PATH)

existing = [c.name for c in client.list_collections()]

if "drug_qna" in existing:
    collection = client.get_collection(name="drug_qna", embedding_function=ef)
    print(f"✅ 저장된 ChromaDB 로드 완료 ({collection.count()}개 문서)\n")
else:
    print("🔄 ChromaDB 최초 생성 중... (최초 1회만 실행)")
    collection = client.create_collection(
        name="drug_qna",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )
    with open("C:/team-ragoon/project-docs/study/week10/dataset/drug_documents.json", "r", encoding="utf-8") as f:
        documents = json.load(f)

    collection.add(
        documents=[doc["page_content"] for doc in documents],
        metadatas=[doc["metadata"] for doc in documents],
        ids=[f"drug_{i}" for i in range(len(documents))]
    )
    print(f"✅ {len(documents)}개 문서 임베딩 및 저장 완료\n")


# 2. LLM 설정
llm = ChatOpenAI(model="gpt-4.1", temperature=0)


# 3. Query Expansion
EXPAND_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """당신은 의약품 검색 전문가입니다.
사용자의 질문을 의학 용어와 일반 용어를 모두 포함해 2가지 다른 표현으로 바꿔주세요.
각 표현을 새 줄에 작성하고, 번호나 기호 없이 문장만 작성하세요."""),
    ("human", "{question}")
])

expand_chain = EXPAND_PROMPT | llm | StrOutputParser()


def expand_query(question: str) -> list[str]:
    """LLM으로 쿼리를 확장해 다양한 표현 생성"""
    expanded = expand_chain.invoke({"question": question})
    extra_queries = [q.strip() for q in expanded.strip().split("\n") if q.strip()]
    queries = [question] + extra_queries[:2]  # 원본 + 최대 2개
    return queries


# 4. Multi-Query Retriever
def retriever_multi(question: str, n_results: int = 5) -> str:
    """여러 쿼리로 검색 후 중복 제거, 유사도 순 정렬해서 반환"""
    queries = expand_query(question)

    print(f"  🔎 확장 쿼리: {queries}")

    # 쿼리별 검색 결과 수집 (item_seq 기준 중복 제거)
    all_docs = {}
    for query in queries:
        results = collection.query(query_texts=[query], n_results=n_results)
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            key = meta.get("item_seq", doc[:50])
            # 같은 문서면 유사도 높은 것(dist 낮은 것)만 유지
            if key not in all_docs or dist < all_docs[key]["dist"]:
                all_docs[key] = {"doc": doc, "dist": dist}

    # 유사도 순 정렬 후 상위 5개
    sorted_docs = sorted(all_docs.values(), key=lambda x: x["dist"])[:5]
    return "\n\n".join([d["doc"] for d in sorted_docs])


# 5. LangChain 체인 구성
few_shot_messages = []
for example in FEW_SHOT_EXAMPLES:
    few_shot_messages.append(("human", example["question"]))
    few_shot_messages.append(("ai", example["answer"]))

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    *few_shot_messages,
    MessagesPlaceholder(variable_name="chat_history"),  # 대화 기록 주입
    ("human", "[의약품 정보]\n{context}\n\n[사용자 질문]\n{question}")
])

chain = prompt | llm | StrOutputParser()


# 6. QnA 실행 함수 (단일 턴 - 시나리오 테스트용)
def ask(question: str):
    print(f"\n{'─' * 60}")
    print(f"🙋 질문: {question}")
    print(f"{'─' * 60}")

    context = retriever_multi(question)
    answer = chain.invoke({"context": context, "question": question, "chat_history": []})

    print(answer)
    print()


# 7. 역질문 응답 감지 함수
# 프롬프트에서 정의한 정보 수집 질문 패턴 → 답변 후 검색 스킵
INFO_QUESTION_PATTERNS = ["나이가", "임산부", "임신 중", "복용 중이"]

def is_followup_response(chat_history: list) -> bool:
    """직전 AI 메시지가 정보 수집 역질문이면 True → 검색 스킵
    증상 확인 질문(소화불량인가요? 등)은 False → 검색 실행
    """
    if not chat_history:
        return False
    last_ai = chat_history[-1].content.strip()
    if not last_ai.endswith("요?"):
        return False
    return any(kw in last_ai for kw in INFO_QUESTION_PATTERNS)


# 8. 멀티턴 대화 루프 (역질문 지원)
def run_chat():
    print("\n질문을 입력하세요. 종료하려면 'q' 또는 'quit'을 입력하세요.")
    print("대화를 초기화하려면 'clear'를 입력하세요.\n")

    chat_history = []  # [HumanMessage, AIMessage, ...] 순서로 누적

    while True:
        question = input("🙋 나: ").strip()
        if not question:
            continue
        if question.lower() in ("q", "quit"):
            print("\n종료합니다.")
            break
        if question.lower() == "clear":
            chat_history = []
            print("─" * 60)
            print("대화 기록이 초기화되었습니다.")
            print("─" * 60)
            continue

        # 역질문 응답이면 검색 스킵, 아니면 ChromaDB 검색
        if is_followup_response(chat_history):
            context = ""
        else:
            context = retriever_multi(question)

        answer = chain.invoke({
            "context": context,
            "question": question,
            "chat_history": chat_history
        })

        print(f"\n👨 약사 AI: {answer}\n")

        # 대화 기록에 이번 턴 추가
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=answer))


# 9. 실행 모드 선택
if __name__ == "__main__":
    print("=" * 60)
    print("  의약품 RAG QnA - Query Expansion 버전")
    print("=" * 60)    
    run_chat()
