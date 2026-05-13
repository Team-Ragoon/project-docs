"""
RAG 기반 의약품 QnA 시스템
- 임베딩: BGE-M3
- 벡터 DB: ChromaDB
- LLM: GPT-4o-mini (OpenAI)
- 역질문: SYSTEM_PROMPT 기반 멀티턴 대화
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

# ─────────────────────────────────────────────
# 1. ChromaDB 로드
# ─────────────────────────────────────────────
DB_PATH = "../chroma_db"

embedder = SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-m3")
db_client = chromadb.PersistentClient(path=DB_PATH)

existing_collections = [c.name for c in db_client.list_collections()]

if "drug_qna" in existing_collections:
    drug_collection = db_client.get_collection(name="drug_qna", embedding_function=embedder)
    print(f"✅ 저장된 ChromaDB 로드 완료 ({drug_collection.count()}개 문서)\n")
else:
    print("🔄 ChromaDB 최초 생성 중... (최초 1회만 실행)")
    drug_collection = db_client.create_collection(
        name="drug_qna",
        embedding_function=embedder,
        metadata={"hnsw:space": "cosine"}
    )
    with open("../dataset/drug_documents.json", "r", encoding="utf-8") as f:
        doc_list = json.load(f)
    drug_collection.add(
        documents=[d["page_content"] for d in doc_list],
        metadatas=[d["metadata"] for d in doc_list],
        ids=[f"drug_{i}" for i in range(len(doc_list))]
    )
    print(f"✅ {len(doc_list)}개 문서 임베딩 및 저장 완료\n")


# ─────────────────────────────────────────────
# 2. LLM 설정
# ─────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ─────────────────────────────────────────────
# 3. Query Expansion + 문서 검색
# ─────────────────────────────────────────────
expand_prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 의약품 검색 전문가입니다.
사용자의 질문을 의학 용어와 일반 용어를 모두 포함해 2가지 다른 표현으로 바꿔주세요.
각 표현을 새 줄에 작성하고, 번호나 기호 없이 문장만 작성하세요."""),
    ("human", "{question}")
])
expand_chain = expand_prompt | llm | StrOutputParser()


def make_query_variants(question: str) -> list[str]:
    """쿼리를 다양한 표현으로 확장"""
    expanded = expand_chain.invoke({"question": question})
    variants = [q.strip() for q in expanded.strip().split("\n") if q.strip()]
    return [question] + variants[:2]


def search_documents(question: str, top_k: int = 5) -> str:
    """여러 쿼리로 검색 후 중복 제거, 유사도 순 정렬"""
    query_list = make_query_variants(question)
    print(f"  🔎 확장 쿼리: {query_list}")

    found = {}
    for q in query_list:
        results = drug_collection.query(query_texts=[q], n_results=top_k)
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            key = meta.get("item_seq", doc[:50])
            if key not in found or dist < found[key]["dist"]:
                found[key] = {"doc": doc, "dist": dist}

    ranked = sorted(found.values(), key=lambda x: x["dist"])[:5]
    return "\n\n".join([d["doc"] for d in ranked])


# ─────────────────────────────────────────────
# 4. LangChain 체인 구성
# ─────────────────────────────────────────────
shot_messages = []
for ex in FEW_SHOT_EXAMPLES:
    shot_messages.append(("human", ex["question"]))
    shot_messages.append(("ai", ex["answer"]))

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    *shot_messages,
    MessagesPlaceholder(variable_name="history"),
    ("human", "[의약품 정보]\n{context}\n\n[사용자 질문]\n{question}")
])

answer_chain = chat_prompt | llm | StrOutputParser()


# ─────────────────────────────────────────────
# 5. 역질문 중인지 판단 (검색 스킵 여부)
# ─────────────────────────────────────────────
COLLECTING_KEYWORDS = ["나이", "임산부", "임신", "수유", "발열", "유당불내증"]

def needs_search(history: list) -> bool:
    """직전 AI 답변이 정보 수집 질문이면 False (검색 스킵)"""
    if not history:
        return True
    last_ai = history[-1].content.strip()
    if not last_ai.endswith("요?"):
        return True
    # 정보 수집 키워드가 있으면 검색 스킵
    return not any(kw in last_ai for kw in COLLECTING_KEYWORDS)


# ─────────────────────────────────────────────
# 6. 대화 루프
# ─────────────────────────────────────────────
def start_conversation():
    print("\n질문을 입력하세요.")
    print("AI 종료: 'q' / 대화 초기화: 'clear'\n")

    history = []

    while True:
        user_input = input("🙋 나: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("q", "quit"):
            print("\n종료합니다.")
            break
        if user_input.lower() == "clear":
            history = []
            print("─" * 60)
            print("대화 기록이 초기화되었습니다.")
            print("─" * 60)
            continue

        # 역질문 응답이면 검색 스킵, 아니면 문서 검색
        if needs_search(history):
            context = search_documents(user_input)
        else:
            context = ""

        response = answer_chain.invoke({
            "context": context,
            "question": user_input,
            "history": history
        })

        print(f"\n💊 약사 AI: {response}\n")

        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=response))


# ─────────────────────────────────────────────
# 7. 실행
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  의약품 RAG QnA")
    print("=" * 60)
    start_conversation()