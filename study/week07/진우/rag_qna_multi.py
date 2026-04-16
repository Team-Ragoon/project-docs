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
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from prompts.system_prompt import SYSTEM_PROMPT, FEW_SHOT_EXAMPLES

load_dotenv()

# 1. ChromaDB 로드
CHROMA_PATH = "C:/RAG/chroma_db"

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
    with open("C:/RAG/drug_documents.json", "r", encoding="utf-8") as f:
        documents = json.load(f)

    collection.add(
        documents=[doc["page_content"] for doc in documents],
        metadatas=[doc["metadata"] for doc in documents],
        ids=[f"drug_{i}" for i in range(len(documents))]
    )
    print(f"✅ {len(documents)}개 문서 임베딩 및 저장 완료\n")


# 2. LLM 설정
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


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
    ("human", "[의약품 정보]\n{context}\n\n[사용자 질문]\n{question}")
])

chain = prompt | llm | StrOutputParser()


# 6. QnA 실행 함수
def ask(question: str):
    print(f"\n{'─' * 60}")
    print(f"🙋 질문: {question}")
    print(f"{'─' * 60}")

    context = retriever_multi(question)
    answer = chain.invoke({"context": context, "question": question})

    print(answer)
    print()


# 7. 실행 모드 선택
if __name__ == "__main__":
    print("=" * 60)
    print("  의약품 RAG QnA - Query Expansion 버전")
    print("=" * 60)
    print("\n실행 모드를 선택하세요:")
    print("  1. 시나리오 테스트 (10개 자동 실행)")
    print("  2. 직접 질문 입력")

    mode = input("\n선택 (1 or 2): ").strip()

    if mode == "1":
        scenarios = [
            "임산부인데 두통이 심해요. 처방전 없이 먹을 수 있는 약 있나요?",
            "8살 아이가 감기로 열이 나요. 어린이용 해열제 추천해주세요.",
            "운동하고 나서 근육이 너무 아파요. 성인용 진통 소염제 알려주세요.",
            "과식해서 소화가 안 돼요. 더부룩하고 배가 불편합니다.",
            "위산이 역류하는 것 같고 속이 쓰려요. 제산제 추천해주세요.",
            "콧물, 기침, 인후통이 같이 있어요. 감기약 추천해주세요.",
            "두드러기가 나고 가려워요. 알러지 증상에 먹는 약이 있나요?",
            "피부에 상처가 났는데 세균 감염이 걱정돼요. 바르는 약 추천해주세요.",
            "생리통이 너무 심해요. 여성 월경통에 효과 있는 약이 뭔가요?",
            "눈이 피로하고 충혈됐어요. 안약 추천해주세요.",
        ]
        for question in scenarios:
            ask(question)

        print("=" * 60)
        print("  테스트 완료")
        print("=" * 60)

    elif mode == "2":
        print("\n질문을 입력하세요. 종료하려면 'q' 또는 'quit'을 입력하세요.\n")
        while True:
            question = input("🙋 질문: ").strip()
            if not question:
                continue
            if question.lower() in ("q", "quit"):
                print("\n종료합니다.")
                break
            ask(question)

    else:
        print("잘못된 입력입니다. 1 또는 2를 선택하세요.")
