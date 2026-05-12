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
import re

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


# 5.1 Clarification 체인 추가: 필요한 추가 정보 식별
CLARIFY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """당신은 의약품 정보의 충분성 판별자입니다.
주어진 [의약품 정보]와 사용자의 질문을 보고, 정확한 약 추천을 위해 '추가로 필요한 정보'가 있으면
간단한 문장 질문으로 각각 새 줄에 출력하세요. 정보가 충분하면 단어 ENOUGH 만 출력하세요.
출력은 번호나 기호 없이 질문만 새 줄에 작성하세요."""),
    ("human", "[의약품 정보]\n{context}\n\n[사용자 질문]\n{question}")
])

clarify_chain = CLARIFY_PROMPT | llm | StrOutputParser()

def _extract_keywords(text: str) -> set:
    # 한글(2자 이상)과 영숫자 키워드 추출
    if not text:
        return set()
    korean = re.findall(r"[가-힣]{2,}", text)
    alnum = re.findall(r"[A-Za-z0-9]{2,}", text)
    kws = set(korean + alnum)
    # 소문자 정규화
    return {k.lower() for k in kws}


def is_context_sufficient(question: str, context: str) -> bool:
    """
    간단 휴리스틱:
    - 질문에서 추출한 키워드(증상·대상 등)가 context(의약품 정보)에 하나라도 포함되어 있으면 충분하다고 판단.
    - 의학적 판단을 완전히 대체하지 않음. 필요하면 조건을 조정하세요.
    """
    qk = _extract_keywords(question)
    ck = _extract_keywords(context)
    if not qk or not ck:
        return False
    # 교집합이 있으면 충분한 근거가 있다고 판단
    return len(qk & ck) > 0


# 6. QnA 실행 함수
def ask(question: str):
    print(f"\n{'─' * 60}")
    print(f"🙋 질문: {question}")
    print(f"{'─' * 60}")

    # 1) 기본 검색으로 컨텍스트 수집
    context = retriever_multi(question)

    # 1.1 휴리스틱으로 우선 판정 — 질문 키워드가 컨텍스트에 포함되면 충분하다고 가정
    if is_context_sufficient(question, context):
        final_context = context
        print("ℹ️ 휴리스틱: 질문의 핵심 단어가 컨텍스트와 일치하므로 추가 질의 없이 진행합니다.")
    else:
        # 2) LLM에게 충분한지 확인하도록 요청
        clarify_out = clarify_chain.invoke({"context": context, "question": question}).strip()
        # 간단 판정: ENOUGH 포함이면 충분하다고 봄
        if "ENOUGH" in clarify_out.upper() or "충분" in clarify_out:
            final_context = context
        else:
            # 필요 정보 목록 파싱 (빈 줄 제거)
            questions_needed = [line.strip() for line in clarify_out.splitlines() if line.strip()]
            if questions_needed:
                print("\n🔎 추가 정보가 필요합니다. 아래 질문에 답해 주세요:")
                answers = []
                for q in questions_needed:
                    ans = input(f"- {q}\n  답변: ").strip()
                    answers.append((q, ans))

                # 사용자의 답변을 컨텍스트에 포함
                extra_section = "\n\n[추가 사용자 응답]\n" + "\n".join(f"Q: {q}\nA: {a}" for q, a in answers)
                final_context = context + extra_section
            else:
                final_context = context

    # 3) 최종 체인 실행 (최종 컨텍스트 포함)
    answer = chain.invoke({"context": final_context, "question": question})

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
            "운동하고 나서 근육이 너무 아파요. 성인용 진통 소염제 알려주세요.", #이건통과
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
