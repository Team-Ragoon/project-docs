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
from system_prompt import SYSTEM_PROMPT, FEW_SHOT_EXAMPLES, EXPAND_SYSTEM_PROMPT
from validator import SlotValidator
from analyzer import build_analyzer_chain
from clarifier import build_clarifier_chain
from rag import build_rag_chain, build_recommend_chain, build_caution_chain
from dialogue import DialogueState
from drug_detector import detect_drug_in_text

load_dotenv()


# 1. ChromaDB 로드
CHROMA_PATH = "C:/Team-Ragoon/project-docs/chroma_db"

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
    with open("C:/Team-Ragoon/project-docs/study/week10/dataset/drug_documents.json", "r", encoding="utf-8") as f:
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
    ("system", EXPAND_SYSTEM_PROMPT),
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


class MedicalChatbot:
    def __init__(self):
        self.state = DialogueState()
        self.validator = SlotValidator(llm = llm)
        self.analyzer = build_analyzer_chain(llm = llm)
        self.clarifier = build_clarifier_chain(llm = llm)
        self.recommender = build_recommend_chain(llm= llm)
        self.caution_answerer = build_caution_chain(llm =llm)
        self.rag_chain = build_rag_chain(llm = llm)
        self._pending_subject: str | None = None
    
    def chat(self, user_input: str) -> str:
        detected_drug = detect_drug_in_text(user_input)
        analysis = self.analyzer.invoke({"user_input" : user_input})

        if detected_drug:
            analysis["drug_name"] = detected_drug
            analysis["query_type"] = "medication"
        else:
            analysis["query_type"] = "symptom_only"

            
        self.state.update_from_analysis(analysis)

        # 직전 역질문 응답 처리
        if self._pending_subject:
            self.state.record_clarify_answer(self._pending_subject, user_input)
            self._pending_subject = None
        
        # 증상만 입력 => 바로 약 추천
        if self.state.query_type == "symptom_only":
            print("Symptom_only\n")
            context = retriever_multi(user_input)
            response = self.rag_chain.invoke({
                "context" : context,
                "question": user_input,
            })
            self.state.add_turn(user_input, response)
            return response

        # 질문에 약이 포함되어 있음 => 주의사항 기반 역질문
        if self.state.query_type == "medication":
            print("Medication\n")
            if self.validator.should_clarity(
                self.state.drug_name,
                self.state.caution_slots,
                self.state.clarify_count,
            ):
                slot = self.validator.get_priority_slot(
                    self.state.drug_name,
                    self.state.caution_slots,
                )
                self._pending_subject = slot["subject"]
                self.state.clarify_count += 1

                response = self.clarifier.invoke({
                    "filled_slots" : self.state.caution_slots,
                    "subject" : slot["subject"],
                    "reason": slot["reason"],
                    "question": slot["question"],
                })
            else:
                response = self._generate_final_answer()
            
            self.state.add_turn(user_input, response)
            return response

        return "죄송해요, 상비약 구매에 관련된 질문에만 답변드릴 수 있어요."

def _generate_final_answer(self) -> str:
    drug_name = self.state.drug_name
    all_contraindications = self.validator.get_contraindications(drug_name)

    applicable = [
        f"- {c['subject']}: {c['reason']}"
        for c in all_contraindications
        if self.state.caution_slots.get(c["subject"]) is True
    ]

    return self.caution_answerer.invoke({
        "drug_name" : drug_name,
        "symptom" : self.state.symptom or "언급 없음",
        "user_profile" : ", ".join(
            f"{s}: {'해당' if v else '해당 없음'}"
            for s, v in self.state.caution_slots.items()
        ) or "일반 성인",
        "applicable_cautions": "\n".join(applicable) or "특별한 금기사항 해당 없음"
    })



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
        bot = MedicalChatbot()
        scenarios = [
            "두통이 심한데 어떤 약을 먹으면 좋을까?",
            "두통이 심한데 타이레놀을 먹어도 될까?",
            "알레르기가 심한데 지르텍 먹어도 될까?"
        ]
        for question in scenarios:
            print(f"\n{'─' * 60}")
            print(f"🙋 질문: {question}")
            print(f"{'─' * 60}")
            print(bot.chat(question))

        print("=" * 60)
        print("  테스트 완료")
        print("=" * 60)

    elif mode == "2":
        bot = MedicalChatbot()
        print("\n질문을 입력하세요. 종료하려면 'q' 또는 'quit'을 입력하세요.\n")
        while True:
            question = input("🙋 질문: ").strip()
            if not question:
                continue
            if question.lower() in ("q", "quit"):
                print("\n종료합니다.")
                break
            print(bot.chat(question))

    else:
        print("잘못된 입력입니다. 1 또는 2를 선택하세요.")
