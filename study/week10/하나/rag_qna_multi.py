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
from system_prompt import EXPAND_SYSTEM_PROMPT
from validator import SlotValidator
from analyzer import build_analyzer_chain
from clarifier import build_clarifier_chain
from rag import build_rag_chain, build_recommend_chain,build_summarize_chian, build_recommend_final_chain
from dialogue import DialogueState
from drug_detector import detect_drugs_in_text
from answer_parser import parse_yes_no

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

# Chatbot class 선언
class MedicalChatbot:
    def __init__(self):
        self.state = DialogueState()
        self.validator = SlotValidator(llm = llm)
        self.analyzer = build_analyzer_chain(llm = llm)
        self.clarifier = build_clarifier_chain(llm = llm)
        self.recommender = build_recommend_chain(llm= llm)
        self.recommend_final = build_recommend_final_chain(llm = llm)
        self.rag_chain = build_rag_chain(llm = llm)
        self.summarizer = build_summarize_chian(llm = llm)
        self._pending_subject: str | None = None
        self._pending_question: str | None = None
    
    def chat(self, user_input: str) -> str:
        if self._pending_subject:
            is_positive = parse_yes_no(
                user_input = user_input,
                question = self._pending_question,
                llm = llm
            )
            self.state.record_clarify_answer(self._pending_subject, is_positive)
            self._pending_subject = None
            self._pending_question = None

            # ── 확인용 출력 ──────────────────────────────────────
            print("\n[caution_slots 현재 상태]")
            for subject, value in self.state.caution_slots.items():
                status = "해당" if value else "해당 없음" if value is False else "미응답"
                print(f"  {subject}: {status}")
            # ────────────────────────────────────────────────────

            # 저장 후 바로 다음 역질문 / 최종 답변으로 이동
            return self._next_clarify_or_answer(user_input)
        
        # 후보 약 탐지 (복수 개)
        matched_drugs, user_keyword = detect_drugs_in_text(user_input)

        analysis = self.analyzer.invoke({
            "history": self.state.get_history(), 
            "user_input" : user_input,
        })

        if matched_drugs:
            analysis["query_type"] = "medication"
        else:
            if analysis.get("symptom"):
                analysis["query_type"] = "symptom_only"
            
        # 상태 update
        self.state.update_from_analysis(analysis)

        if matched_drugs:
            self.state.set_drug_candidates(matched_drugs, user_keyword)


        # 증상만 입력 => 바로 약 추천
        if self.state.query_type == "symptom_only":
            context = retriever_multi(user_input)
            response = self.rag_chain.invoke({
                "context" : context,
                "question": user_input,
            })
            self.state.add_turn(user_input, response)
            return response

        # 질문에 약이 포함되어 있음 => 주의사항 기반 역질문
        if self.state.query_type == "medication":
            response = self._next_clarify_or_answer(user_input)
            self.state.add_turn(user_input, response)
            return response

        return "죄송해요, 상비약 구매에 관련된 질문에만 답변드릴 수 있어요."

    def _next_clarify_or_answer(self, user_input: str) -> str:
        """
        다음 역질문이 있으면 역질문 반환
        모든 슬롯이 채워졌으면 최종 답변 생성
        역질문 응답 처리 후 / medication 진입 시 공통 호출
        """
        if self.validator.should_clarify(
            self.state.drug_names,
            self.state.caution_slots,
            self.state.clarify_count,
        ):
            slot = self.validator.get_priority_slot(
                self.state.drug_names,
                self.state.caution_slots,
            )
            response = self.clarifier.invoke({
                "filled_slots": self.state.caution_slots,
                "subject":      slot["subject"],
                "reason":       slot["reason"],
                "question":     slot["question"],
            })
            # 역질문 텍스트 + subject 저장
            self._pending_subject = slot["subject"]
            self._pending_question = response
            self.state.clarify_count += 1
            return response

        # 모든 슬롯 충족 → 최종 답변
        return self._generate_final_answer()


    def _build_user_profile(self) -> str:
        # caution_slots -> 자연어로 변환
        lines = []
        for subject, value in self.state.caution_slots.items():
            status = "해당" if value else "해당 없음"
            lines.append(f"-{subject} : {status}")
        return "\n".join(lines) if lines else "- 특이사항 없음"

    def _generate_final_answer(self) -> str:
        drug_names =self.state.drug_names
        drug_keyword = self.state.drug_keyword
        all_contraindications = self.validator.get_contraindications(drug_names)
        user_profile = self._build_user_profile()

        summary = self.summarizer.invoke({
            "drug_name" : drug_keyword,
            "symptom" : self.state.symptom or "언급 없음",
            "user_profile" : user_profile,
        })

        applicable = [
            f"- {c['subject']}: {c['reason']}"
            for c in all_contraindications
            if self.state.caution_slots.get(c["subject"]) is True
        ]


        recommendation = self.recommend_final.invoke({
            "drug_keyword" : drug_keyword,
            "drug_candidates" : "\n".join(f"- {d}" for d in drug_names),
            "user_profile" : user_profile,
            "applicable_cautions": "\n".join(applicable) or "특별한 금기사항 해당 없음"
        })

        return f"{summary}\n\n{recommendation}"


"""
역질문 흐름 테스트
- 챗봇이 역질문을 하면 미리 정의된 답변으로 응답
- 실제 대화 흐름과 동일하게 턴 단위로 진행
"""

def run_clarify_test(scenario: dict):
    """
    scenario 구조:
    {
        "name": "테스트 이름",
        "first_input": "첫 질문",
        "answers": {
            "임산부": "아니요",           # 역질문 subject → 사용자 답변
            "소아": "아니요",
            "신장질환자": "네, 있어요",
        }
    }
    """
    print(f"\n{'=' * 60}")
    print(f"  테스트: {scenario['name']}")
    print(f"{'=' * 60}")

    bot = MedicalChatbot()
    MAX_TURNS = 10  # 무한 루프 방지

    # 1턴: 첫 질문
    user_input = scenario["first_input"]
    turn = 0

    while turn < MAX_TURNS:
        print(f"\n🙋 사용자: {user_input}")
        response = bot.chat(user_input)
        print(f"🤖 챗봇: {response}")

        # 역질문이 끝나고 최종 답변이 나왔으면 종료
        if not bot._pending_subject:
            break

        # 역질문 subject에 맞는 답변 찾기
        subject = bot._pending_subject
        if subject in scenario["answers"]:
            user_input = scenario["answers"][subject]
        else:
            # 시나리오에 없는 역질문은 기본값 "아니요"로 응답
            print(f"  ⚠️ '{subject}' 에 대한 시나리오 답변 없음 → 기본값 '아니요' 사용")
            user_input = "아니요"

        turn += 1

    print(f"\n{'─' * 60}")


if __name__ == "__main__":
    scenarios = [
        {
            "name": "증상만 입력",
            "first_input": "알러지 반응이 올라와. 어떤 약을 먹으면 좋을까?",
        },
        {
            "name": "타이레놀 - 간질환 해당",
            "first_input": "두통이 있는데 타이레놀 먹어도 되나요?",
            "answers": {
                "간질환자":  "네, 간염이 있어요",
                "알코올":   "아니요",
                "알레르기": "아니요",
            }
        },
        {
            "name": "이부프로펜 - 특이사항 없음",
            "first_input": "이부프로펜 먹으려고요",
            "answers": {
                "임산부":    "아니요",
                "소아":      "아니요",
                "신장질환자": "아니요",
                "소화성 궤양": "아니요",
            }
        },
    ]

    for scenario in scenarios:
        run_clarify_test(scenario)

