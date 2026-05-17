"""
RAG 기반 의약품 QnA 시스템 - Query Expansion 버전
- 임베딩: BGE-M3
- 벡터 DB: ChromaDB
- LLM: GPT-4o-mini (OpenAI)
- 추가: Query Expansion (LLM으로 쿼리 자동 확장)
"""

import json
from pathlib import Path
import chromadb
from dotenv import load_dotenv
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from prompts.system_prompt import EXPAND_SYSTEM_PROMPT
from validator import SlotValidator
from analyzer import build_analyzer_chain, build_situation_extractor_chain
from clarifier import build_clarifier_chain
from rag import (
    build_rag_chain,build_summarize_chain,
    build_cannot_recommend_chain, build_recommend_final_chain)
from dialogue import DialogueState
from drug_detector import detect_drugs_in_text
from answer_parser import parse_yes_no

load_dotenv()


# 1. ChromaDB 로드
_DEVELOP_ROOT = Path(__file__).resolve().parent.parent
CHROMA_PATH = str(_DEVELOP_ROOT / "chroma_db")
_DRUG_DOCUMENTS_PATH = _DEVELOP_ROOT / "dataset" / "drug_documents.json"

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
    with open(_DRUG_DOCUMENTS_PATH, "r", encoding="utf-8") as f:
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
    queries = [question] + extra_queries[:3]  # 원본 + 최대 3개
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
                all_docs[key] = {"doc": doc, "meta" : meta, "dist": dist}
    

    # 유사도 순 정렬 후 상위 5개
    sorted_docs = sorted(all_docs.values(), key=lambda x: x["dist"])[:5]


    # 확인용 출력 코드--------------------------------------
    print(f"\n[검색된 문서 목록]")
    for i, d in enumerate(sorted_docs, 1):
        print(f"  {i}. 유사도: {d['dist']:.4f}")
        print(f"     약품명: {d['meta'].get('drug_name')}") 
    #-------------------------------------------------------

    return "\n\n".join([d["doc"] for d in sorted_docs])


# Chatbot class 선언
class MedicalChatbot:
    def __init__(self):
        self.state = DialogueState()
        self.validator = SlotValidator(llm = llm)
        self.analyzer = build_analyzer_chain(llm = llm)
        self.situation_extractor = build_situation_extractor_chain(llm = llm)
        self.clarifier = build_clarifier_chain(llm = llm)
        self.recommend_final = build_recommend_final_chain(llm = llm)
        self.rag_chain = build_rag_chain(llm = llm)
        self.summarizer = build_summarize_chain(llm = llm)
        self.cannot_recommend = build_cannot_recommend_chain(llm = llm)
        self._pending_subject: str | None = None
        self._pending_question: str | None = None
    
    def chat(self, user_input: str) -> str:

        # 역질문 진행.
        if self._pending_subject:
            # 역질문에 대한 사용자의 답변의 긍정/부정 여부를 판단. 
            is_positive = parse_yes_no(
                user_input = user_input,
                question = self._pending_question,
                subject = self._pending_subject,
                llm = llm
            )
            # 긍정/부정 여부를 질문과 함께 저장한 후, 다음 질문을 위한 초기화.
            self.state.record_clarify_answer(self._pending_subject, is_positive)
            self._pending_subject = None
            self._pending_question = None


            # 확인용 출력 -----------------------------------------
            print("\n[caution_slots 현재 상태]")
            for subject, value in self.state.caution_slots.items():
                status = "해당" if value else "해당 없음" if value is False else "미응답"
                print(f"  {subject}: {status}")
            #-------------------------------------------------------

            # 다음 역질문 / 최종 답변으로 이동
            return self._next_clarify_or_answer(user_input)
        

        # 새로운 query에 대한 응답을 위해 초기화 진행.
        self.state.start_new_turn()
        

        # 사용자가 입력한 약의 이름과 matching되는 약들을 DB에서 찾기.
        matched_drugs, user_keyword = detect_drugs_in_text(user_input)


        # 사용자의 입력 -> query의 종류 및 증상 판단.
        analysis = self.analyzer.invoke({
            "user_input" : user_input,
        })

        # 확인용 출력 --------------------------------
        print(f"[analysis 전체] {analysis}")
        #--------------------------------------------

        # analyzer가 미리 탐지하지만 보완하는 역할. (잘못 판단한 경우를 대비)
        if matched_drugs:
            analysis["query_type"] = "medication"
            self.state.set_drug_candidates(matched_drugs, user_keyword)
        elif not analysis.get("query_type") and analysis.get("symptom"):
                analysis["query_type"] = "symptom_only"
            
        # 상태 update
        self.state.update_from_analysis(analysis)

        # 확인용 출력 -----------------------------------
        print(f"[drug_names] {self.state.drug_names}")
        #------------------------------------------------

        # query_type이 medication이라면, query에 사용자의 상황이 포함되어 있는지 확인.
        if self.state.query_type == "medication":
            try:
                situation = self.situation_extractor.invoke({
                    "user_input" : user_input
                })
            except Exception as e:
                print(f" [상황 추출 실패] {e}")
                situation = {}
            

            # query에 사용자의 상황이 포함되어 있는 경우
            # 사용자의 상황 중 약들의 금기 사항에 포함되어 있는지 먼저 확인. 
            if situation:
                caution_subjects = [
                    c["subject"]
                    for c in self.validator.get_contraindications(self.state.drug_names)
                ] if self.state.drug_names else []

                self.state.apply_extracted_situation(situation, caution_subjects)

                #미리 채워진 슬롯 수만큼 clarify_count 차감
                self.state.clarify_count += len(self.state.caution_slots)

            # 확인용 출력-----------------------------------------
            print(f"[caution_slots] {self.state.caution_slots}")
            print(f"[extra_context] {self.state.extra_context}")
            #-----------------------------------------------------
        

        # 흐름 A : 증상만 입력 -> 약 추천. (아직 필수 역질문 구현 X)
        if self.state.query_type == "symptom_only":
            context = retriever_multi(user_input)

            response = self.rag_chain.invoke({
                    "context" : context,
                    "question": user_input,
            })
            self.state.add_turn(user_input, response)
            return response


        # 흐름 B : 질문에 약 이름이 포함되어 있음 => 주의사항 기반 역질문
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
            self.state.extra_context,
        ):
            slot = self.validator.get_priority_slot(
                self.state.drug_names,
                self.state.caution_slots,
                self.state.extra_context,
            )
            response = self.clarifier.invoke({
                "filled_slots": self.state.caution_slots,
                "subject":      slot["subject"],
                "reason":       slot["reason"],
                "question":     slot["question"],
            })
            self._pending_subject = slot["subject"]
            self._pending_question = response
            self.state.clarify_count += 1
            return response

        # 모든 슬롯 충족 → 최종 답변
        return self._generate_final_answer()


    # caution_slots을 자연어로 변환하는 함수.
    def _build_user_profile(self) -> str:
        lines = []
        for subject, value in self.state.caution_slots.items():
            status = "해당" if value else "해당 없음"
            lines.append(f"-{subject} : {status}")
        return "\n".join(lines) if lines else "- 특이사항 없음"
    
    # 사용자의 상황 중 금기 사항이 아닌 상황을 text로 변환.
    def _build_extra_context(self) -> str:
        if not self.state.extra_context:
            return ""
        lines = []
        for subject, value in self.state.extra_context.items():
            status = "해당" if value else "해당 없음"
            lines.append(f"- {subject} : {status}")
        return "\n".join(lines)

    # 역질문 후 최종 답변 생성.(요약 -> 약 복용 가능 여부를 판단)
    def _generate_final_answer(self) -> str:
        drug_names =self.state.drug_names
        drug_keyword = self.state.drug_keyword
        all_contraindications = self.validator.get_contraindications(drug_names)
        user_profile = self._build_user_profile()
        extra_context = self._build_extra_context()

        # 최종 답변 전에 언급할 사용자의 현 상황 요약.
        summary = self.summarizer.invoke({
            "drug_name" : drug_keyword,
            "symptom" : self.state.symptom or "언급 없음",
            "user_profile" : user_profile,
        })

        # 사용자가 해당 약을 복용할 수 있는지 판단.
        applicable = [
            f"- {c['subject']}: {c['reason']}"
            for c in all_contraindications
            if self.state.caution_slots.get(c["subject"]) is True
        ]

        if applicable:
            # 복용 불가 -> 이유 설명
            answer = self.cannot_recommend.invoke({
                "drug_keyword" : drug_keyword,
                "user_profile" : user_profile,
                "applicable_cautions" : "\n".join(applicable),
            })
        else:
            # 복용 가능 -> 약 추천.
            answer = self.recommend_final.invoke({
                "drug_keyword" : drug_keyword,
                "drug_candidates" : "\n".join(f"- {d}" for d in drug_names),
                "user_profile" : user_profile,
                "applicable_cautions": "특별한 금기사항 해당 없음",
                "extra_context" : extra_context or "없음",
            }) 

        return f"{summary}\n\n{answer}"


"""
역질문 흐름 테스트
- 챗봇이 역질문을 하면 미리 정의된 답변으로 응답
- 실제 대화 흐름과 동일하게 턴 단위로 진행
"""

def run_clarify_test(scenario: dict):
    print(f"\n{'=' * 60}")
    print(f"  테스트: {scenario['name']}")
    print(f"{'=' * 60}")

    bot = MedicalChatbot()
    MAX_TURNS = 15  # 무한 루프 방지
    # 첫 턴 : 질문
    user_input = scenario["first_input"]
    turn = 0

    while turn < MAX_TURNS:
        print(f"\n🙋 사용자: {user_input}")
        response = bot.chat(user_input)
        print("\n")
        print(f"🤖 챗봇: {response}")

        # 역질문이 끝나고 최종 답변이 나왔으면 종료
        if not bot._pending_subject:
            break

        # 역질문 subject에 맞는 답변 찾기
        subject = bot._pending_subject
        answers = scenario.get("answers", {})
        if subject in answers:
            user_input = answers[subject]
            print(f" -> '{subject}' 답변: {user_input}")
        else:
            # 시나리오에 없는 역질문은 기본값 "아니요"로 응답
            print(f"  ⚠️ '{subject}' 에 대한 시나리오 답변 없음 → 기본값 '아니요' 사용")
            user_input = "아니요"

        turn += 1

    print(f"\n{'─' * 60}")

def run_interactive():
    """
    직접 질문 입력 모드
    역질문이 발생하면 사용자가 직접 답변 입력
    """
    bot = MedicalChatbot()
    print("\n질문을 입력하세요. 종료하려면 'q' 또는 'quit'을 입력하세요.\n")

    while True:
        # 역질문 대기 중이면 답변 입력 안내
        if bot._pending_subject:
            user_input = input(f"💬 '{bot._pending_subject}' 에 대한 답변: ").strip()
            print("\n")
        else:
            user_input = input("🙋 질문: ").strip()

        if not user_input:
            continue
        if user_input.lower() in ("q", "quit"):
            print("\n종료합니다.")
            break

        response = bot.chat(user_input)
        print(f"🤖 챗봇: {response}")


if __name__ == "__main__":
    print("=" * 60)
    print("  의약품 RAG QnA - Query Expansion 버전")
    print("=" * 60)
    print("\n실행 모드를 선택하세요:")
    print("  1. 시나리오 테스트 (자동 실행)")
    print("  2. 직접 질문 입력")

    mode = input("\n선택 (1/2): ").strip()
    if mode == "1":
        scenarios = [
            {
                "name": "증상만 입력(1)",
                "first_input": "비염이 있는데 눈이 따가워.",
            },
            {
                "name": "증상만 입력(2)",
                "first_input": "비염이 있는데 눈이 뜨거워.",
            },
            {
                "name": "증상만 입력(3) ",
                "first_input": "비염 때문에 눈의 작열감이 느껴져.",
            },
            {
                "name": "증상만 입력(4) ",
                "first_input": "비염 때문에 눈에 작열감이 느껴져.",
            },
            # {
            #     "name": "세노바퀵",
            #     "first_input": "알러지 때문에 비염이 심한데 세노바퀵 먹어도 될까?",
            #     "answers": {
            #         "과민증" : "네",
            #         "임산부" : "아니요",
            #     }
            # },
            # {
            #     "name": "세트린",
            #     "first_input": "세트린을 복용하려고 해.",
            #     "answers": {
            #         "임산부":    "네",
            #         "소아":      "아니요",
            #         "간장애": "아니요",
            #         "신장애": "네",
            #     }
            # },
        ]

        for scenario in scenarios:
            run_clarify_test(scenario)
    
    elif mode == "2":
        run_interactive()        

    else:
        print("잘못된 입력입니다. 1/2 중에서 선택하세요.")