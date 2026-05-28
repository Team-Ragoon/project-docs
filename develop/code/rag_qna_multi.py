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
from answer_parser import parse_yes_no, parse_age
from medication_loader import get_drug_info, is_acetaminophen_only, is_diarrhea_medicine,  has_corydalis
from stdio_utf8 import configure_stdio_utf8

load_dotenv()
configure_stdio_utf8()


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
llm = ChatOpenAI(model="gpt-5-mini", temperature=0)


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


# 흐름 A 최종 추천 완료 감지 함수
# LLM 응답에 "💊" 또는 "추천 약:"이 포함되면 역질문 종료로 간주
def is_final_recommendation(response: str) -> bool:
    return "💊" in response or "추천 약:" in response or "🚫" in response


# 약 이름으로 ChromaDB에서 해당 약의 문서를 직접 가져오는 함수
# 흐름 A 역질문 중 사용자가 약 이름을 언급하면 context에 추가하기 위해 사용
def fetch_drug_documents(drug_names: list[str]) -> str:
    if not drug_names:
        return ""
    try:
        results = collection.get(where={"drug_name": {"$in": drug_names}})
        return "\n\n".join(results["documents"]) if results["documents"] else ""
    except Exception as e:
        print(f"[fetch_drug_documents 오류] {e}")
        return ""


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

    def _flow_a_reply(self, user_input: str, *, first_turn: bool) -> str:
        
        # 흐름 A 역질문 진행 중 (상태 플래그로 판단 — 패턴 매칭보다 안전)
        # 검색 스킵 + 첫 턴에서 캐시된 context 재사용 + chat_history로 LLM이 다음 질문/최종 답변 생성
        if first_turn:
            self.state._cached_context = retriever_multi(user_input)
            self.state._in_flow_a_clarify = True
        else:
            # 사용자가 새 약 이름을 언급했는지 탐지 -> 발견 시 context에 해당 약 문서 추가
            # (예: "복용 중인 약?" → "판콜" → 판콜 문서를 context에 보강)
            mentioned_drugs, _ = detect_drugs_in_text(user_input)
            if mentioned_drugs:
                extra_docs = fetch_drug_documents(mentioned_drugs)
                if extra_docs:
                    self.state._cached_context += "\n\n" + extra_docs
                    print(f"[context 보강] 추가된 약: {mentioned_drugs}")

        response = self.rag_chain.invoke({
            "context": self.state._cached_context,
            "question": user_input,
            "chat_history": self.state.get_history(),
        })
        self.state.add_turn(user_input, response)
        # 첫 턴에서 바로 최종 답변이 나온 경우 (정보가 충분했던 케이스)
        # 최종 추천이 나오면 흐름 A 종료
        if is_final_recommendation(response):
            self.state._in_flow_a_clarify = False
        return response

    def chat(self, user_input: str) -> str:
        if self.state._in_flow_a_clarify:
            return self._flow_a_reply(user_input, first_turn=False)
        # 흐름 B 역질문 진행.
        if self._pending_subject:
            if self._pending_subject == "나이":
                # 나이는 숫자로 추출 후 extra_context에 저장
                # 나이 금기 슬롯이 있을 때만 caution_slots에도 저장
                # 금기 슬롯이 없는 경우(제형 구분 목적)는 extra_context만 사용
                age = parse_age(user_input, llm)
                contraindications = self.validator.get_contraindications(self.state.drug_names)
                age_slot = next((c for c in contraindications if c["subject"] == "나이"), None)
                if age is not None:
                    self.state.extra_context["나이"] = age
                    contraindications = self.validator.get_contraindications(self.state.drug_names)
                    age_slot = next((c for c in contraindications if c["subject"] == "나이"), None)
                    is_contraindicated = (
                        age_slot is not None
                        and any(
                            (age <= t["age"] if t["inclusive"] else age < t["age"])
                            for t in age_slot.get("age_thresholds", {}).values()
                        )
                    )
                    self.state.caution_slots["나이"] = is_contraindicated
                else:
                    # 나이를 전혀 파악할 수 없는 경우: 나이 금기 슬롯이 있을 때만 모름으로 처리
                    if age_slot is not None:
                        self.state.caution_slots["나이"] = None
            else:
                # 나이 외 슬롯: 기존 yes/no 판단
                is_positive = parse_yes_no(
                    user_input=user_input,
                    question=self._pending_question,
                    subject=self._pending_subject,
                    llm=llm,
                )
                self.state.record_clarify_answer(self._pending_subject, is_positive)

            self._pending_subject = None
            self._pending_question = None


            # 확인용 출력 -----------------------------------------
            # print("\n[caution_slots 현재 상태]")
            # for subject, value in self.state.caution_slots.items():
            #     status = "해당" if value else "해당 없음" if value is False else "미응답"
            #     print(f"  {subject}: {status}")
            # print("\n")
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
        #print(f"[analysis 전체] {analysis}")
        #--------------------------------------------

        # analyzer가 미리 탐지하지만 보완하는 역할. (잘못 판단한 경우를 대비)
        # 단, analyzer가 "comparison"으로 판단한 경우는 덮어쓰지 않음 (비교 질문은 흐름 A에서 처리)
        if matched_drugs and analysis.get("query_type") != "comparison":
            analysis["query_type"] = "medication"
            self.state.set_drug_candidates(matched_drugs, user_keyword)
        elif not analysis.get("query_type") and analysis.get("symptom"):
                analysis["query_type"] = "symptom_only"
            
        # 상태 update
        self.state.update_from_analysis(analysis)

        # 확인용 출력 -----------------------------------
        #print(f"[drug_names] {self.state.drug_names}")
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
            #print(f"[caution_slots] {self.state.caution_slots}")
            #print(f"[extra_context] {self.state.extra_context}")
            #-----------------------------------------------------
        

        # 흐름 A : 증상만 입력 또는 비교 질문 -> 시스템 프롬프트 9번 역질문 규칙에 따라 LLM이 자율 진행
        # 첫 턴이므로 chat_history는 빈 리스트. LLM이 필요한 정보 부족하면 역질문 생성.
        # 검색한 context를 캐시 -> 역질문 턴에서도 재사용
        # 플래그를 ON -> 이후 사용자 답변은 흐름 A 역질문 응답으로 처리
        # 비교 질문은 SYSTEM_PROMPT 8번 규칙에 따라 역질문 없이 즉시 답변됨
        if self.state.query_type in ("symptom_only", "comparison"):
            return self._flow_a_reply(user_input, first_turn=True)

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
        # 나이 선행 질문: caution 슬롯 여부와 무관하게 항상 나이를 먼저 확인
        # 어린이/성인 제형 구분 및 연령별 금기 자동 처리에 필요
        # 숫자 나이가 이미 추출된 경우에만 건너뜀 (소아여부만 있는 경우는 질문)
        if "나이" not in self.state.extra_context:
            question = "복용하실 분의 정확한 나이를 알려주세요. (예: 7세, 30세)"
            self._pending_subject = "나이"
            self._pending_question = question
            return question  # clarify_count 증가 없음 — 필수 선행 질문

        # 현재까지 True인 슬롯 기준으로 복용 가능한 약 먼저 계산
        applicable_subjects = {s for s, v in self.state.caution_slots.items() if v is True}
        can_drugs = (
            self._get_remaining_candidates(applicable_subjects)
            if applicable_subjects
            else list(self.state.drug_names)
        )

        # 복용 가능한 약이 없으면 최종 답변 (모두 금기)
        if not can_drugs:
            return self._generate_final_answer()

        # 임산부 선행 질문: can_drugs에 설사약 또는 현호색/연호색 성분 약이 있고 임산부 여부 미확인인 경우
        # DB 금기 미기재 약도 임신 중 위험 성분(현호색 등) 또는 설사약은 반드시 확인 필요
        needs_preg_check = any(
            is_diarrhea_medicine(d) or has_corydalis(d)
            for d in can_drugs
        )
        preg_known = "임산부" in self.state.caution_slots or "임산부" in self.state.extra_context
        if needs_preg_check and not preg_known and not self.validator._should_skip(
            "임산부", self.state.caution_slots, self.state.extra_context
        ):
            question = "혹시 임산부이시거나 임신 가능성이 있으신가요?"
            self._pending_subject = "임산부"
            self._pending_question = question
            return question  # clarify_count 증가 없음 — 필수 선행 질문

        # 캐시 조회는 전체 drug_names로(재파싱 방지), 질문 필터링은 can_drugs로
        if self.validator.should_clarify(
            self.state.drug_names,
            self.state.caution_slots,
            self.state.clarify_count,
            self.state.extra_context,
            can_drugs=can_drugs,
        ):
            # should_clarify 내부에서 나이 슬롯이 자동 채워졌을 수 있음 → can_drugs 재계산
            updated_subjects = {s for s, v in self.state.caution_slots.items() if v is True}
            if updated_subjects != applicable_subjects:
                can_drugs = self._get_remaining_candidates(updated_subjects)
                if not can_drugs:
                    return self._generate_final_answer()

            slot = self.validator.get_priority_slot(
                self.state.drug_names,
                self.state.caution_slots,
                self.state.extra_context,
                can_drugs=can_drugs,
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
            if value is True:
                status = "금기 해당"
            elif value is False:
                status = "해당 없음"
            else:
                status = "확인 불가"
            lines.append(f"-{subject} : {status}")
        return "\n".join(lines) if lines else "- 특이사항 없음"
    
    # 사용자의 상황 중 금기 사항이 아닌 상황을 text로 변환.
    def _build_extra_context(self) -> str:
        if not self.state.extra_context:
            return ""
        lines = []
        for subject, value in self.state.extra_context.items():
            if subject == "소아여부" and value is True:
                lines.append("- 어린이/소아 복용 예정")  # LLM이 이해하기 쉽게
            elif subject == "나이" and isinstance(value, float):
                lines.append(f"- 나이: {value}세")
            else:
                status = "해당" if value else "해당 없음"
                lines.append(f"- {subject}: {status}")
        return "\n".join(lines)
    
    # 어린이 / 소아로 확인된 경우 키즈/어린이 제형만 반환. / 없으면 전체 반환.
    def _filter_candidates_by_age(self, drug_names: list[str] = None) -> list[str]:
        names = drug_names if drug_names is not None else self.state.drug_names
        context = self.state.caution_slots
        extra = self.state.extra_context
        
        user_age = extra.get("나이")
        is_child = (
            (user_age is not None and float(user_age) <= 10)
            or context.get("나이") is True
            or extra.get("소아여부") is True
        )
        
        KIDS_KEYWORDS = ["키즈", "소아", "아동", "어린이"]

        if is_child:
            kids_drugs = [d for d in names if any(k in d for k in KIDS_KEYWORDS)]
            return kids_drugs if kids_drugs else names

        # 성인: 키즈 제형 제외
        adult_drugs = [d for d in names if not any(k in d for k in KIDS_KEYWORDS)]
        return adult_drugs if adult_drugs else names

    # 약별 성분·효능·이상반응 요약 (RAG 데이터 기반)
    def _build_drug_profiles_for(self, drug_names: list[str]) -> str:
        lines = []
        for name in drug_names:
            drug = get_drug_info(name)
            if not drug:
                continue
            ingredient  = (drug.get("ingredient_api")    or "정보 없음").strip()
            efficacy    = (drug.get("efcyQesitm")        or "정보 없음").strip()[:200]
            side_effect = (drug.get("seQesitm")          or "정보 없음").strip()[:300]
            lines.append(
                f"- {name}\n"
                f"  성분: {ingredient}\n"
                f"  효능·효과: {efficacy}\n"
                f"  이상반응: {side_effect}"
            )
        return "\n\n".join(lines) if lines else "없음"
    
    # 1개 추천 조건 코드로 판별. (LLM 판단만으로 부족함.)
    def _should_recommend_single(self) -> bool:
    
        # 1. 후보 약 자체가 1개
        if len(self.state.drug_names) == 1:
            return True
        
        # 2. 금기 배제 후 남는 약이 1개
        applicable_subjects = {
            subject
            for subject, value in self.state.caution_slots.items()
            if value is True
        }
        if applicable_subjects:
            remaining = self._get_remaining_candidates(applicable_subjects)
            if len(remaining) == 1:
                return True
            if len(remaining) == 0:
                # 아예 추천 가능한 약이 없는 경우.
                # 원래 cannot_recommend로 가야 하지만, 여기에선 False 반환.
                return False 
        
        # 3. 증상이 후보 약 중 1개에만 명시된 경우
        if self._symptom_matches_only_one():
            return True
        
        return False  # 나머지는 모두 2~3개 추천


    # 금기 배제 후 남은 후보 약 목록
    # applicable_drugs 필드로 텍스트 재검색 없이 결정.
    # "나이" subject는 age_thresholds + extra_context["나이"]로 약별 정밀 판단.
    def _get_remaining_candidates(self, applicable_subjects: set) -> list[str]:
        if not applicable_subjects:
            return list(self.state.drug_names)
        contraindications = self.validator.get_contraindications(self.state.drug_names)
        user_age = self.state.extra_context.get("나이")
        excluded = set()
        for c in contraindications:
            subject = c["subject"]
            if subject not in applicable_subjects:
                continue
            if subject == "나이" and user_age is not None:
                # 약별 나이 기준으로 개별 판단 (합산 기준 아님)
                for drug, threshold in c.get("age_thresholds", {}).items():
                    if user_age <= threshold["age"] if threshold["inclusive"] else user_age < threshold["age"]:
                        excluded.add(drug)
            else:
                excluded.update(c.get("applicable_drugs", self.state.drug_names))
        return [d for d in self.state.drug_names if d not in excluded]


    # 증상이 후보 약 중 오직 1개의 효능에만 포함되는지 확인.
    def _symptom_matches_only_one(self) -> bool:
        symptoms = [s.strip() for s in (self.state.symptom or "").split(",") if s.strip()]
        
        if not symptoms:
            return False

        efficacy_map = {}
        for name in self.state.drug_names:
            drug = get_drug_info(name)
            if not drug:
                continue
            efficacy_map[name] = (drug.get("efcyQesitm") or "")

        # 증상별로 몇 개 약에 포함되는지 확인
        for symptom in symptoms:
            matched = [
                name for name, efficacy in efficacy_map.items()
                if symptom in efficacy
            ]
            # 이 증상이 오직 1개 약에만 있으면 → 1개 추천
            if len(matched) == 1:
                return True

        return False


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

        # applicable_subjects는 applicable에서 추출
        applicable_subjects = {
            c["subject"]
            for c in all_contraindications
            if self.state.caution_slots.get(c["subject"]) is True
        }

        # ── 확인용 print ────────────────────────────────
        # print(f"[caution_slots] {self.state.caution_slots}")
        # print(f"[applicable] {applicable}")
        # print(f"[applicable_subjects] {applicable_subjects}")
        # ─────────────────────────────────────────────

        can_drugs = self._get_remaining_candidates(applicable_subjects) if applicable_subjects else drug_names

        # 임산부 guard: extra_context에서 임산부 확인 시 아세트아미노펜 단일 성분 약만 허용
        if self.state.extra_context.get("임산부") is True:
            safe_drugs = [d for d in can_drugs if is_acetaminophen_only(d)]
            if not safe_drugs:
                answer = self.cannot_recommend.invoke({
                    "drug_keyword": drug_keyword,
                    "user_profile": user_profile,
                    "applicable_cautions": "- 임산부: 후보 약이 모두 복합 성분으로, 임산부에게는 아세트아미노펜 단일 성분 약만 권장됩니다.",
                })
                return f"{summary}\n\n{answer}"
            can_drugs = safe_drugs

        # 임산부 + 설사약 guard: 설사약은 임산부에게 안전성 미확인으로 추천 불가
        if self.state.extra_context.get("임산부") is True:
            non_diarrhea = [d for d in can_drugs if not is_diarrhea_medicine(d)]
            if len(non_diarrhea) < len(can_drugs):
                if not non_diarrhea:
                    answer = self.cannot_recommend.invoke({
                        "drug_keyword": drug_keyword,
                        "user_profile": user_profile,
                        "applicable_cautions": "- 임산부: 설사약은 임신 중 안전성이 확인되지 않아 복용을 권장하지 않습니다.",
                    })
                    return f"{summary}\n\n{answer}"
                can_drugs = non_diarrhea

        # 임산부 + 현호색/연호색 성분 guard: 현호색 성분은 임산부 복용 금기
        if self.state.extra_context.get("임산부") is True:
            non_corydalis = [d for d in can_drugs if not has_corydalis(d)]
            if len(non_corydalis) < len(can_drugs):
                if not non_corydalis:
                    answer = self.cannot_recommend.invoke({
                        "drug_keyword": drug_keyword,
                        "user_profile": user_profile,
                        "applicable_cautions": "- 임산부: 현호색(연호색) 성분이 포함된 약은 임신 중 복용을 권장하지 않습니다.",
                    })
                    return f"{summary}\n\n{answer}"
                can_drugs = non_corydalis

        # 복용 가능한 약이 없는 경우 → 전체 복용 불가
        if not can_drugs or (applicable and set(can_drugs) == set(drug_names)):
            answer = self.cannot_recommend.invoke({
                "drug_keyword": drug_keyword,
                "user_profile": user_profile,
                "applicable_cautions": "\n".join(applicable),
            })
            return f"{summary}\n\n{answer}"
        
        # 확인 안 된 slot (역질문 하지 않은 사항들)
        unchecked = [
            c["subject"]
            for c in all_contraindications
            if self.state.caution_slots.get(c["subject"]) is None
            and not self.validator._should_skip(c["subject"], self.state.caution_slots, self.state.extra_context)
        ]


        # 연령 기반 후보 필터링
        filtered_names = self._filter_candidates_by_age(can_drugs)
        #print(f"[can_drugs] {can_drugs}")

        # 필터링된 후보로 1개 추천 여부 판별
        original_names = self.state.drug_names
        self.state.drug_names = filtered_names  # 임시 교체
        recommend_single = self._should_recommend_single()
        recommend_count = "1개" if recommend_single else "2~3개"
        self.state.drug_names = original_names  # 복구

        drug_profiles = self._build_drug_profiles_for(filtered_names)

        answer = self.recommend_final.invoke({
            "drug_keyword" : drug_keyword,
            "symptom" : self.state.symptom or "언급 없음",
            "drug_candidates" : "\n".join(f"- {d}" for d in filtered_names),
            "drug_profiles" : drug_profiles, 
            "user_profile" : user_profile,
            "applicable_cautions": "특별한 금기사항 해당 없음",
            "extra_context" : extra_context or "없음",
            "unchecked_cautions": ", ".join(unchecked) if unchecked else "없음",
            "recommend_count" : recommend_count,
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

SCENARIOS = [
    # ── 소화제 / 흐름 A (증상만) ──────────────────────────
    {
        "name": "[소화제/흐름A] 소화불량 증상만 - 성인",
        "first_input": "밥 먹고 나서 속이 더부룩하고 소화가 안 돼",
        "answers": {
            "나이":         "30살",
            "임산부/수유부": "아니요",
        },
    },
    {
        "name": "[소화제/흐름A] 소화불량 - 5세 어린이",
        "first_input": "밥 먹고 나서 속이 더부룩하고 소화가 안 돼",
        "answers": {
            "나이": "5살",
        },
    },
    {
        "name": "[소화제/흐름A] 소화불량 - 2개월 영아 (병원 안내 기대)",
        "first_input": "밥 먹고 나서 속이 더부룩하고 소화가 안 돼",
        "answers": {
            "나이": "2개월",
        },
    },
    {
        "name": "[소화제/흐름A] 과식·체함 - 성인",
        "first_input": "과식해서 속이 너무 답답하고 체한 것 같아",
        "answers": {
            "나이":         "30살",
            "임산부/수유부": "아니요",
        },
    },

    # ── 소화제 / 흐름 B (약 이름 포함) ───────────────────
    {
        "name": "[소화제/흐름B] 훼스탈 - 성인 복용 가능",
        "first_input": "훼스탈 먹어도 될까?",
        "answers": {
            "나이":         "아니요",
            "임산부/수유부": "아니요",
        },
    },
    {
        "name": "[소화제/흐름B] 훼스탈 - 7살 아이 (만 7세 이하 금기)",
        "first_input": "훼스탈 먹어도 될까? 7살 아이에게 줘도 되나요?",
        "answers": {},
    },
    {
        "name": "[소화제/흐름B] 베아제 - 임산부",
        "first_input": "베아제 먹어도 돼? 나 임산부야",
        "answers": {},
    },

    # ── 복통약 / 흐름 A (증상만) ──────────────────────────
    {
        "name": "[복통/흐름A] 설사 증상만 - 성인",
        "first_input": "설사가 심하고 배가 아파",
        "answers": {
            "나이":         "30살",
            "임산부/수유부": "아니요",
            "유당불내증":   "아니요",
        },
    },
    {
        "name": "[복통/흐름A] 설사 - 유당불내증 있음",
        "first_input": "설사가 심하고 배가 아파",
        "answers": {
            "나이":         "30살",
            "임산부/수유부": "아니요",
            "유당불내증":   "네",
        },
    },
    {
        "name": "[복통/흐름A] 과민성대장증후군 의심 - 성인",
        "first_input": "배가 계속 아프고 설사랑 변비가 반복돼",
        "answers": {
            "나이":         "30살",
            "임산부/수유부": "아니요",
            "유당불내증":   "아니요",
        },
    },

    # ── 복통약 / 흐름 B (약 이름 포함) ───────────────────
    {
        "name": "[복통/흐름B] 장엔폴 - 성인 복용 가능",
        "first_input": "장엔폴 먹어도 돼?",
        "answers": {
            "나이":       "아니요",
            "임산부/수유부": "아니요",
            "유당불내증": "아니요",
        },
    },
    {
        "name": "[복통/흐름B] 장엔폴 - 유당불내증 (동성정로환 대체 기대)",
        "first_input": "장엔폴 먹어도 돼?",
        "answers": {
            "나이":       "아니요",
            "임산부/수유부": "아니요",
            "유당불내증": "네",
        },
    },
    {
        "name": "[복통/흐름B] 장엔폴 - 임산부 (복용 불가 기대)",
        "first_input": "장엔폴 먹으려는데 임신 중이야",
        "answers": {
            "나이": "아니요",
        },
    },
    {
        "name": "[복통/흐름B] 동성정로환 - 성인 복용 가능",
        "first_input": "동성정로환 먹어도 돼?",
        "answers": {
            "나이": "아니요",
        },
    },
    {
        "name": "[복통/흐름B] 타라부틴 - 유당불내증 (복용 불가 기대)",
        "first_input": "타라부틴 먹어도 돼?",
        "answers": {
            "유당불내증": "네",
        },
    },
]


if __name__ == "__main__":
    print("=" * 60)
    print("  의약품 RAG QnA - Query Expansion 버전")
    print("=" * 60)
    print("\n실행 모드를 선택하세요:")
    print("  1. 시나리오 테스트 (자동 실행)")
    print("  2. 직접 질문 입력")

    mode = input("\n선택 (1/2): ").strip()
    if mode == "1":
        for scenario in SCENARIOS:
            run_clarify_test(scenario)
    
    elif mode == "2":
        run_interactive()        

    else:
        print("잘못된 입력입니다. 1/2 중에서 선택하세요.")
