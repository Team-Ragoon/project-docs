"""
시나리오를 챗봇에 자동 입력해서 결과 수집

각 시나리오마다:
1. 챗봇 인스턴스 새로 생성 (이전 대화 영향 차단)
2. first_input 전달 → 챗봇 응답 수집
3. 챗봇이 역질문하면 user_answers에서 매칭되는 답변 전달
4. 최종 추천이 나올 때까지 반복
5. 결과를 JSON으로 저장

수집 정보:
- 최종 답변 (final_answer)
- 전체 대화 (turns)
- 검색된 문서 (contexts) - retriever_multi 호출 시점에 수집
- 사용된 user_answer (실제 매칭된 답변)
"""

import sys
import json
import io
from pathlib import Path

# develop/code 경로를 sys.path에 추가 (챗봇 모듈 import용)
_HERE = Path(__file__).resolve().parent
_CODE_DIR = _HERE.parent / "code"
sys.path.insert(0, str(_CODE_DIR))

# 한글 출력 안정화 + line_buffering=True 로 즉시 출력 (버퍼링 방지)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", line_buffering=True)

from scenarios import SCENARIOS


# 챗봇 모듈 import (전역 LLM/ChromaDB가 초기화됨)
# import 시 시간이 좀 걸림 (BGE-M3 로드)
print("⏳ 챗봇 모듈 로드 중...")
import rag_qna_multi
from rag_qna_multi import MedicalChatbot, retriever_multi, fetch_drug_documents
print("✅ 챗봇 모듈 로드 완료\n")


# retriever_multi 호출 결과를 가로채서 수집하는 래퍼
# (시나리오 실행 중 어떤 문서가 검색됐는지 추적)
_original_retriever_multi = retriever_multi
_collected_contexts: list[str] = []


def _wrapped_retriever_multi(question: str, n_results: int = 5) -> str:
    """retriever_multi 호출 시 결과를 _collected_contexts에 추가"""
    result = _original_retriever_multi(question, n_results)
    _collected_contexts.append(result)
    return result


# 챗봇 모듈에서 retriever_multi를 가로채기
rag_qna_multi.retriever_multi = _wrapped_retriever_multi


def find_answer_for_question(ai_question: str, user_answers: dict) -> str:
    """
    챗봇이 던진 역질문(ai_question)에 매칭되는 사용자 답변을 찾기.

    매칭 규칙:
    1. 흐름 B 슬롯 이름이 질문에 포함되면 → 해당 답변
    2. 흐름 B 슬롯 이름의 일부 토큰이 질문에 포함되면 → 해당 답변
    3. 흐름 A 표준 패턴 (나이/임산부/복용약) 매칭
    4. 매칭 안 되면 "아니요" 반환
    """
    # 1. 정확한 subject 매칭 (흐름 B용, 가장 우선)
    # 긴 subject 먼저 매칭 (특이성 우선)
    for subject in sorted(user_answers.keys(), key=len, reverse=True):
        if subject in ai_question:
            return user_answers[subject]

    # 2. 흐름 A 표준 패턴 매칭
    if any(kw in ai_question for kw in ["나이", "몇 살", "몇세", "만 나이"]):
        # "나이"가 시나리오에 없으면 "모르겠어요"로 응답 (무한 재질문 방지)
        # — "아니요"는 챗봇이 나이 답변으로 인식 못해 재질문 유발
        return user_answers.get("나이", "모르겠어요")
    if any(kw in ai_question for kw in ["임산부", "임신"]):
        return user_answers.get("임산부", user_answers.get("임산부/수유부", "아니요"))
    if any(kw in ai_question for kw in ["수유", "모유"]):
        return user_answers.get("수유부", user_answers.get("임산부/수유부", "아니요"))
    if any(kw in ai_question for kw in ["복용 중", "복용하고", "복용하시", "다른 약", "감기약"]):
        # 우선순위: 복용약 → 해열진통제 복용자 → 아세트아미노펜
        for key in ["복용약", "해열진통제 복용자", "아세트아미노펜 최대 용량", "아세트아미노펜 함유 제품 복용자"]:
            if key in user_answers:
                return user_answers[key]
        return "아니요"

    # 3. 키워드 부분 매칭 (subject의 토큰이 질문에 포함되는지)
    for subject, answer in user_answers.items():
        # subject를 공백·괄호로 분리한 토큰들 중 하나라도 질문에 있으면 매칭
        tokens = subject.replace("(", " ").replace(")", " ").replace("/", " ").split()
        for token in tokens:
            if len(token) >= 2 and token in ai_question:
                return answer

    return "아니요"


def run_scenario(scenario: dict, max_turns: int = 10) -> dict:
    """
    하나의 시나리오를 챗봇에 실행하고 결과 수집.

    Returns:
        {
            "scenario_id": "...",
            "turns": [{"role": "user", "content": "..."}, ...],
            "final_answer": "...",
            "contexts": [...],
            "completed": True/False,
            "turn_count": N,
        }
    """
    global _collected_contexts
    _collected_contexts = []  # 시나리오마다 초기화

    bot = MedicalChatbot()
    turns = []
    user_input = scenario["user_input"]
    user_answers = scenario["user_answers"]
    final_answer = None

    for turn_idx in range(max_turns):
        turns.append({"role": "user", "content": user_input})
        response = bot.chat(user_input)
        turns.append({"role": "assistant", "content": response})

        # 종료 판정 (확장):
        # - 추천 키워드: 💊, 🚫, 추천 약:
        # - 비추천/거부 안내: 음주, 임산부 항히스타민, 영아 등 (정상 종료)
        # - 분류 실패 거부 응답
        recommendation_kw = ["💊", "🚫", "추천 약:"]
        refusal_kw = [
            "추천드리기 어렵", "추천해드리기 어렵", "추천드릴 수 없",
            "복용하지 마세요", "복용을 권하지 않",
            "상비약 구매에 관련된 질문에만",  # 분류 실패 거부
        ]
        is_refusal = any(kw in response for kw in refusal_kw)
        is_recommendation = any(kw in response for kw in recommendation_kw)
        flow_a_done = scenario["flow"] == "A" and not bot.state._in_flow_a_clarify
        flow_b_done = scenario["flow"] == "B" and bot._pending_subject is None

        # 거부 답변(음주 후 추천 불가 등)은 역질문이 아니므로 무조건 종료.
        # 챗봇이 '💊/🚫' 기호 없는 순수 텍스트 거부를 줄 때 _in_flow_a_clarify가
        # 풀리지 않아(is_final_recommendation 미인식) 동일 답변이 반복되는 것을 차단.
        if is_refusal:
            final_answer = response
            break
        if is_recommendation and (flow_a_done or flow_b_done):
            final_answer = response
            break

        # 안전망: 직전 챗봇 응답과 거의 동일하면 무한 반복으로 간주
        if len(turns) >= 4:
            prev_resp = turns[-3]["content"]  # 이전 assistant 응답
            if response[:80] == prev_resp[:80]:  # 앞 80자 동일
                final_answer = response
                print(f"  ⚠️ 응답 반복 감지 → 종료")
                break

        # 다음 사용자 답변 결정
        if scenario["flow"] == "B" and bot._pending_subject:
            # 흐름 B: 정확히 subject 매칭
            user_input = user_answers.get(bot._pending_subject, "아니요")
        else:
            # 흐름 A: 질문 패턴 매칭
            user_input = find_answer_for_question(response, user_answers)

    if final_answer is None and turns:
        # 종료 못한 경우 마지막 응답을 final_answer로
        final_answer = turns[-1]["content"]

    # ── contexts 수집 (챗봇 상태 기반, 래퍼보다 안정적) ──
    # 흐름 A: 챗봇이 실제 사용한 _cached_context (역질문 중 캐시 재사용 포함)
    # 흐름 B: caution_parser가 분석한 후보 약들의 문서를 가져옴 (retriever 미사용이므로)
    contexts = []
    drug_names = getattr(bot.state, "drug_names", []) or []
    if scenario["flow"] == "A":
        cached = getattr(bot.state, "_cached_context", "") or ""
        if cached.strip():
            contexts = [cached]
        elif _collected_contexts:
            contexts = _collected_contexts.copy()
    else:  # 흐름 B
        if drug_names:
            docs = fetch_drug_documents(drug_names)
            if docs.strip():
                contexts = [docs]

    if not contexts:
        contexts = _collected_contexts.copy()

    return {
        "scenario_id": scenario["scenario_id"],
        "flow": scenario["flow"],
        "category": scenario["category"],
        "user_input": scenario["user_input"],
        "ground_truth": scenario["ground_truth"],
        "expected_drugs": scenario["expected_drugs"],
        "forbidden_drugs": scenario["forbidden_drugs"],
        "turns": turns,
        "final_answer": final_answer or "",
        "contexts": contexts,
        "drug_names": drug_names if scenario["flow"] == "B" else [],
        "completed": final_answer is not None,
        "turn_count": len(turns) // 2,  # user-assistant 쌍 단위
    }


def main():
    print(f"\n{'='*60}")
    print(f"  시나리오 자동 실행 시작 — {len(SCENARIOS)}개")
    print(f"{'='*60}\n")

    results = []
    for i, scenario in enumerate(SCENARIOS, 1):
        print(f"\n{'─'*60}")
        print(f"[{i}/{len(SCENARIOS)}] {scenario['scenario_id']} ({scenario['flow']})")
        print(f"입력: {scenario['user_input']}")
        print(f"{'─'*60}")

        try:
            result = run_scenario(scenario)
            results.append(result)
            status = "✅" if result["completed"] else "⚠️"
            print(f"{status} 완료 (턴 수: {result['turn_count']}, 검색 횟수: {len(result['contexts'])})")
        except Exception as e:
            print(f"❌ 오류: {e}")
            results.append({
                "scenario_id": scenario["scenario_id"],
                "error": str(e),
                "completed": False,
            })

    # 결과 저장
    out_path = _HERE / "results" / "chatbot_outputs.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"  ✅ 모든 시나리오 실행 완료")
    print(f"  📁 결과 저장: {out_path}")
    print(f"{'='*60}")

    # 요약
    completed = sum(1 for r in results if r.get("completed"))
    print(f"\n  완료: {completed}/{len(SCENARIOS)}")

