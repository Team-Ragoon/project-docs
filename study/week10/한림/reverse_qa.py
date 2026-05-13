"""
역질문(Reverse Q) 핵심 로직
- rag_qna_multi.py 의 파이프라인은 건드리지 않음
- retriever_multi() 만 import해서 사용
"""

import json
import re
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# 기존 RAG 파이프라인에서 검색기만 가져옴 (파이프라인 수정 없음)
from rag_qna_multi import retriever_multi

from prompts.reverse_prompt import (
    CONTRAINDICATION_LABEL_MAP,
    NONE_OPTION,
    REVERSE_QUESTION_PROMPT,
    FOLLOW_UP_PROMPT,
    SUMMARIZE_PROMPT,
    FINAL_ANSWER_PROMPT,
)

# ─────────────────────────────────────────────
# LLM 설정 (기존과 동일 모델)
# ─────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()

reverse_question_chain = REVERSE_QUESTION_PROMPT | llm | parser
follow_up_chain = FOLLOW_UP_PROMPT | llm | parser
summarize_chain = SUMMARIZE_PROMPT | llm | parser
final_answer_chain = FINAL_ANSWER_PROMPT | llm | parser


# ─────────────────────────────────────────────
# 주의사항 텍스트에서 복용금지 섹션만 추출
# ─────────────────────────────────────────────
def extract_contraindication_text(context: str) -> str:
    """
    RAG 검색 결과(context)에서 '3.주의사항' 섹션만 추출.
    복용금지 판단에만 사용.
    """
    sections = re.findall(r"3\.주의사항\n(.+?)(?=\n\d+\.|$)", context, re.DOTALL)
    return "\n".join(sections).strip() if sections else context


# ─────────────────────────────────────────────
# LLM이 뽑은 키워드 → 사용자 친화적 선택지 변환
# ─────────────────────────────────────────────
def keywords_to_options(keywords: list[str]) -> list[str]:
    """
    ["임신", "음주", "간장애"] → 중복 없이 사용자 친화적 라벨로 변환
    """
    seen = set()
    options = []
    for kw in keywords:
        label = CONTRAINDICATION_LABEL_MAP.get(kw)
        if label and label not in seen:
            seen.add(label)
            options.append(label)
    return options


# ─────────────────────────────────────────────
# 콘솔 객관식 출력 및 입력 받기
# ─────────────────────────────────────────────
def ask_multiple_choice(options: list[str]) -> list[str]:
    """
    선택지를 번호로 출력하고, 사용자 입력을 받아 선택된 항목 반환.
    복수 선택 가능 (쉼표 구분), 0번은 해당 없음.
    """
    print("\n" + "─" * 50)
    print("💬 약 추천 전에 확인이 필요해요. 해당되는 항목을 선택해주세요.")
    print("   (복수 선택 가능, 쉼표로 구분. 예: 1,3)")
    print()

    all_options = options + [NONE_OPTION]
    for i, opt in enumerate(all_options, 1):
        print(f"  {i}. {opt}")

    print()

    while True:
        raw = input("선택 번호: ").strip()
        if not raw:
            continue
        try:
            indices = [int(x.strip()) for x in raw.split(",")]
            selected = []
            for idx in indices:
                if 1 <= idx <= len(all_options):
                    selected.append(all_options[idx - 1])
            if selected:
                return selected
            print("  ⚠️  올바른 번호를 입력해주세요.")
        except ValueError:
            print("  ⚠️  숫자만 입력해주세요.")


# ─────────────────────────────────────────────
# 대화 히스토리 기반 추가 역질문 (불친절한 사용자 대응)
# ─────────────────────────────────────────────
def follow_up_if_needed(
    conversation_history: list[dict],
    known_info: dict,
    missing_items: list[str],
    max_turns: int = 4,
) -> dict:
    """
    정보가 부족하면 LLM이 자연스럽게 추가 질문.
    사용자 응답을 받아 known_info 업데이트 후 반환.
    max_turns 초과 시 현재까지 파악된 정보로 진행.
    """
    turns = 0

    while missing_items and turns < max_turns:
        history_text = "\n".join(
            f"{'사용자' if m['role'] == 'user' else 'AI'}: {m['content']}"
            for m in conversation_history
        )
        known_text = "\n".join(f"- {k}: {v}" for k, v in known_info.items()) or "없음"
        missing_text = ", ".join(missing_items)

        response = follow_up_chain.invoke({
            "conversation_history": history_text,
            "known_info": known_text,
            "missing_items": missing_text,
        })

        response = response.strip()

        # LLM이 충분하다고 판단하면 종료
        if response.upper() == "OK":
            break

        # 추가 질문 출력
        print(f"\n💬 {response}")
        user_answer = input("👤 ").strip()

        if not user_answer:
            continue

        conversation_history.append({"role": "user", "content": user_answer})
        conversation_history.append({"role": "assistant", "content": response})

        # 사용자 답변에서 known_info 업데이트 (단순 키워드 매칭)
        lower_answer = user_answer.lower()
        remaining_missing = []
        for item in missing_items:
            matched = False
            for kw, label in CONTRAINDICATION_LABEL_MAP.items():
                if kw in lower_answer or kw in user_answer:
                    known_info[label] = "해당됨"
                    matched = True
                    break
            if not matched:
                # 부정 표현 확인
                if any(neg in user_answer for neg in ["아니", "없어", "아냐", "안 ", "노", "no"]):
                    known_info[item] = "해당 없음"
                else:
                    remaining_missing.append(item)

        missing_items = remaining_missing
        turns += 1

    return known_info


# ─────────────────────────────────────────────
# 메인 역질문 플로우
# ─────────────────────────────────────────────
def reverse_ask(question: str, verbose: bool = True) -> None:
    """
    역질문 포함 전체 플로우 실행.
    1. RAG 검색 (기존 retriever_multi 사용)
    2. 복용금지 항목 추출 → 객관식 역질문
    3. 추가 정보 필요 시 follow-up
    4. 통합 질문으로 최종 답변 생성
    """
    print(f"\n{'═' * 60}")
    print(f"🙋 질문: {question}")
    print(f"{'═' * 60}")

    # ── Step 1: RAG 검색 ──────────────────────────
    if verbose:
        print("\n🔍 관련 의약품 검색 중...")
    context = retriever_multi(question)

    # ── Step 2: 복용금지 항목 추출 ────────────────
    contraindication_text = extract_contraindication_text(context)

    raw_keywords_json = reverse_question_chain.invoke({
        "question": question,
        "contraindications": contraindication_text,
    })

    # JSON 파싱 (LLM 출력 안전하게 처리)
    try:
        clean = raw_keywords_json.strip().strip("```json").strip("```").strip()
        keywords = json.loads(clean)
    except Exception:
        keywords = []

    options = keywords_to_options(keywords)

    # ── fallback: 키워드 추출 실패 시 기본 역질문 항목 사용 ──
    if not options:
        options = [
            "임신 중이거나 임신 가능성 있음",
            "평소 음주를 즐기거나 최근 음주함",
            "만 15세 미만 어린이",
            "간 질환 보유",
        ]

    # ── Step 3: 역질문 ────────
    selected = []
    known_info = {}

    if options:
        selected = ask_multiple_choice(options)

        # 해당 없음 선택 시 known_info 비워둠
        if NONE_OPTION in selected:
            known_info = {}
        else:
            for item in selected:
                known_info[item] = "해당됨"

        # ── Step 4: 추가 역질문 (정보 부족 시) ────
        # 아직 확인 안 된 항목 계산
        missing_items = [opt for opt in options if opt not in known_info and opt != NONE_OPTION]

        if missing_items and NONE_OPTION not in selected:
            conversation_history = [
                {"role": "user", "content": question},
            ]
            known_info = follow_up_if_needed(
                conversation_history=conversation_history,
                known_info=known_info,
                missing_items=missing_items,
            )

    # ── Step 5: 대화 전체 요약 → 최종 질문 생성 ──────
    # 첫 질문 + 역질문 대화를 LLM이 한 문장으로 요약
    full_conversation = f"사용자: {question}"
    if known_info:
        conditions = ", ".join(k for k, v in known_info.items() if v == "해당됨")
        no_conditions = ", ".join(k for k, v in known_info.items() if v == "해당 없음")
        if conditions:
            full_conversation += f"\n확인된 해당 조건: {conditions}"
        if no_conditions:
            full_conversation += f"\n해당 없음으로 확인된 조건: {no_conditions}"

    integrated_question = summarize_chain.invoke({
        "conversation_history": full_conversation,
    }).strip()

    if verbose:
        print(f"\n📝 요약된 질문: {integrated_question}")

    # ── Step 6: 최종 답변 생성 ───────────────────
    print(f"\n{'─' * 60}")
    print("💊 약사 AI 답변:")
    print(f"{'─' * 60}")

    answer = final_answer_chain.invoke({
        "context": context,
        "integrated_question": integrated_question,
    })

    print(answer)
    print()