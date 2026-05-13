"""
역질문 RAG QnA - 실행 진입점
실행: python reverse_runner.py

시나리오:
  1. 소화제 - 친절한 사용자 (정보 충분)
  2. 진통제 - 불친절한 사용자 (정보 부족, 역질문 여러 번)
  3. 직접 입력
"""

from reverse_qa import reverse_ask


# ─────────────────────────────────────────────
# 시나리오 정의
# ─────────────────────────────────────────────
SCENARIOS = {
    "1": {
        "label": "소화제 시나리오 (친절한 사용자)",
        "question": (
            "오늘 치킨이랑 콜라 먹었는데 평소보다 좀 많이 먹었나봐요. "
            "소화가 안 되는 건지 배에 팽만감도 있고 살짝 아프기도 해요. "
            "원래 평소에도 소화가 잘 안 되는 편이라 소화기능이 약한 것 같아요. "
            "뭐 먹어야 될까요?"
        ),
    },
    "2": {
        "label": "진통제 시나리오 (불친절한 사용자)",
        "question": "나 머리아파",
    },
}


# ─────────────────────────────────────────────
# 실행
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  의약품 RAG QnA - 역질문(Reverse Q) 버전")
    print("=" * 60)
    print("\n실행 모드를 선택하세요:")
    print("  1. 소화제 시나리오 (친절한 사용자)")
    print("  2. 진통제 시나리오 (불친절한 사용자)")
    print("  3. 직접 질문 입력")

    mode = input("\n선택 (1 / 2 / 3): ").strip()

    if mode in ("1", "2"):
        scenario = SCENARIOS[mode]
        print(f"\n▶ {scenario['label']} 시작")
        reverse_ask(scenario["question"])

    elif mode == "3":
        print("\n질문을 입력하세요. 종료하려면 'q' 또는 'quit'를 입력하세요.\n")
        while True:
            question = input("🙋 질문: ").strip()
            if not question:
                continue
            if question.lower() in ("q", "quit"):
                print("\n종료합니다.")
                break
            reverse_ask(question)

    else:
        print("잘못된 입력입니다. 1, 2, 또는 3을 선택하세요.")


if __name__ == "__main__":
    main()
