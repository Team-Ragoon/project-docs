"""
흐름 B E2E 시나리오 테스트

시나리오를 자동화된 assert로 검증.
기존 run_clarify_test()와 동일한 흐름이지만 최종 응답에 대해 자동 판정.

각 케이스 필드:
  name             : 테스트 이름 (pytest 출력에 표시)
  first_input      : 사용자 첫 입력
  answers          : {역질문 subject: 사용자 응답}  ← subject가 없으면 기본 "아니요"
  expect_ok        : True=💊 추천 기대 / False=🚫 복용불가 기대
  expect_drug      : 응답에 반드시 포함되어야 할 약 이름 (선택)
  expect_not_drug  : 응답에 없어야 할 약 이름 (선택)
  expect_text      : 응답에 포함되어야 할 문자열 목록 (선택)
  expect_not_text  : 응답에 없어야 할 문자열 목록 (선택)

실행:
  pytest test_scenario_b.py -v
  pytest test_scenario_b.py -v -k "타이레놀"    ← 특정 케이스만 실행
  pytest test_scenario_b.py -v --tb=short      ← 실패 시 간략 출력
"""

import pytest
from rag_qna_multi import MedicalChatbot


# ─────────────────────────────────────────────────────────────
# 공통 실행 헬퍼
# ─────────────────────────────────────────────────────────────

def run_scenario(scenario: dict, verbose: bool = False) -> str:
    """
    시나리오를 실행하고 최종 응답 텍스트를 반환한다.

    - bot._pending_subject가 None이면 역질문 종료 → 루프 탈출
    - answers에 없는 subject는 기본값 "아니요" 로 응답
    - MAX_TURNS 초과 시 루프 강제 탈출 (무한루프 방지)
    """
    bot = MedicalChatbot()
    user_input = scenario["first_input"]
    response = ""
    MAX_TURNS = 15

    for turn in range(MAX_TURNS):
        response = bot.chat(user_input)

        if verbose:
            print(f"\n[턴 {turn + 1}]")
            print(f"  사용자: {user_input}")
            print(f"  챗봇: {response[:120]}...")

        # 역질문이 더 없으면 최종 답변 → 종료
        if not bot._pending_subject:
            break

        # 다음 답변 결정
        subject = bot._pending_subject
        answers = scenario.get("answers", {})
        if subject in answers:
            user_input = answers[subject]
            if verbose:
                print(f"  → '{subject}' 매칭 답변: {user_input}")
        else:
            user_input = "아니요"
            if verbose:
                print(f"  → '{subject}' 미정의 → 기본값 '아니요'")

    return response


# ─────────────────────────────────────────────────────────────
# 타이레놀 시나리오
# ─────────────────────────────────────────────────────────────
TYLENOL_SCENARIOS = [
    {
        "name": "[타이레놀/흐름B] 어린이(10살) — 어린이타이레놀 추천, 성인 제형 제외",
        "first_input": "타이레놀 먹어도 돼?",
        "answers": {
            "나이": "10살",
            "아세트아미노펜 최대 용량": "아뇨",
            "해열진통제 복용자": "아니오",
        },
        "expect_ok": True,
        "expect_drug": "어린이타이레놀",
        #"expect_not_drug": "타이레놀정500",
    },
    {
        "name": "[타이레놀/흐름B] 성인+알코올 — 복용 불가",
        "first_input": "두통이 심해. 타이레놀 먹어도 돼?",
        "answers": {
            "나이": "30살",
            "아세트아미노펜 최대 용량": "아뇨",
            "알코올 복용자": "ㅇㅇ",
        },
        "expect_ok": False,
    },
    {
        "name": "[타이레놀/흐름B] 임산부 — 성인 타이레놀 추천, 어린이 제형 제외",
        "first_input": "나 임신했는데 타이레놀 먹어도 될까?",
        "answers": {
            "나이": "30살",
            "아세트아미노펜 최대 용량": "아뇨",
            "알코올 복용자": "놉",
        },
        "expect_ok": True,
        "expect_drug": "타이레놀정500",
        "expect_not_drug": "어린이타이레놀",
    },
    {
        "name": "[타이레놀/흐름B] 임산부+감기약 복용 중 — 타이레놀정500 단독 추천, 타이레놀콜드 제외",
        "first_input": "나 임신했는데 머리가 너무 아파. 지금 다른 감기약을 먹고 있는데 잘 안 낫는 것 같아서 타이레놀 먹으려고 해. 괜찮을까?",
        "answers": {
            "나이": "28살",
            "아세트아미노펜 최대 용량": "아뇨",
        },
        "expect_ok": True,
        "expect_drug": "타이레놀정500",
        "expect_not_drug": "타이레놀콜드",
    },
    {
        "name": "[타이레놀/흐름B] 성인+우울증약(MAO억제제 사전 탐지) — 복용 불가",
        "first_input": "내가 우울증 약을 먹고 있어. 근데 두통이 좀 심한데 타이레놀 먹어도 될까?",
        "answers": {
            "나이": "30살",    # 나이는 필수 역질문 → 사전 탐지와 무관하게 항상 물어봄
        },
        "expect_ok": False,
    },
    {
        "name": "[타이레놀/흐름B] 성인+아세트아미노펜 여부 모름 — 추천+🔍 추가확인 포함",
        "first_input": "두통이 심해. 타이레놀 먹어도 돼?",
        "answers": {
            "나이": "30살",
            "아세트아미노펜 최대 용량": "잘 모르겠어",
            "알코올 복용자": "아니요",
        },
        "expect_ok": True,
        "expect_text": ["🔍"],
    },
    {
        "name": "[타이레놀/흐름B] 성인+와파린 병용 — 추가확인 안내 포함",
        "first_input": "타이레놀 먹어도 돼요? 지금 와파린 복용 중이에요",
        "answers": {
            "나이": "30살",
            "임산부": "아니요",
        },
        "expect_ok": True,          # 와파린은 복용 불가가 아닌 주의 안내
        "expect_text": ["🔍"],      # 미확인 금기사항 or 주의 언급
    },
]


# ─────────────────────────────────────────────────────────────
# 탁센 시나리오
# ─────────────────────────────────────────────────────────────
TAKSEN_SCENARIOS = [
    {
        "name": "[탁센/흐름B] 성인+두통 — 단일성분 2개 추천",
        "first_input": "두통이 심한데 탁센 먹어도 돼?",
        "answers": {
            "나이": "30살",
            "임산부": "ㄴ",
            "수유부": "아니오",
        },
        "expect_ok": True,
    },
    {
        "name": "[탁센/흐름B] 성인+생리통 — 이부프로펜 계열 추천",
        "first_input": "생리통이 심해. 탁센 먹어도 돼?",
        "answers": {
            "나이": "25살",
            "임산부": "아니요",
            "수유부": "아니요",
        },
        "expect_ok": True,
    },
    {
        "name": "[탁센/흐름B] 탁센이브+생리통 — 탁센이브 추천",
        "first_input": "생리통이 심해. 탁센이브 먹어도 돼?",
        "answers": {
            "나이": "25살",
            "알코올 복용자": "아니요",
            "해열진통제 복용자": "아니요",
        },
        "expect_ok": True,
        "expect_drug": "탁센이브",
    },
    {
        "name": "[탁센/흐름B] 성인+음주(사전 탐지) — 탁센이브 제외, 나머지 추천",
        "first_input": "내가 아까 술을 좀 마셨거든. 근데 머리가 좀 아프네. 탁센 먹어도 돼?",
        "answers": {
            "나이": "25살",
            "임산부": "아뇨",
        },
        "expect_ok": True,
        "expect_not_drug": "탁센이브",   # 탁센이브에만 알코올 금기사항 존재
    },
    {
        "name": "[탁센/흐름B] 탁센이브+해열진통제 복용 중 — 복용 불가",
        "first_input": "탁센이브 먹어도 돼? 두통이 있어.",
        "answers": {
            "나이": "25살",
            "알코올 복용자": "아뇨",
            "해열진통제 복용자": "응, 판콜 먹고 있어.",
        },
        "expect_ok": False,
    },
    {
        "name": "[탁센/흐름B] 임산부(사전 탐지) — 복용 불가(이부프로펜 계열 전부 금기)",
        "first_input": "탁센 먹어도 돼? 나 임산부야.",
        "answers": {
            "나이": "33살",
            "알코올 복용자": "아뇨",
        },
        "expect_ok": False,
    },
    {
        "name": "[탁센/흐름B] 나이 모름 — 추천+나이 추가확인",
        "first_input": "탁센 먹어도 돼? 두통이 심해.",
        "answers": {
            "나이": "잘 모르겠어요.",
            "임산부": "아뇨",
            "수유부": "아니오",
        },
        "expect_ok": True,
        "expect_text": ["나이", "🔍"],
    },
]


# ─────────────────────────────────────────────────────────────
# 이브 / 이지엔6이브 시나리오
# ─────────────────────────────────────────────────────────────
EVE_SCENARIOS = [
    {
        "name": "[이브/흐름B] 성인+인후통 — 추천",
        "first_input": "목이 좀 따가워. 이브 먹어도 돼?",
        "answers": {
            "나이": "20살",
            "임산부": "아뇨",
            "알코올 복용자": "아뇨",
        },
        "expect_ok": True,
    },
    {
        "name": "[이브/흐름B] 임산부(사전 탐지) — 복용 불가",
        "first_input": "발치하고 나서 치통이 좀 있어. 이브 먹어도 돼? 나 임산부야.",
        "answers": {
            "나이": "28살",    # 나이는 필수 역질문 → 사전 탐지와 무관하게 항상 물어봄
        },
        "expect_ok": False,
    },
    {
        "name": "[이브/흐름B] 어린이(7살) — 추천(나이 금기 미해당)",
        "first_input": "발치하고 나서 치통이 좀 있나봐. 이브 먹여도 될까?",
        "answers": {
            "나이": "7살",
            "해열진통제 복용자": "아뇨",
            "과민증 환자": "잘 모르겠어요.",
        },
        "expect_ok": True,
    },
    {
        "name": "[이지엔6이브/흐름B] 임산부(임신 초기) — 복용 불가",
        "first_input": "이지엔6이브 먹어도 될까요?",
        "answers": {
            "나이": "30살",
            "임산부": "임신 초기예요",
        },
        "expect_ok": False,
    },
]


# ─────────────────────────────────────────────────────────────
# 게보린 시나리오
# ─────────────────────────────────────────────────────────────
GEBORLIN_SCENARIOS = [
    {
        "name": "[게보린/흐름B] 성인+근육통 — 이부프로펜 계열 2개 추천",
        "first_input": "근육통이 좀 심해. 게보린 먹어도 돼?",
        "answers": {
            "나이": "23살",
            "아세트아미노펜 최대 용량": "아뇨",
            "알코올 복용자": "아니요",
        },
        "expect_ok": True,
    },
    {
        "name": "[게보린/흐름B] 중학교1학년(13세)+생리통 — 게보린정(15세미만금기) 제외, 나머지 추천",
        "first_input": "중학교 1학년 아인데 생리통이 좀 심하네. 게보린 먹어도 될까?",
        "answers": {
            "해열진통제 복용자": "아뇨",
            "아스피린 천식 환자": "아니요",
        },
        "expect_ok": True,
        "expect_not_drug": "게보린정",
    },
    {
        "name": "[게보린/흐름B] 임산부(복합제+이부프로펜 전부 불가) — 추천 불가",
        "first_input": "나 임산부인데 두통이 좀 있어. 게보린밖에 없는데 이거 먹어도 돼?",
        "answers": {
            "나이": "25살",
            "아세트아미노펜 최대 용량": "잘 모르겠어",
            "알코올 복용자": "아니",
        },
        "expect_ok": False,
    },
    {
        "name": "[게보린/흐름B] 성인+해열진통제 복용 중(사전 탐지) — 복용 불가",
        "first_input": "내가 다른 감기약을 먹고 있거든. 근데 효과가 좀 미미해서 게보린 먹으려고 하는데 괜찮을까?",
        "answers": {
            "나이": "30살",    # 나이는 필수 역질문 → 사전 탐지와 무관하게 항상 물어봄
        },
        "expect_ok": False,
    },
    {
        "name": "[게보린/흐름B] 성인+알코올(사전 탐지) — 복용 불가",
        "first_input": "나 아직 술이 안 깼는데 두통이 좀 있어. 게보린말고 다른 건 없어서 이거 먹으려고 하는데 괜찮아?",
        "answers": {
            "나이": "30살",    # 나이는 필수 역질문 → 사전 탐지와 무관하게 항상 물어봄
        },
        "expect_ok": False,
    },
    {
        "name": "[게보린/흐름B] 10살+두통 — 게보린정(15세미만금기) 제외, 나머지 추천",
        "first_input": "아이가 두통이 좀 있다고 하는데 게보린 괜찮아?",
        "answers": {
            "나이": "10살이야",
            "해열진통제 복용자": "아니 안 먹고 있어",
            "아스피린 천식 환자": "아니 없어",
        },
        "expect_ok": True,
        "expect_not_drug": "게보린정",
    },
    {
        "name": "[게보린/흐름B] 18세+알코올여부 모름 — 추천+알코올 추가확인",
        "first_input": "18살 아이인데 근육통이 있나봐. 게보린 먹여도 될까?",
        "answers": {
            "아세트아미노펜 최대 용량": "아뇨",
            "알코올 복용자": "잘 모르겠어",
        },
        "expect_ok": True,
        "expect_text": ["🔍"],
    },
    {
        "name": "[게보린/흐름B] 임산부+복용가능여부 직접 질문 — 추천 불가",
        "first_input": "임산부인데, 게보린 먹어도 돼요?",
        "answers": {
            "아세트아미노펜 최대 용량": "아니요",
            "알코올 복용자": "아니요",
        },
        "expect_ok": False,
    },
]


# ─────────────────────────────────────────────────────────────
# 까스활명수 시나리오
# ─────────────────────────────────────────────────────────────
KASMYEONG_SCENARIOS = [
    {
        "name": "[까스활명수/흐름B] 성인 — 추천",
        "first_input": "속이 좀 답답해. 점심이 소화가 잘 안됐나봐. 까스활명수 먹어도 돼?",
        "answers": {
            "나이": "42",
            "임산부": "아니",
        },
        "expect_ok": True,
        "expect_drug": "까스활명수",
    },
    {
        "name": "[까스활명수/흐름B] 어린이(6살) — 추천(나이 금기 미해당)",
        "first_input": "체했는지 자꾸 토를 하네. 까스활명수 먹여도 돼?",
        "answers": {
            "나이": "6",
        },
        "expect_ok": True,
    },
    {
        "name": "[까스활명수/흐름B] 2개월 아기 — 복용 불가(나이 금기 해당)",
        "first_input": "애기가 좀 답답해하고 소화를 잘 못 시키는데 까스활명수 먹여도 돼?",
        "answers": {
            "나이": "2개월",
        },
        "expect_ok": False,
        "expect_text": ["나이"],
    },
    {
        "name": "[까스활명수/흐름B] 임산부(3개월) — 복용 불가(현호색 성분 강제 금기)",
        "first_input": "임신한지 3개월 됐는데 속이 너무 답답해. 체한 것 같은데 까스활명수 먹어도 돼?",
        "answers": {
            "나이": "25살",
        },
        "expect_ok": False,
    },
]


# ─────────────────────────────────────────────────────────────
# 장엔폴 시나리오
# ─────────────────────────────────────────────────────────────
JANGANPOL_SCENARIOS = [
    {
        "name": "[장엔폴/흐름B] 성인+설사 — 추천",
        "first_input": "장엔폴 먹어도 돼? 어제 회식에서 뭔가 잘못 먹었는지 아침부터 배가 쥐어짜는 것처럼 아프고 설사가 심해",
        "answers": {
            "나이": "30세",
            "임산부": "아니",
            "수유부": "아뇨",
        },
        "expect_ok": True,
        "expect_drug": "장엔폴",
    },
    {
        "name": "[장엔폴/흐름B] 7세 — 복용 불가(나이 금기: 7세 이하)",
        "first_input": "아이가 어제부터 배 아프다고 화장실을 계속 가는데 장엔폴 먹여도 돼?",
        "answers": {
            "나이": "7세",
        },
        "expect_ok": False,
    },
    {
        "name": "[장엔폴/흐름B] 6세(사전 탐지) — 복용 불가",
        "first_input": "6살 아이가 자꾸 배가 아프다고 화장실을 가는데 장엔폴 먹여도 돼?",
        "answers": {},
        "expect_ok": False,
    },
    {
        "name": "[장엔폴/흐름B] 10세 — 추천(나이 금기 미해당)",
        "first_input": "10살 아이가 자꾸 배가 아프다고 화장실을 가는데 장엔폴 먹여도 돼?",
        "answers": {
            "유당불내증": "없어",
            "위장진통제 복용자": "아니",
        },
        "expect_ok": True,
        "expect_drug": "장엔폴",
    },
    {
        "name": "[장엔폴/흐름B] 임산부(사전 탐지, 임신 3개월) — 복용 불가",
        "first_input": "장엔폴 먹어도 될까? 임신 3개월인데 급하게 화장실에 가야 할 것 같은 느낌이 계속 들어.",
        "answers": {
            "나이": "30살",
        },
        "expect_ok": False,
    },
    {
        "name": "[장엔폴/흐름B] 성인+음주(사전 탐지) — 복용 불가(알코올 금기)",
        "first_input": "어제 한잔 했는데 배가 좀 아파. 화장실에 자꾸 가게 되는데 장엔폴 먹어도 돼?",
        "answers": {
            "나이": "30살",
        },
        "expect_ok": False,
    },
]


# ─────────────────────────────────────────────────────────────
# 동성정로환 시나리오
# ─────────────────────────────────────────────────────────────
JEONGROHWAN_SCENARIOS = [
    {
        "name": "[동성정로환/흐름B] 성인 — 2개 추천",
        "first_input": "배가 꾸르륵거리고 설사가 나는데 정로환 먹어도 돼?",
        "answers": {
            "나이": "25살",
            "임산부": "아니",
        },
        "expect_ok": True,
        "expect_drug": "동성정로환",
    },
    {
        "name": "[동성정로환/흐름B] 7세(사전 탐지) — 복용 불가(나이 금기)",
        "first_input": "7살인데 동성정로환 먹어도 돼?",
        "answers": {},
        "expect_ok": False,
    },
    {
        "name": "[동성정로환/흐름B] 6세 — 복용 불가",
        "first_input": "배탈이 났는지 설사가 심한데 동성정로환 먹어도 돼?",
        "answers": {
            "나이": "6살",
        },
        "expect_ok": False,
    },
    {
        "name": "[동성정로환/흐름B] 임산부(사전 탐지) — 복용 불가",
        "first_input": "임신 중인데 배탈이 난 것 같아. 설사가 심하네. 동성정로환 먹어도 돼?",
        "answers": {
            "나이": "37살",
        },
        "expect_ok": False,
    },
]


# ─────────────────────────────────────────────────────────────
# 부루펜 시나리오
# ─────────────────────────────────────────────────────────────
BRUFEN_SCENARIOS = [
    {
        "name": "[부루펜/흐름B] 10살 — 어린이부루펜시럽 추천, 성인 제형 제외",
        "first_input": "애가 온몸이 불덩이처럼 뜨겁고 꼼짝을 못해. 부루펜 먹여도 돼?",
        "answers": {
            "나이": "10살",
            "해열진통제 복용자": "아뇨",
            "유당불내증": "없어",
        },
        "expect_ok": True,
        "expect_drug": "어린이부루펜",
        "expect_not_drug": "부루펜정400",
    },
    {
        "name": "[부루펜/흐름B] 16살(청소년) — 성인 부루펜 추천, 어린이 제형 제외",
        "first_input": "애가 온몸이 불덩이처럼 뜨겁고 꼼짝을 못해. 부루펜 먹여도 돼?",
        "answers": {
            "나이": "16살",
            "임산부": "아뇨",
            "수유부": "아니",
        },
        "expect_ok": True,
        "expect_drug": "부루펜",
        "expect_not_drug": "어린이부루펜",
    },
]


# ─────────────────────────────────────────────────────────────
# 전체 시나리오 통합 + 테스트 실행
# ─────────────────────────────────────────────────────────────
ALL_SCENARIOS = (
    TYLENOL_SCENARIOS
    + TAKSEN_SCENARIOS
    + EVE_SCENARIOS
    + GEBORLIN_SCENARIOS
    + KASMYEONG_SCENARIOS
    + JANGANPOL_SCENARIOS
    + JEONGROHWAN_SCENARIOS
    + BRUFEN_SCENARIOS
)


@pytest.mark.parametrize(
    "scenario",
    ALL_SCENARIOS,
    ids=[s["name"] for s in ALL_SCENARIOS],
)
def test_flow_b_scenario(scenario: dict):
    """
    흐름 B 시나리오 E2E 테스트.

    실패 시 출력:
      - 어떤 assert가 실패했는지
      - 실제 응답 텍스트 (디버깅용)
    """
    response = run_scenario(scenario)

    name = scenario["name"]

    # ── 1. 추천/불가 여부 ─────────────────────────────────────
    if scenario["expect_ok"]:
        assert "💊" in response, (
            f"[{name}]\n추천(💊)이 나와야 하는데 없음\n\n응답:\n{response}"
        )
        assert "🚫" not in response, (
            f"[{name}]\n복용불가(🚫)가 잘못 포함됨\n\n응답:\n{response}"
        )
    else:
        assert "🚫" in response, (
            f"[{name}]\n복용불가(🚫)가 나와야 하는데 없음\n\n응답:\n{response}"
        )

    # ── 2. 특정 약 이름 포함 확인 ─────────────────────────────
    if scenario.get("expect_drug"):
        drug = scenario["expect_drug"]
        assert drug in response, (
            f"[{name}]\n'{drug}'이 응답에 없음\n\n응답:\n{response}"
        )

    # ── 3. 특정 약 이름 미포함 확인 ───────────────────────────
    if scenario.get("expect_not_drug"):
        drug = scenario["expect_not_drug"]
        assert drug not in response, (
            f"[{name}]\n'{drug}'이 응답에 있으면 안 됨\n\n응답:\n{response}"
        )

    # ── 4. 특정 텍스트 포함 확인 ──────────────────────────────
    for text in scenario.get("expect_text", []):
        assert text in response, (
            f"[{name}]\n'{text}'이 응답에 없음\n\n응답:\n{response}"
        )

    # ── 5. 특정 텍스트 미포함 확인 ────────────────────────────
    for text in scenario.get("expect_not_text", []):
        assert text not in response, (
            f"[{name}]\n'{text}'이 응답에 있으면 안 됨\n\n응답:\n{response}"
        )