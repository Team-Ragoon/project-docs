"""
흐름 B 컴포넌트 단위 테스트

각 함수를 독립적으로 검증합니다.
LLM 호출 여부에 따라 두 구간으로 구분됩니다.

  [섹션 1] parse_yes_no — 규칙 기반 케이스  (LLM 호출 없음, 빠름)
  [섹션 2] parse_yes_no — LLM 판단 필요 케이스
  [섹션 3] parse_age    — LLM 필요
  [섹션 4] _build_user_profile — caution_slots → 텍스트 변환

실행:
  pytest test_component_b.py -v
  pytest test_component_b.py -v -m "not slow"   ← LLM 없이 빠르게만 실행
"""

import pytest
from unittest.mock import MagicMock

from answer_parser import parse_yes_no, parse_age

# ─────────────────────────────────────────────────────────────
# LLM 초기화 (섹션 2, 3, 4에서 사용)
# rag_qna_multi 임포트 시 ChromaDB 초기화가 함께 실행되므로
# LLM이 필요 없는 섹션 1은 이 임포트 없이도 동작함
# ─────────────────────────────────────────────────────────────
try:
    from rag_qna_multi import llm as _real_llm, MedicalChatbot
    _llm_available = True
except Exception as e:
    _real_llm = None
    MedicalChatbot = None
    _llm_available = False
    print(f"[경고] LLM 초기화 실패 → 섹션 2·3·4 스킵됨: {e}")

_skip_if_no_llm = pytest.mark.skipif(
    not _llm_available, reason="LLM/ChromaDB 초기화 실패 — OPENAI_API_KEY 또는 chroma_db 경로 확인 필요"
)


# ═════════════════════════════════════════════════════════════
# 섹션 1 | parse_yes_no — 규칙 기반 (LLM 호출 없음)
# ═════════════════════════════════════════════════════════════

# 규칙 기반 케이스에서 LLM이 호출되면 테스트 구조 문제이므로 MagicMock으로 감지
_DUMMY_LLM = MagicMock()

@pytest.mark.parametrize("user_input, expected", [
    # ── 명확한 긍정 ──────────────────────────────────────────
    ("네",           True),
    ("네요",         True),
    ("응",           True),
    ("맞아요",       True),
    ("있어요",       True),
    ("있어",         True),
    ("ㅇㅇ",         True),
    ("해당해요",     True),
    # ── 명확한 부정 ──────────────────────────────────────────
    ("아니요",       False),
    ("아뇨",         False),
    ("아니",         False),
    ("없어요",       False),
    ("없어",         False),
    ("ㄴ",           False),
    ("ㄴㄴ",         False),
    ("아님",         False),
    ("놉",           False),   # Notion 시나리오에 등장하는 표현
    # ── 부정 우선 처리 (NEGATION_OVERRIDE) ───────────────────
    ("아니 안 먹고 있어",  False),   # "있어"(긍정) 포함하지만 앞에 "안"
    ("안해요",            False),
    ("하지 않아요",       False),
    ("안 하고 있어요",    False),
    # ── 모름 / 불확실 ────────────────────────────────────────
    ("잘 모르겠어요",     None),
    ("잘 모르겠어",       None),
    ("모르겠어요",        None),
    ("몰라요",            None),
])
def test_parse_yes_no_rule_based(user_input: str, expected):
    """
    규칙 기반으로 처리되는 케이스는 LLM 없이 즉시 결과를 반환해야 한다.
    MagicMock LLM이 실제로 invoke되면 AttributeError가 발생해 테스트 실패로 감지된다.
    """
    result = parse_yes_no(
        user_input=user_input,
        question="임산부이신가요?",
        llm=_DUMMY_LLM,
        subject="임산부",
    )
    assert result == expected, (
        f"입력: '{user_input}' → 예상: {expected}, 실제: {result}"
    )


# ═════════════════════════════════════════════════════════════
# 섹션 2 | parse_yes_no — LLM 판단 필요 케이스
# ═════════════════════════════════════════════════════════════

@_skip_if_no_llm
@pytest.mark.slow
@pytest.mark.parametrize("user_input, question, expected", [
    # 표면은 긍정어("네")지만 문맥상 부정
    (
        "네, 저는 임산부가 아니에요",
        "임산부이시거나 임신 가능성이 있으신가요?",
        False,
    ),
    # 불확실한 긍정
    (
        "임신 가능성이 있긴 해요",
        "임산부이시거나 임신 가능성이 있으신가요?",
        True,
    ),
    # 긍정+부정 혼재 → LLM이 문맥 판단
    (
        "있긴 한데 잘 모르겠어요",
        "현재 다른 약을 복용 중이신가요?",
        None,
    ),
])
def test_parse_yes_no_llm_fallback(user_input: str, question: str, expected):
    """
    규칙 기반으로 판단 불가한 표현은 LLM이 문맥을 보고 결정해야 한다.
    """
    result = parse_yes_no(
        user_input=user_input,
        question=question,
        llm=_real_llm,
        subject="임산부",
    )
    assert result == expected, (
        f"입력: '{user_input}' → 예상: {expected}, 실제: {result}"
    )


# ═════════════════════════════════════════════════════════════
# 섹션 3 | parse_age — LLM 필요
# ═════════════════════════════════════════════════════════════

@_skip_if_no_llm
@pytest.mark.slow
@pytest.mark.parametrize("user_input, expected_min, expected_max", [
    # (입력, 예상 최솟값, 예상 최댓값)  — 범위로 확인하는 이유: LLM 반올림 허용
    ("30살",                29.9,  30.1),
    ("10살이야",              9.9,  10.1),
    ("7세",                   6.9,   7.1),
    ("2개월",                 0.1,   0.2),    # ≈ 2/12 ≈ 0.167
    ("18개월",                1.4,   1.6),    # ≈ 18/12 = 1.5
    ("초등학교 1학년",         6.9,   7.1),
    ("중학교 1학년",          12.9,  13.1),
    ("고등학교 3학년",        17.9,  18.1),
])
def test_parse_age_numeric(user_input: str, expected_min: float, expected_max: float):
    """숫자·학년·개월 표현이 올바른 연령(세)으로 변환되어야 한다."""
    result = parse_age(user_input, _real_llm)
    assert result is not None, (
        f"입력: '{user_input}' → 숫자 반환 기대, 실제: None"
    )
    assert expected_min <= result <= expected_max, (
        f"입력: '{user_input}' → [{expected_min}, {expected_max}] 기대, 실제: {result}"
    )


@_skip_if_no_llm
@pytest.mark.slow
@pytest.mark.parametrize("user_input", [
    "잘 모르겠어요.",
    "모르겠어요",
])
def test_parse_age_unknown(user_input: str):
    """나이를 전혀 알 수 없는 경우 None을 반환해야 한다."""
    result = parse_age(user_input, _real_llm)
    assert result is None, (
        f"입력: '{user_input}' → None 기대, 실제: {result}"
    )


# ═════════════════════════════════════════════════════════════
# 섹션 4 | _build_user_profile — caution_slots → 자연어 변환
# ═════════════════════════════════════════════════════════════

@_skip_if_no_llm
@pytest.mark.parametrize("caution_slots, must_contain, must_not_contain", [
    # True → "금기 해당" 표기
    (
        {"나이": True},
        ["나이", "금기 해당"],
        ["해당 없음", "확인 불가"],
    ),
    # False → "해당 없음" 표기
    (
        {"임산부": False},
        ["임산부", "해당 없음"],
        ["금기 해당", "확인 불가"],
    ),
    # None → "확인 불가" 표기
    (
        {"알코올 복용자": None},
        ["알코올 복용자", "확인 불가"],
        ["금기 해당", "해당 없음"],
    ),
    # 복합: True + False 동시 포함
    (
        {"나이": True, "임산부": False},
        ["나이", "금기 해당", "임산부", "해당 없음"],
        ["확인 불가"],
    ),
    # 빈 slots → 특이사항 없음
    (
        {},
        ["특이사항 없음"],
        ["금기 해당", "해당 없음"],
    ),
])
def test_build_user_profile(caution_slots: dict, must_contain: list, must_not_contain: list):
    """
    caution_slots 딕셔너리가 올바른 자연어 텍스트로 변환되어야 한다.
    True  → "금기 해당"
    False → "해당 없음"
    None  → "확인 불가"
    """
    bot = MedicalChatbot()
    bot.state.caution_slots = caution_slots
    profile = bot._build_user_profile()

    for text in must_contain:
        assert text in profile, (
            f"'{text}'이 프로파일에 없음\ncaution_slots: {caution_slots}\n프로파일: {profile}"
        )
    for text in must_not_contain:
        assert text not in profile, (
            f"'{text}'이 프로파일에 있으면 안 됨\ncaution_slots: {caution_slots}\n프로파일: {profile}"
        )