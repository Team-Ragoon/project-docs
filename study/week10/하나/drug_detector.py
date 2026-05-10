from medication_loader import list_drug_names
from rapidfuzz import process, fuzz
import re

FUZZY_THRESHOLD = 80
MIN_TOKEN_LEN = 2 # 너무 짧은 토큰은 오탐 방지

# 한국어 조사 목록
JOSA = [
    "으로부터", "에서부터", "로부터",  # 긴 것 먼저 (순서 중요)
    "이라고", "라고", "이라면", "라면",
    "에게서", "한테서",
    "으로서", "로서", "으로써", "로써",
    "이지만", "지만", "이지만", "지만",
    "에서", "에게", "한테", "에서", "부터", "까지",
    "으로", "로",
    "이랑", "랑", "이나", "나",
    "이며", "이고", "이지", "이든",
    "을", "를", "이", "가", "은", "는",
    "도", "만", "의", "와", "과",
]

def _remove_josa(word: str) -> str:
    # 단어 끝의 조사 제거
    for josa in JOSA:
        if word.endswith(josa) and len(word) > len(josa):
            return word[: -len(josa)]
    return word


def _tokenize(text: str) -> list[str]:
    # 문장을 단어 단위로 분리
    raw_tokens = [w for w in re.split(r"[\s?.,!]", text) if len(w) >= MIN_TOKEN_LEN]

    # 원본 token + 조사 제거 token 모두 포함
    tokens = set()
    for token in raw_tokens:
        tokens.add(token)
        tokens.add(_remove_josa(token))
    return [t for t in tokens if len(t) >= MIN_TOKEN_LEN]


def detect_drug_in_text(text: str) -> str | None:
    # 텍스트에서 DB 약 이름 탐지.
    drug_names = list_drug_names()
    tokens = _tokenize(text)

    for name in drug_names:
        for token in tokens:
            if token in name:
                return name

            if fuzz.ratio(token, name) >= FUZZY_THRESHOLD:
                return name
    return None