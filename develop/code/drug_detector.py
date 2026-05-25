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

STOPWORDS = {
    # 인물
    "아이", "어른", "성인", "어린이", "소아", "아기", "아저씨", "아주머니",
    "엄마", "아빠", "나", "우리", "사람", "여자", "남자", "내", "내"
    # 시간/장소
    "오늘", "어제", "내일", "방금", "지금", "여기", "거기",
    # 증상
    "감기", "두통", "복통", "비염", "알러지", "알레르기", "두드러기",
    # 지시어
    "그게", "그건", "이게", "이건", "근데", "약은", "약이",
    "좀", "안", "좋아", "원래",
}

# 단어 끝의 조사를 제거하는 함수
def _remove_josa(word: str) -> str:
    for josa in JOSA:
        if word.endswith(josa) and len(word) > len(josa):
            return word[: -len(josa)]
    return word


# 문장을 단어 단위로 분리하여 원본 token + 조사 제거 token 모두 찾는 함수
# STOPWORDS는 제외 (일반 한국어 단어가 약 이름의 substring으로 잘못 매칭되는 것 방지)
def _tokenize(text: str) -> list[str]:
    # 문장을 단어 단위로 분리
    raw_tokens = [w for w in re.split(r"[\s?.,!]", text) if len(w) >= MIN_TOKEN_LEN]

    # 원본 token + 조사 제거 token 모두 포함
    tokens = set()
    for token in raw_tokens:
        tokens.add(token)
        tokens.add(_remove_josa(token))
    return [t for t in tokens if len(t) >= MIN_TOKEN_LEN and t not in STOPWORDS]


# text에서 DB에 적재되어 있는 약 이름 탐지하여 약 이름 전체를 찾는 함수.
# DB에 적재되어 있는 약의 이름들 + 사용자가 입력한 keyword 반환.
# - substring 매칭은 prefix(시작 부분)일 때만 허용 (오탐 방지)
#   예: "아이" → "판콜아이콜드시럽" 같은 중간 매칭 제거
# - Fuzzy 매칭은 길이 3 이상 토큰만 (짧은 토큰은 오탐 위험 높음)
def detect_drugs_in_text(text: str) -> tuple[list[str], str | None]:
    drug_names = list_drug_names()
    tokens = _tokenize(text)
    #print(f"[토큰 목록] {tokens}")

    matched = []
    user_keyword = None

    for name in drug_names:
        for token in tokens:
            cleaned = _remove_josa(token)
            # cleaned가 STOPWORDS에 해당하면 조사 제거 버전으로 매칭하지 않음
            safe_cleaned = cleaned if cleaned not in STOPWORDS else token

            # 1차: 토큰이 약 이름에 포함
            if safe_cleaned in name or token in name:
                #print(f"[포함 매칭] token='{token}' → '{name}'")
                if name not in matched:
                    matched.append(name)
                if not user_keyword:
                    user_keyword = safe_cleaned
                break

            # 2차: Fuzzy 매칭 (길이 3 이상 토큰만)
            if len(token) >= 3 and fuzz.ratio(token, name) >= FUZZY_THRESHOLD:
                #print(f"[퍼지 매칭] token='{token}' → '{name}', score={fuzz.ratio(token, name)}")
                if name not in matched:
                    matched.append(name)
                if not user_keyword:
                    user_keyword = safe_cleaned
                break

    return matched, user_keyword