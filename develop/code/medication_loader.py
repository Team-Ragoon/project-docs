import json
from pathlib import Path
from functools import lru_cache

DB_PATH = Path(__file__).parent.parent/"dataset"/"drug_data.json"

@lru_cache(maxsize=1)
def _load_db() -> list[dict]:
    # json 파일(drug_data.json) 파일을 최초 1회만 load 후 caching
    with open(DB_PATH, encoding = "utf-8") as f:
        return json.load(f)

def _find_drug(drug_name: str) -> dict | None:
    # 약 이름으로 항목 검색(부분 일치하는 것도 포함)
    db = _load_db()
    # 약 이름이 완전 일치하는 경우
    for item in db:
        if item["itemName"] == drug_name:
            return item
    # 약 이름이 부분 일치하는 경우
    for item in db:
        if drug_name in item["itemName"] or item["itemName"] in drug_name:
            return item
    return None


def get_drug_all_caution_texts(drug_name: str) -> dict[str, str]:
    # atpnQesitm , intrcQesitm 2개의 정보를 동시 반환
    # caution_parser.py에서 사용되어 역질문 list 생성.
    drug = _find_drug(drug_name)
    if not drug:
        return {"atpnQesitm": "", "intrcQesitm": "", }
    return {
        "atpnQesitm" : (drug.get("atpnQesitm") or "").strip(),
        "intrcQesitm" : (drug.get("intrcQesitm") or "").strip(),
    }

@lru_cache(maxsize = 1)
def list_drug_names() -> list[str]:
    # lru_cache 적용 -> 파일 중복 읽기 방지
    # drug_detector.py에서 사용되며 약 이름 matching에 사용됨.
    return [item["itemName"] for item in _load_db()]


# 후보 약 중 아세트아미노펜 성분이 포함된 약이 있는지 확인하기 위한 함수.
# 역질문 시 우선순위로 넣기 위해서 필요함.
def has_acetaminophen(drug_names: list[str]) -> bool:
    for drug_name in drug_names:
        drug = _find_drug(drug_name)
        if not drug:
            continue
        ingredient = (drug.get("ingredient_api") or "").lower()
        if "아세트아미노펜" in ingredient:
            return True
    return False

# 임산부에게 추천할 때 아세트아미노펜 단일 성분 약인지 확인. (prompt가 가끔 놓칠 때가 있어서 코드로 강제)
def is_acetaminophen_only(drug_name: str) -> bool:
    drug = _find_drug(drug_name)
    if not drug:
        return False
    ingredient = drug.get("ingredient_api") or ""
    ingredients = {i.strip() for i in ingredient.split(",") if i.strip()}
    return ingredients == {"아세트아미노펜"}

def get_drug_info(drug_name: str) -> dict | None:
    return _find_drug(drug_name)

# '설사'약인지 확인. (임산부에게 추천하지 않기 위해서.)
def is_diarrhea_medicine(drug_name: str) -> bool:
    drug = _find_drug(drug_name)
    if not drug:
        return False
    return "설사" in (drug.get("efcyQesitm") or "")