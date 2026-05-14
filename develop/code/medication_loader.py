import json
from pathlib import Path
from functools import lru_cache

DB_PATH = r"C:\Team-Ragoon\project-docs\study\week10\dataset\drug_data.json"

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
