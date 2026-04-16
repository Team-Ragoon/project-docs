import os
import requests
import json
from urllib.parse import quote
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ["PUBLIC_DATA_API_KEY"]
ENCODED_API_KEY = quote(API_KEY, safe='')

# e약은요 API
EASY_DRUG_URL = "https://apis.data.go.kr/1471000/DrbEasyDrugInfoService/getDrbEasyDrugList"

# 낱알식별 API
PILL_INFO_URL = "https://apis.data.go.kr/1471000/MdcinGrnIdntfcInfoService03/getMdcinGrnIdntfcInfoList03"

# 주성분 상세정보 API (승인 후 활성화됨)
INGREDIENT_URL = "https://apis.data.go.kr/1471000/DrugPrdtPrmsnInfoService07/getDrugPrdtMcpnDtlInq07"

drug_names = [
    # 두통 / 해열 / 진통
    "타이레놀", "이부프로펜", "게보린", "부루펜", "애드빌",

    # 소화 / 위장
    "훼스탈", "베아제", "노루모", "겔포스", "개비스콘",

    # 감기 / 기침
    "판콜", "화이투벤", "테라플루", "판피린", "콜대원",

    # 알러지 / 두드러기
    "지르텍", "클라리틴", "알레그라",

    # 피부 / 상처
    "후시딘", "마데카솔",

    # 눈
    "아이미루",
]

# 약품명에서 제형 추출 (낱알식별 DB에 없는 액상/연고류)
def extract_form_from_name(item_name):
    form_keywords = [
        ("건조시럽", "건조시럽"),   # '시럽'보다 먼저 체크해야 함
        ("점안액", "점안액"),
        ("점이액", "점이액"),
        ("현탁액", "현탁액"),
        ("내복액", "액제"),
        ("시럽", "시럽"),
        ("액", "액제"),
        ("연고", "연고"),
        ("크림", "크림"),
        ("겔", "겔"),
        ("산", "산제"),
        ("패치", "패치"),
        ("스프레이", "스프레이"),
    ]
    for keyword, form in form_keywords:
        if keyword in item_name:
            return form
    return "정보 없음"

# 주성분 API로 성분 정보 가져오기
def fetch_ingredient(item_name):
    try:
        response = requests.get(
            f"{INGREDIENT_URL}?serviceKey={ENCODED_API_KEY}",
            params={
                "Prduct": item_name,
                "numOfRows": 10,  # 성분이 여러 개일 수 있으므로 넉넉하게
                "pageNo": 1,
                "type": "json"
            }
        )
        data = response.json()
        items = data["body"]["items"]
        if items:
            # 성분명(MTRAL_NM)을 모두 수집해서 합치기
            ingredients = [item.get("MTRAL_NM", "") for item in items if item.get("MTRAL_NM")]
            return ", ".join(ingredients)
    except:
        pass
    return ""

# 낱알식별 API로 제형 정보 가져오기
def fetch_pill_info(item_name):
    params = {
        "serviceKey": ENCODED_API_KEY,
        "item_name": item_name,
        "numOfRows": 1,
        "pageNo": 1,
        "type": "json"
    }
    try:
        response = requests.get(PILL_INFO_URL, params=params)
        data = response.json()
        items = data["body"]["items"]
        if items:
            item = items[0]
            # 제형 관련 필드 추출
            return {
                "drug_shape": item.get("DRUG_SHAPE", ""),       # 모양 (원형, 타원형 등)
                "color_class": item.get("COLOR_CLASS1", ""),    # 색상
                "form_code": item.get("FORM_CODE_NAME", ""),    # 제형 (정제, 캡슐 등)
                "img_url": item.get("ITEM_IMAGE", ""),          # 이미지 URL
            }
    except:
        pass
    return {}

# API 요청 함수 (재시도 포함)
def safe_get(url, params, retries=3):
    for i in range(retries):
        try:
            response = requests.get(url, params=params, timeout=10)
            return response
        except Exception as e:
            if i < retries - 1:
                print(f"   ⚠️ 연결 오류, {i+1}번째 재시도... ({e})")
            else:
                raise

# e약은요 API로 기본 정보 가져오기
results = []

for name in drug_names:
    params = {
        "serviceKey": API_KEY,
        "itemName": name,
        "numOfRows": 3,
        "pageNo": 1,
        "type": "json"
    }

    try:
        response = safe_get(EASY_DRUG_URL, params)
        data = response.json()
        items = data["body"]["items"]

        if items:
            for item in items:
                # 주성분 API로 성분 정보 추가
                item["ingredient_api"] = fetch_ingredient(item.get("itemName", ""))

                # 낱알식별 API로 제형 정보 추가
                pill_info = fetch_pill_info(item.get("itemName", ""))

                # API에서 못 가져온 경우 약품명에서 제형 추출
                if not pill_info:
                    pill_info = {
                        "drug_shape": "",
                        "color_class": "",
                        "form_code": extract_form_from_name(item.get("itemName", "")),
                        "img_url": "",
                    }

                item["pill_info"] = pill_info

            results.extend(items)
            print(f"✅ {name}: {len(items)}개 가져옴")
        else:
            print(f"❌ {name}: 결과 없음")
    except Exception as e:
        print(f"❌ {name}: 오류로 건너뜀 - {e}")

# 중복 제거 (itemSeq 기준)
seen = set()
unique_results = []
for item in results:
    seq = item.get("itemSeq", "")
    if seq not in seen:
        seen.add(seq)
        unique_results.append(item)

print(f"\n중복 제거: {len(results)}개 → {len(unique_results)}개")

# JSON 파일로 저장
with open("C:/RAG/drug_data.json", "w", encoding="utf-8") as f:
    json.dump(unique_results, f, ensure_ascii=False, indent=2)

print(f"총 {len(unique_results)}개 저장 완료 → C:/RAG/drug_data.json")

print(f"\n총 {len(results)}개 저장 완료 → C:/RAG/drug_data.json")
