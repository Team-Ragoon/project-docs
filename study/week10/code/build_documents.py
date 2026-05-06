import json
import re
from langchain_core.documents import Document

# HTML 태그 제거 함수
def clean_html(text):
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# 성분명 추출 함수 (약품명 괄호 안에서 추출)
def extract_ingredient(item_name):
    match = re.search(r'\(([^)]+)\)', item_name)
    return match.group(1) if match else ""

# API 데이터 → 설명서 형식 변환
def format_drug(item):
    name = clean_html(item.get('itemName', ''))
    # API로 가져온 성분 우선 사용, 없으면 약품명 괄호에서 추출
    ingredient = item.get('ingredient_api', '') or extract_ingredient(name)
    efcy = clean_html(item.get('efcyQesitm', ''))
    use_method = clean_html(item.get('useMethodQesitm', ''))
    atpn_warn = clean_html(item.get('atpnWarnQesitm', ''))
    atpn = clean_html(item.get('atpnQesitm', ''))
    se = clean_html(item.get('seQesitm', ''))

    # 제형 정보 (낱알식별 API 승인 후 채워짐)
    pill_info = item.get('pill_info', {})
    form = pill_info.get('form_code', '')
    shape = pill_info.get('drug_shape', '')
    color = pill_info.get('color_class', '')

    form_text = ', '.join(filter(None, [form, shape, color])) or "정보 없음"

    # 주의사항 합치기
    atpn_combined = '\n'.join(filter(None, [atpn_warn, atpn]))

    # 동의어 보완: 검색 유사도 향상을 위해 문서 내 표현 통일
    efcy = efcy.replace("월경곤란증", "월경곤란증(생리통, 월경통)")

    content = f"""[{name} 설명서]

1.효능·효과
{efcy}

2.용법·용량
{use_method}

3.주의사항
{atpn_combined}

4.부작용
{se}

5.성분
{ingredient if ingredient else '정보 없음'}

6.제형(외형)
{form_text}
""".strip()

    return content

# JSON 불러오기
with open("C:/RAG/drug_data.json", "r", encoding="utf-8") as f:
    items = json.load(f)

print(f"총 {len(items)}개 데이터 로드\n")

# Document 변환
documents = []
for item in items:
    content = format_drug(item)
    name = clean_html(item.get('itemName', ''))
    doc = Document(
        page_content=content,
        metadata={
            "drug_name": name,
            "ingredient": item.get('ingredient_api', '') or extract_ingredient(name),
            "company": clean_html(item.get('entpName', '')),
            "item_seq": item.get('itemSeq', ''),
        }
    )
    documents.append(doc)

# 결과 파일로 저장
output = []
for doc in documents:
    output.append({
        "page_content": doc.page_content,
        "metadata": doc.metadata
    })

with open("C:/RAG/drug_documents.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

# 샘플 출력
print("=== 변환 결과 샘플 ===\n")
print(documents[0].page_content)
print("\n=== 메타데이터 ===")
print(documents[0].metadata)
print(f"\n총 {len(documents)}개 Document 생성 완료 → C:/RAG/drug_documents.json")
