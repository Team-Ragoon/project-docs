import re
from caution_parser import parse_contraindications_for_drugs, ACETAMINOPHEN_SLOT
from medication_loader import has_acetaminophen
from langchain_openai import ChatOpenAI

AGE_PATTERN = re.compile(r"(만\s*)?\d+\s*(개월|세)\s*(미만|이하)|소아|영아|유아|젖먹이|신생아|영유아|어린이")

CHILD_AGE_PATTERN = re.compile(
    r"(만\s*)?\b([1-9]|10)\b\s*세\s*(미만|이하)"
    r"|\d+\s*개월\s*(미만|이하)"
    r"|소아|영아|유아|젖먹이|신생아|영유아|어린이"
)

PREG_PATTERN = re.compile(r"임부|임신 가능성|임산부|수유부|수유 중")


class SlotValidator:
    MAX_CLARIFY = 3 # 역질문 횟수

    def __init__(self, llm : ChatOpenAI):
        self.llm = llm


    # 이미 채워진 슬롯 기반으로 특정 슬롯을 건너뛸지 판단.
    # 임산부/수유부에 해당 -> 나이(01세 미만) 관련 역질문 스킵
    # 나이(10세 미만) 해당 -> 임산부/수유부 역질문 스킵

    def _should_skip(self, subject: str, filled: dict, extra_context: dict = {}) -> bool:
        is_preg_subject = bool(PREG_PATTERN.search(subject) or subject == "임산부/수유부")
        is_age_subject  = bool(CHILD_AGE_PATTERN.search(subject) or subject == "나이")

        # caution_slots 또는 extra_context에서 임산부/수유부 확인
        preg_positive = (
            filled.get("임산부/수유부") is True
            or any(PREG_PATTERN.search(k) and v is True for k, v in extra_context.items())
        )

        # caution_slots 또는 extra_context에서 나이 확인
        age_positive = (
            filled.get("나이") is True
            or any(
                (k == "나이" and isinstance(v, bool) and v is True)  # bool로 변환된 경우
                or (CHILD_AGE_PATTERN.search(k) and v is True)       # 키워드인 경우
                for k, v in extra_context.items()
    )
        )

        if is_age_subject and preg_positive:
            return True
        if is_preg_subject and age_positive:
            return True


        return False


    # 약 이름을 사용해서 금기 대상 목록 찾기. (llm으로 추고화까지 끝냄)
    # 우선 순위 정렬된 금기 목록을 반환. (아세트아미노펜 성분이 있다면 3번째 위치에 최대 용량 슬롯 삽입하기.)
    def get_contraindications(self, drug_names: str) -> list[dict]:
        items = list(parse_contraindications_for_drugs(drug_names, self.llm))
        
        if has_acetaminophen(drug_names):
            items = self._insert_acetaminophen_slot(items)
        
        return items
    
    
    # 아세트아미노펜 성분이 있을 때 슬롯 삽입하는 동작 구현.
    def _insert_acetaminophen_slot(self, items: list[dict]) -> list[dict]:
        # 우선순위 3위에 삽입. 

        if any(item["subject"] == ACETAMINOPHEN_SLOT["subject"] for item in items):
            return items

        insert_idx = 0
        for item in items:
            text = f"{item.get('subject', '')} {item.get('reason', '')}"
            is_age  = bool(re.search(r"(만\s*)?\d+\s*(개월|세)\s*(미만|이하)", text))
            is_preg = bool(re.search(r"임부|임신 가능성|임산부|수유부|수유 중|알코올|음주|술", text))
            if is_age or is_preg:
                insert_idx += 1
            else:
                break

        items.insert(insert_idx, ACETAMINOPHEN_SLOT)
        return items       


    # 금기 목록 중 아직 질문 하지 않은 것만 filtering
    def get_missing_slots(self, drug_names: list[str], filled: dict, extra_context: dict = {}) -> list[dict]:
        return [
            c for c in self.get_contraindications(drug_names)
            if filled.get(c["subject"]) is None
            and not self._should_skip(c["subject"], filled, extra_context)
        ]


    # 역질문 횟수를 넘기기 않았는지 판단.
    def should_clarify(self, drug_name: str, filled: dict, count: int, extra_context: dict = {}) -> bool:
        if count >= self.MAX_CLARIFY:
            return False
        return len(self.get_missing_slots(drug_name, filled, extra_context)) > 0
    

    # 다음 금기 목록 1개 반환.
    def get_priority_slot(self, drug_name: str, filled: dict, extra_context: dict = {}) -> dict | None:
        missing = self.get_missing_slots(drug_name, filled, extra_context)
        return missing[0] if missing else None