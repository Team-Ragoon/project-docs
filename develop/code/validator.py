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

    # subject에서 기준 나이 추출. (6세 미만의 유아...)
    # 개월 -> 세로 변환.
    def _get_age_threshold(self, text: str) -> int | None:
        month_match = re.search(r"(\d+)\s*개월", text)
        if month_match:
            return int(month_match.group(1)) / 12  # 개월 → 세

        year_match = re.search(r"(\d+)\s*세", text)
        if year_match:
            return int(year_match.group(1))

        return None

    # 이미 채워진 슬롯 기반으로 특정 슬롯을 건너뛸지 판단.
    # 임산부/수유부에 해당 -> 나이(01세 미만) 관련 역질문 스킵
    # 나이(10세 미만) 해당 -> 임산부/수유부 역질문 스킵
    def _should_skip(self, subject: str, filled: dict, extra_context: dict = {}) -> bool:
        is_preg_subject = bool(PREG_PATTERN.search(subject) or subject == "임산부/수유부")
        is_age_subject  = bool(CHILD_AGE_PATTERN.search(subject) or subject == "나이")

        user_age = extra_context.get("나이")

        # caution_slots 또는 extra_context에서 임산부/수유부 확인
        preg_positive = (
            filled.get("임산부/수유부") is True
            or any(PREG_PATTERN.search(k) and v is True for k, v in extra_context.items())
        )


     # 나이 슬롯이 True로 채워졌거나, extra_context 나이가 기준 미만인 경우
        age_positive = (filled.get("나이") is True
                        or (user_age is not None and user_age <= 10)
                        or extra_context.get("소아여부") is True)
        
        if not age_positive and user_age is not None and is_preg_subject:
            # 임산부/수유부 스킵 여부: 나이가 10세 이하면 스킵
            age_positive = user_age <= 10       


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
            subject = item.get("subject", "")
            # 정규화된 subject 기준으로 판별
            is_age  = subject == "나이"
            is_preg = subject == "임산부/수유부"
            if is_age or is_preg:
                insert_idx += 1
            else:
                break

        items.insert(insert_idx, ACETAMINOPHEN_SLOT)
        return items       


    # 금기 목록 중 아직 질문 하지 않은 것만 filtering
    def get_missing_slots(self, drug_names: list[str], filled: dict, extra_context: dict = {}) -> tuple[list[dict],int]:
        result = []
        auto_filled = 0 # 자동 처리된 슬롯 수
        user_age = extra_context.get("나이")  # 숫자값

        for c in self.get_contraindications(drug_names):
            subject = c["subject"]

            # 이미 답변된 슬롯 스킵
            if filled.get(subject) is not None:
                continue

            # 스킵 조건 확인
            if self._should_skip(subject, filled, extra_context):
                continue

            # 나이 슬롯: extra_context에 나이가 있으면 기준과 비교해서 자동 처리
            if subject == "나이" and user_age is not None:
                threshold = self._get_age_threshold(c["question"])  # question에서 기준 나이 추출
                if threshold is not None:
                    # 기준 나이 미만이면 True(금기), 이상이면 False(비금기)로 자동 저장
                    filled["나이"] = user_age < threshold
                    auto_filled += 1
                continue  # 역질문 안 함

            #user_age는 없지만 소아 여부가 있는 경우 -> 역질문 포함
            if subject == "나이" and extra_context.get("소아여부") is True:
                result.append(c)
                continue

            result.append(c)
        return result, auto_filled


    # 역질문 횟수를 넘기기 않았는지 판단.
    def should_clarify(self, drug_name: str, filled: dict, count: int, extra_context: dict = {}) -> bool:
        if count >= self.MAX_CLARIFY:
            return False
        slots, _ = self.get_missing_slots(drug_name, filled, extra_context)
        return len(slots) > 0
    

    # 다음 금기 목록 1개 반환.
    def get_priority_slot(self, drug_name: str, filled: dict, extra_context: dict = {}) -> dict | None:
        slots, _ = self.get_missing_slots(drug_name, filled, extra_context)
        return slots[0] if slots else None