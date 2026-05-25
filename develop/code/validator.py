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

    # 사용자가 직접 입력한 나이가 10세 이하이거나 소아여부가 확인된 경우 임산부/수유부 역질문 스킵.
    # 임산부/수유부 확인 여부와 관계없이 나이 역질문은 항상 진행 (약마다 threshold가 다르므로).
    def _should_skip(self, subject: str, filled: dict, extra_context: dict = {}) -> bool:
        is_preg_subject = bool(PREG_PATTERN.search(subject) or subject == "임산부/수유부")
        user_age = extra_context.get("나이")  # 직접 입력받은 숫자 나이

        if is_preg_subject:
            if user_age is not None and user_age <= 10:
                return True
            if extra_context.get("소아여부") is True:
                return True

        return False


    # 약 이름을 사용해서 금기 대상 목록 찾기. (llm으로 추고화까지 끝냄)
    # 우선 순위 정렬된 금기 목록을 반환. (아세트아미노펜 성분이 있다면 3번째 위치에 최대 용량 슬롯 삽입하기.)
    def get_contraindications(self, drug_names: list[str]) -> list[dict]:
        items = list(parse_contraindications_for_drugs(drug_names, self.llm))

        if has_acetaminophen(drug_names):
            items = self._insert_acetaminophen_slot(items, drug_names)

        return items


    # 아세트아미노펜 성분이 있을 때 슬롯 삽입하는 동작 구현.
    def _insert_acetaminophen_slot(self, items: list[dict], drug_names: list[str]) -> list[dict]:
        # 우선순위 3위에 삽입.

        if any(item["subject"] == ACETAMINOPHEN_SLOT["subject"] for item in items):
            return items

        # 아세트아미노펜 성분이 있는 약만 applicable_drugs에 포함
        applicable = [d for d in drug_names if has_acetaminophen([d])]
        slot = {**ACETAMINOPHEN_SLOT, "applicable_drugs": applicable or list(drug_names)}

        insert_idx = 0
        for item in items:
            subject = item.get("subject", "")
            is_age  = subject == "나이"
            is_preg = subject == "임산부/수유부"
            if is_age or is_preg:
                insert_idx += 1
            else:
                break

        items.insert(insert_idx, slot)
        return items


    # 금기 목록 중 아직 질문 하지 않은 것만 filtering
    # can_drugs: 현재 복용 가능한 약 목록. 지정 시 applicable_drugs가 can_drugs와 겹치지 않는 슬롯은 스킵.
    def get_missing_slots(self, drug_names: list[str], filled: dict, extra_context: dict = {}, can_drugs: list[str] | None = None) -> tuple[list[dict],int]:
        result = []
        auto_filled = 0 # 자동 처리된 슬롯 수
        user_age = extra_context.get("나이")  # 숫자값
        can_drugs_set = set(can_drugs) if can_drugs is not None else None

        for c in self.get_contraindications(drug_names):
            subject = c["subject"]

            # 이미 답변된 슬롯 스킵
            if filled.get(subject) is not None:
                continue

            # can_drugs 필터: applicable_drugs 중 can_drugs에 남아있는 약이 없으면 스킵
            if can_drugs_set is not None:
                applicable = c.get("applicable_drugs", drug_names)
                if not any(d in can_drugs_set for d in applicable):
                    continue

            # 스킵 조건 확인
            if self._should_skip(subject, filled, extra_context):
                continue

            # 나이 슬롯: extra_context에 나이가 있으면 약별 threshold와 비교해서 자동 처리
            if subject == "나이" and user_age is not None:
                age_thresholds = c.get("age_thresholds", {})
                if age_thresholds:
                    # 약별 기준 나이 중 하나라도 미만이면 True(금기 약 존재)
                    filled["나이"] = any(user_age < t for t in age_thresholds.values())
                else:
                    # age_thresholds 없는 경우 question 텍스트에서 threshold 추출 (fallback)
                    threshold = self._get_age_threshold(c["question"])
                    filled["나이"] = (user_age < threshold) if threshold is not None else False
                auto_filled += 1
                continue  # 역질문 안 함

            #user_age는 없지만 소아 여부가 있는 경우 -> 역질문 포함
            if subject == "나이" and extra_context.get("소아여부") is True:
                result.append(c)
                continue

            result.append(c)
        return result, auto_filled


    # 역질문 횟수를 넘기기 않았는지 판단.
    def should_clarify(self, drug_name: str, filled: dict, count: int, extra_context: dict = {}, can_drugs: list[str] | None = None) -> bool:
        if count >= self.MAX_CLARIFY:
            return False
        slots, _ = self.get_missing_slots(drug_name, filled, extra_context, can_drugs)
        return len(slots) > 0


    # 다음 금기 목록 1개 반환.
    def get_priority_slot(self, drug_name: str, filled: dict, extra_context: dict = {}, can_drugs: list[str] | None = None) -> dict | None:
        slots, _ = self.get_missing_slots(drug_name, filled, extra_context, can_drugs)
        return slots[0] if slots else None