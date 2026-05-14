from caution_parser import parse_contraindications_for_drugs
from langchain_openai import ChatOpenAI


class SlotValidator:
    MAX_CLARIFY = 3 # 역질문 횟수

    def __init__(self, llm : ChatOpenAI):
        self.llm = llm

    # 약 이름을 사용해서 금기 대상 목록 찾기. (llm으로 추고화까지 끝냄)
    def get_contraindications(self, drug_name: str) -> list[dict]:
        return list(parse_contraindications_for_drugs(drug_name, self.llm))


    # 금기 목록 중 아직 질문 하지 않은 것만 filtering
    def get_missing_slots(self, drug_name: str, filled: dict) -> list[dict]:
        return [
            c for c in self.get_contraindications(drug_name)
            if filled.get(c["subject"]) is None
        ]


    # 역질문 횟수를 넘기기 않았는지 판단.
    def should_clarify(self, drug_name: str, filled: dict, count: int) -> bool:
        if count >= self.MAX_CLARIFY:
            return False
        return len(self.get_missing_slots(drug_name, filled)) > 0
    

    # 다음 금기 목록 1개 반환.
    def get_priority_slot(self, drug_name: str, filled: dict) -> dict | None:
        missing = self.get_missing_slots(drug_name, filled)
        return missing[0] if missing else None