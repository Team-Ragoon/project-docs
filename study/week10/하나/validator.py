from caution_parser import parse_contraindications
from langchain_openai import ChatOpenAI

class SlotValidator:
    MAX_CLARIFY = 3

    def __init__(self, llm : ChatOpenAI):
        self.llm = llm

    def get_contraindications(self, drug_name: str) -> list[dict]:
        # 약 이름 -> 금기 대상 목록 (llm으로 구조화까지 끝남)
        return list(parse_contraindications(drug_name, self.llm))

    def get_missing_slots(self, drug_name: str, filled: dict) -> list[dict]:
        # 전체 금기 중 아직 질문 안 한 것만 필터
        return [
            c for c in self.get_contraindications(drug_name)
            if filled.get(c["subject"]) is None
        ]

    def should_clarity(self, drug_name: str, filled: dict, count: int) -> bool:
        # 역질문을 해야 하는지 bool 판단
        if count >= self.MAX_CLARIFY:
            return False
        return len(self.get_missing_slots(drug_name, filled)) > 0
    
    def get_priority_slot(self, drug_name: str, filled: dict) -> dict | None:
        # 다음에 물어볼 금기 1개를 반환.
        missing = self.get_missing_slots(drug_name, filled)
        return missing[0] if missing else None