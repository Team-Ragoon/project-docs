from langchain_core.chat_history import InMemoryChatMessageHistory

class DialogueState:
    def __init__(self):
        self.query_type: str | None = None
        self.symptom : str | None = None
        self.drug_name : str | None = None
        self.caution_slots : dict = {}
        self.clarify_count : int = 0
        self._history = InMemoryChatMessageHistory()
    
    def update_from_analysis(self, analysis: dict):
        # query_type은 매 turn 갱신
        self.query_type = analysis.get("query_type")

        # symptom, drug_name은 누적 유지
        if analysis.get("symptom") and not self.symptom:
            self.symptom = analysis["symptom"]
        if analysis.get("drug_name") and not self.drug_name:
            self.drug_name = analysis["drug_name"]
    
    def record_clarify_answer(self, subject: str, user_input: str):
        # 역질문 응답을 예/아니오로 변환해 저장
        positive = ["예", "네", "맞아", "있어", "있음", "그렇", "해당"]
        self.caution_slots[subject] = any(k in user_input for k in positive)
    
    def get_history(self) -> list:
        return self._history.messages

    def add_turn(self, user: str, assistant: str):
        self._history.add_user_message(user)
        self._history.add_ai_message(assistant)