from langchain_core.messages import HumanMessage, AIMessage

class DialogueState:
    def __init__(self):
        self.query_type: str | None = None
        self.symptom : str | None = None
        self.drug_names : list[str] = [] # DB 후보 약 목록(검색용)
        self.drug_keyword : str | None = None # 사용자 입력 키워드 (출력용)
        self.caution_slots : dict = {}
        self.clarify_count : int = 0
        self._history : list = []

    def start_new_turn(self):
        self.query_type = None
        self.symptom = None
        self.caution_slots = {}
        self.clarify_count = 0
    
    def update_from_analysis(self, analysis: dict):
        # query_type은 매 turn 갱신
        self.query_type = analysis.get("query_type")
        #if analysis.get("symptom") and not self.symptom:
        self.symptom = analysis.get("symptom")

    def set_drug_candidates(self, drug_names: list[str], keyword: str):
        # 후보 약 목록 및 사용자 키워드 저장하기
        if not self.drug_names:
            self.drug_names = drug_names
            self.drug_keyword = keyword
    
    def apply_extracted_situation(self, situation: dict):
        # 사용자 입력에서 추출한 상황을 caution_slots에 사전 저장. (null은 저장 x)
        for subject, value in situation.items():
            # value in not None and
            if subject not in self.caution_slots:
                self.caution_slots[subject] = value
                print(f" [사전 추출]{subject}: {'해당' if value else '해당 없음'}")
    
    def record_clarify_answer(self, subject: str, is_positive: bool):
        # 역질문 응답에 대해 예/아니오 판단은 호출하는 chat에서 처리한 후 bool로 전달
        # 이 함수는 저장만 담당함.
        self.caution_slots[subject] = is_positive;
    
    def get_history(self) -> list:
        return self._history

    def add_turn(self, user: str, assistant: str):
        self._history.append(HumanMessage(content = user))
        self._history.append(AIMessage(content = assistant))