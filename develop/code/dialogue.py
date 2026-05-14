from langchain_core.messages import HumanMessage, AIMessage

class DialogueState:
    def __init__(self):
        self.query_type: str | None = None # 현재 query가 증상만 있는지(symptom_only) / 약 이름이 포함되는지(medication)
        self.symptom : str | None = None # query에서 추출한 증상
        self.drug_names : list[str] = [] # DB 후보 약 목록(검색용 - 예를 들어, 콜대원콜드에스시럽, 콜대원키즈이부펜시럽..)
        self.drug_keyword : str | None = None # 사용자 입력 키워드(출력용 - 예를 들어, 콜대원)
        self.caution_slots : dict = {} # 금기 사항에 대한 사용자의 정보
        self.extra_context: dict = {}  # query에서 금기 사항에 해당하지는 않지만 사용자의 증상에 해당하는 text
        self.clarify_count : int = 0 # 역질문 횟수
        self._history : list = []

    
    # 현재 코드 -> 새로운 query를 새로운 사용자로 인식. (모두 초기화해야 함.)
    def start_new_turn(self):
        self.query_type = None
        self.symptom = None
        self.drug_names = []
        self.drug_keyword = None
        self.caution_slots = {}
        self.extra_context = {}
        self.clarify_count = 0
    
    
    # query에 대한 분석 후에 갱신을 위한 함수.
    # query_type, symptom은 매 turn마다(=query) 갱신.
    def update_from_analysis(self, analysis: dict):
        self.query_type = analysis.get("query_type")
        self.symptom = analysis.get("symptom")


    # 후보 약 목록 및 사용자 키워드 저장하는 함수.
    def set_drug_candidates(self, drug_names: list[str], keyword: str):
        if not self.drug_names:
            self.drug_names = drug_names
            self.drug_keyword = keyword
    

    # 사용자 입력에서 추출한 상황을 금기 여부에 따라 분류하는 함수.
    # 만약, 금기 목록에 존재 -> caution_slots에 저장.(나중에 역질문 skip)
    # 만약, 금기 목록에 존재 X -> extra_context에 저장하여 약 추천 시 고려.
    def apply_extracted_situation(self, situation: dict, caution_subjects: list[str]):
        for subject, value in situation.items():
            if subject in caution_subjects: 
                if subject not in self.caution_slots:
                    self.caution_slots[subject] = value
                    status = "해당" if value else "해당 없음"
            else:
                if subject not in self.extra_context:
                    self.extra_context[subject] = value
                    status = "해당" if value else "해당 없음"


    # 역질문에 대한 응답을 저장. 
    # (해당 함수 호출 전에 긍정/부정 응답 판단을 위한 함수인 parse_yes_no 호출한 상태 -> parse_yes_no가 반환한 bool 타입 값을 저장.)
    def record_clarify_answer(self, subject: str, is_positive: bool):
        self.caution_slots[subject] = is_positive;
    

    def get_history(self) -> list:
        return self._history

    def add_turn(self, user: str, assistant: str):
        self._history.append(HumanMessage(content = user))
        self._history.append(AIMessage(content = assistant))