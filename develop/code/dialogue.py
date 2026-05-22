import re
from langchain_core.messages import HumanMessage, AIMessage

AGE_PATTERN = re.compile(
    r"(만\s*)?\d+\s*(개월|세)\s*(미만|이하)|소아|영아|유아|젖먹이|신생아|영유아|어린이|^나이$"
)

PREG_PATTERN = re.compile(r"임부|임신 가능성|임산부|수유부|수유 중")

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

    
    def _normalize_subject(self, subject: str) -> str:
        if AGE_PATTERN.search(subject):
            return "나이"
        elif PREG_PATTERN.search(subject):
            return "임산부/수유부"
        return subject
    

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
        # 숫자 나이가 있는지 먼저 확인
        has_numeric_age = any(
            k == "나이" and isinstance(v, (int, float))
            for k, v in situation.items()
        )

        print(f"[situation] {situation}")
        for subject, value in situation.items():
            # 나이가 숫자로 추출된 경우 10세 이하면 True, 초과면 False
            if subject == "나이" and isinstance(value, (int, float)):
                self.extra_context["나이"] = float(value)
                continue

            normalized = self._normalize_subject(subject)

            # "소아", "영유아" 등 나이 키워드인데 숫자 없이 True로 온 경우
            if normalized == "나이" and isinstance(value, bool):
                if not has_numeric_age:
                    self.extra_context["소아여부"] = value
                continue
            
            if normalized in caution_subjects:
                if normalized not in self.caution_slots:
                    self.caution_slots[normalized] = value
            else:
                if subject not in self.extra_context:
                    self.extra_context[subject] = value


    # 역질문에 대한 응답을 저장. 
    # (해당 함수 호출 전에 긍정/부정 응답 판단을 위한 함수인 parse_yes_no 호출한 상태 -> parse_yes_no가 반환한 bool 타입 값을 저장.)
    def record_clarify_answer(self, subject: str, is_positive: bool):
        self.caution_slots[subject] = is_positive
        #print(f"[record_clarify_answer] {subject}: {is_positive}")
    

    def get_history(self) -> list:
        return self._history

    def add_turn(self, user: str, assistant: str):
        self._history.append(HumanMessage(content = user))
        self._history.append(AIMessage(content = assistant))