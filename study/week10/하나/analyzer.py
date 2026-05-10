from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from system_prompt import ANALYZER_SYSTEM_PROMPT


def build_analyzer_chain(llm: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages([
        ("system", ANALYZER_SYSTEM_PROMPT),
        ("human", "{user_input}"),
    ])
    return prompt | llm | JsonOutputParser()