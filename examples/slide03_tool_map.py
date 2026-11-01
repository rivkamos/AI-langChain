"""שקף 3: מפת הכלים - LangChain, LangGraph ו-LangSmith.

אותה משימה פעמיים:
1. ב-LangChain: Chain (אבני בניין מחוברות).
2. ב-LangGraph: גרף עם Node אחד שמריץ את אותו Chain.
LangGraph רץ מתחת ל-LangChain, והם עובדים ביחד.
"""
import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

load_dotenv()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# --- LangChain: אבני הבניין ---
prompt = ChatPromptTemplate.from_template("הסבר במשפט אחד מהו {topic}.")
llm = ChatOpenAI(model=MODEL, temperature=0)
chain = prompt | llm | StrOutputParser()


# --- LangGraph: גרף עם Node אחד ---
class State(TypedDict):
    topic: str
    answer: str


def explain(state: State) -> dict:
    return {"answer": chain.invoke({"topic": state["topic"]})}


def main():
    print("--- LangChain: Chain ---")
    print(chain.invoke({"topic": "LangChain"}))

    builder = StateGraph(State)
    builder.add_node("explain", explain)
    builder.add_edge(START, "explain")
    builder.add_edge("explain", END)
    graph = builder.compile()

    print("\n--- LangGraph: גרף שמריץ את אותו Chain ---")
    print(graph.invoke({"topic": "LangGraph", "answer": ""})["answer"])

    # LangSmith: מספיק להגדיר ב-.env את LANGSMITH_TRACING=true ואת LANGSMITH_API_KEY,
    # וכל הרצה מופיעה באתר, צעד אחר צעד. אין צורך לשנות את הקוד.
    print("\nLangSmith פעיל:", os.getenv("LANGSMITH_TRACING", "false"))


if __name__ == "__main__":
    main()
