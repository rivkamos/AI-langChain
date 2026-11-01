"""שקף 9: מתי Chain כבר לא מספיק.

חלק א: Chain של LCEL, מתאים לתהליך קבוע.
חלק ב: אותו Chain בתוך Node של LangGraph, כשצריך לולאה. כאן: כתיבה, ביקורת, ושיפור עד שמאשרים.
זה לא או-או: LCEL בתוך Node, ו-LangGraph לניהול התזרים.
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

llm = ChatOpenAI(model=MODEL, temperature=0)
parser = StrOutputParser()

# --- Chains של LCEL (רכיבים רגילים) ---
write_chain = (
    ChatPromptTemplate.from_template(
        "Write a tagline (max 8 words) for: {product}.\nPrevious feedback (may be empty): {feedback}"
    )
    | llm
    | parser
)
review_chain = (
    ChatPromptTemplate.from_template(
        "Review this tagline: '{tagline}'. If it is at most 6 words, reply exactly 'OK'. "
        "Otherwise reply with one short sentence of feedback."
    )
    | llm
    | parser
)


class State(TypedDict):
    product: str
    tagline: str
    feedback: str
    rounds: int


# --- Nodes: בתוכם רץ ה-Chain ---
def write(state: State) -> dict:
    tagline = write_chain.invoke({"product": state["product"], "feedback": state["feedback"]})
    return {"tagline": tagline, "rounds": state["rounds"] + 1}


def review(state: State) -> dict:
    return {"feedback": review_chain.invoke({"tagline": state["tagline"]})}


def route(state: State) -> str:
    # לולאה: חוזרים לכתיבה עד שהביקורת מאשרת (או עד 3 סיבובים)
    if state["feedback"].strip().upper().startswith("OK") or state["rounds"] >= 3:
        return "done"
    return "again"


def main():
    print("--- חלק א: Chain קבוע ---")
    print(write_chain.invoke({"product": "a smart water bottle", "feedback": ""}))

    print("\n--- חלק ב: אותו Chain בתוך גרף עם לולאה ---")
    builder = StateGraph(State)
    builder.add_node("write", write)
    builder.add_node("review", review)
    builder.add_edge(START, "write")
    builder.add_edge("write", "review")
    builder.add_conditional_edges("review", route, {"again": "write", "done": END})
    graph = builder.compile()

    final = graph.invoke({"product": "a smart water bottle", "tagline": "", "feedback": "", "rounds": 0})
    print("סיבובים:", final["rounds"])
    print("תוצאה:", final["tagline"])
    print("ביקורת אחרונה:", final["feedback"])


if __name__ == "__main__":
    main()
