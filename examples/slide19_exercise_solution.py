"""שקף 19: פתרון לדוגמה לתרגיל המסכם - תהליך מחקר קטן.

שלב 1: גרף עם שלושה Nodes: plan, search, answer
שלב 2: Edge מותנה - אם חסר מידע חוזרים ל-search
שלב 3: InMemorySaver ו-thread_id
שלב 4: interrupt לאישור לפני התשובה

(למרצה: מומלץ לא לשתף עם הסטודנטיות לפני שסיימו.)
"""
import operator
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

load_dotenv()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
llm = ChatOpenAI(model=MODEL, temperature=0)

MIN_RESULTS = 3  # כמה ממצאים צריך לפני שעוברים לתשובה

# "מנוע חיפוש" מדומה. מי שמסיים מהר יכול להחליף אותו בכלי אמיתי
FAKE_WEB = [
    "LangGraph מתאר תהליך כגרף של צמתים (Nodes) וחיצים (Edges).",
    "ה-Checkpointer שומר את מצב הגרף אחרי כל צעד, לפי thread_id.",
    "הפונקציה interrupt עוצרת את הגרף, ו-Command(resume=...) ממשיך אותו.",
    "ה-Reducers קובעים איך עדכוני ה-Nodes משתלבים ב-State.",
]


class State(TypedDict):
    question: str
    plan: str
    results: Annotated[list[str], operator.add]  # Reducer: כל חיפוש מוסיף ממצא
    answer: str


def plan(state: State) -> dict:
    text = llm.invoke(f"כתוב תוכנית מחקר בשני שלבים לשאלה: {state['question']}").content
    print("[plan] התוכנית נוצרה")
    return {"plan": text}


def search(state: State) -> dict:
    index = len(state["results"])  # כל הרצה מביאה את הממצא הבא
    finding = FAKE_WEB[index % len(FAKE_WEB)]
    print(f"[search] ממצא {index + 1}: {finding}")
    return {"results": [finding]}


def enough_info(state: State) -> str:
    # Edge מותנה: חסר מידע -> חוזרים ל-search
    return "answer" if len(state["results"]) >= MIN_RESULTS else "search"


def answer(state: State) -> dict:
    # שלב 4: עוצרים לאישור אדם לפני שמייצרים את התשובה
    approved = interrupt(
        {"question": "נאספו ממצאים. להפיק תשובה?", "findings": state["results"]}
    )
    if approved != "yes":
        return {"answer": "(המשתמש ביטל לפני הפקת התשובה)"}
    sources = "\n".join(f"- {r}" for r in state["results"])
    text = llm.invoke(
        f"ענה על השאלה בעזרת הממצאים בלבד.\nשאלה: {state['question']}\nממצאים:\n{sources}"
    ).content
    return {"answer": text}


def main():
    builder = StateGraph(State)
    builder.add_node("plan", plan)
    builder.add_node("search", search)
    builder.add_node("answer", answer)
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "search")
    builder.add_conditional_edges("search", enough_info, {"search": "search", "answer": "answer"})
    builder.add_edge("answer", END)

    graph = builder.compile(checkpointer=InMemorySaver())  # שלב 3
    config = {"configurable": {"thread_id": "research-1"}}

    result = graph.invoke(
        {"question": "איך השמירה והחידוש ב-LangGraph עובדים?", "plan": "", "results": [], "answer": ""},
        config,
    )

    request = result["__interrupt__"][0].value
    print("\n" + request["question"])
    decision = input("yes/no: ").strip().lower() or "no"

    final = graph.invoke(Command(resume=decision), config)
    print("\nתשובה:\n" + final["answer"])


if __name__ == "__main__":
    main()
