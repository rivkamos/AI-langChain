"""שקף 16: Human-in-the-loop עם interrupt ו-Command(resume=...).

הגרף כותב טיוטת מייל, עוצר לאישור אדם, וממשיך רק אם אישרו.
שימו לב להדפסה בתחילת approval: אחרי החידוש ה-Node רץ מההתחלה, ולכן היא מופיעה פעמיים.
"""
import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

load_dotenv()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
llm = ChatOpenAI(model=MODEL, temperature=0)


class State(TypedDict):
    topic: str
    draft: str
    approved: bool
    status: str


def write_draft(state: State) -> dict:
    draft = llm.invoke(f"כתוב מייל קצר של שני משפטים ללקוח בנושא: {state['topic']}").content
    return {"draft": draft}


def approval(state: State) -> dict:
    # אזהרה: כל קוד שכאן, לפני interrupt, ירוץ שוב אחרי החידוש
    print("[approval] ה-Node התחיל לרוץ")
    answer = interrupt({"question": "לשלוח את המייל?", "draft": state["draft"]})
    return {"approved": answer == "yes"}


def send(state: State) -> dict:
    print("[send] המייל נשלח (מדומה)")
    return {"status": "sent"}


def cancel(state: State) -> dict:
    print("[cancel] השליחה בוטלה")
    return {"status": "cancelled"}


def after_approval(state: State) -> str:
    return "send" if state["approved"] else "cancel"


def main():
    builder = StateGraph(State)
    builder.add_node("write_draft", write_draft)
    builder.add_node("approval", approval)
    builder.add_node("send", send)
    builder.add_node("cancel", cancel)
    builder.add_edge(START, "write_draft")
    builder.add_edge("write_draft", "approval")
    builder.add_conditional_edges("approval", after_approval, {"send": "send", "cancel": "cancel"})
    builder.add_edge("send", END)
    builder.add_edge("cancel", END)

    # interrupt עובד רק עם Checkpointer: בלי שמירה אין מה לחדש
    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "email-1"}}

    # הרצה ראשונה: נעצרת ב-interrupt
    result = graph.invoke(
        {"topic": "עיכוב בהזמנה שלו", "draft": "", "approved": False, "status": ""}, config
    )
    payload = result["__interrupt__"][0].value
    print("\nהגרף עצר ומחכה לאדם.")
    print(payload["question"])
    print("--- טיוטה ---")
    print(payload["draft"])
    print("-------------")

    decision = input("\nלאשר? (yes/no): ").strip().lower() or "no"

    # חידוש: הערך שמעבירים הוא מה ש-interrupt מחזירה בתוך ה-Node
    final = graph.invoke(Command(resume=decision), config)
    print("\nסטטוס סופי:", final["status"])


if __name__ == "__main__":
    main()
