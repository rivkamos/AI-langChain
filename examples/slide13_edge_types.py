"""שקף 13: ארבעה סוגי מעבר בין Nodes.

1. add_edge - מעבר קבוע
2. add_conditional_edges - ניתוב דינמי לפי State
3. Send - פיזור לכמה Nodes במקביל (Map-Reduce)
4. Command - עדכון State וניתוב באותה פעולה
"""
import operator
import os
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send

load_dotenv()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
llm = ChatOpenAI(model=MODEL, temperature=0)


# ---------- 1. add_edge: תמיד לאותו Node ----------
class FixedState(TypedDict):
    text: str


def demo_fixed_edge():
    print("=== 1. add_edge (קבוע) ===")
    builder = StateGraph(FixedState)
    builder.add_node("a", lambda s: {"text": s["text"] + " -> a"})
    builder.add_node("b", lambda s: {"text": s["text"] + " -> b"})
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("b", END)
    print(builder.compile().invoke({"text": "start"})["text"])


# ---------- 2. add_conditional_edges: לפי ה-State ----------
class RouteState(TypedDict):
    question: str
    answer: str


def demo_conditional_edges():
    print("\n=== 2. add_conditional_edges (דינמי) ===")

    def route(state: RouteState) -> Literal["quick", "detailed"]:
        return "quick" if len(state["question"]) < 30 else "detailed"

    def quick(state: RouteState) -> dict:
        return {"answer": "[קצר] " + llm.invoke(state["question"] + " (ענה ב-5 מילים)").content}

    def detailed(state: RouteState) -> dict:
        return {"answer": "[מפורט] " + llm.invoke(state["question"] + " (ענה ב-2 משפטים)").content}

    builder = StateGraph(RouteState)
    builder.add_node("quick", quick)
    builder.add_node("detailed", detailed)
    builder.add_conditional_edges(START, route)
    builder.add_edge("quick", END)
    builder.add_edge("detailed", END)
    graph = builder.compile()

    print(graph.invoke({"question": "מהי בירת ספרד?"})["answer"])
    print(graph.invoke({"question": "למה השמיים כחולים ביום ואדומים בשקיעה?"})["answer"])


# ---------- 3. Send: Map-Reduce ----------
class OverallState(TypedDict):
    topics: list[str]
    jokes: Annotated[list[str], operator.add]  # Reducer: מאחד את התוצאות של כל ה-Nodes


class JokeState(TypedDict):
    topic: str


def demo_send():
    print("\n=== 3. Send (פיזור במקביל) ===")

    def fan_out(state: OverallState):
        # לכל נושא נוצרת הרצה נפרדת של make_joke
        return [Send("make_joke", {"topic": t}) for t in state["topics"]]

    def make_joke(state: JokeState) -> dict:
        joke = llm.invoke(f"ספר בדיחה קצרה מאוד על {state['topic']}.").content
        return {"jokes": [f"{state['topic']}: {joke}"]}

    builder = StateGraph(OverallState)
    builder.add_node("make_joke", make_joke)
    builder.add_conditional_edges(START, fan_out, ["make_joke"])
    builder.add_edge("make_joke", END)
    result = builder.compile().invoke({"topics": ["חתולים", "פייתון", "קפה"], "jokes": []})
    for j in result["jokes"]:
        print("*", j)


# ---------- 4. Command: עדכון + ניתוב ביחד ----------
class MoodState(TypedDict):
    text: str
    mood: str
    reply: str


def demo_command():
    print("\n=== 4. Command (עדכון + ניתוב) ===")

    def classify(state: MoodState) -> Command[Literal["happy", "sad"]]:
        mood = "happy" if "מצוין" in state["text"] else "sad"
        # גם מעדכן את ה-State וגם קובע לאן ממשיכים, בלי Edge מותנה
        return Command(update={"mood": mood}, goto=mood)

    def happy(state: MoodState) -> dict:
        return {"reply": "שמח לשמוע!"}

    def sad(state: MoodState) -> dict:
        return {"reply": "אני כאן אם צריך."}

    builder = StateGraph(MoodState)
    builder.add_node("classify", classify)
    builder.add_node("happy", happy)
    builder.add_node("sad", sad)
    builder.add_edge(START, "classify")
    builder.add_edge("happy", END)
    builder.add_edge("sad", END)
    graph = builder.compile()

    for text in ["היום היה מצוין", "היה לי יום קשה"]:
        r = graph.invoke({"text": text})
        print(f"{text!r} -> mood={r['mood']}, reply={r['reply']}")


def main():
    demo_fixed_edge()
    demo_conditional_edges()
    demo_send()
    demo_command()


if __name__ == "__main__":
    main()
