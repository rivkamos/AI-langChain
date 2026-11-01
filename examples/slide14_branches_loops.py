"""שקף 14: ענפים ולולאות.

הדוגמה הכי פשוטה, בלי כלים ובלי סוכנים:
Node אחד כותב משפט, ופונקציית ניתוב (route) בודקת בקוד רגיל אם הוא ארוך מדי.
ארוך מדי -> חוזרים לכתוב שוב (לולאה). קצר מספיק -> END (ענף יציאה).
"""
import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

load_dotenv()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
llm = ChatOpenAI(model=MODEL, temperature=0)

MAX_WORDS = 8
MAX_ATTEMPTS = 3


class State(TypedDict):
    topic: str
    sentence: str
    attempts: int


# Node שקורא למודל
def write(state: State) -> dict:
    attempts = state["attempts"] + 1
    if attempts == 1:
        prompt = f"כתוב משפט אחד מפורט על {state['topic']}."
    else:
        prompt = f"כתוב משפט אחד על {state['topic']}, באורך של עד {MAX_WORDS} מילים."

    sentence = llm.invoke(prompt).content.strip()
    print(f"[write] ניסיון {attempts}: {len(sentence.split())} מילים -> {sentence}")
    return {"sentence": sentence, "attempts": attempts}


# פונקציית ניתוב: בודקת בקוד רגיל (בלי מודל) לאן ממשיכים
def route(state: State) -> str:
    too_long = len(state["sentence"].split()) > MAX_WORDS
    if too_long and state["attempts"] < MAX_ATTEMPTS:
        return "again"  # לולאה: חוזרים ל-write
    return "done"       # יוצאים


def main():
    builder = StateGraph(State)
    builder.add_node("write", write)

    builder.add_edge(START, "write")
    builder.add_conditional_edges("write", route, {"again": "write", "done": END})
    graph = builder.compile()

    result = graph.invoke({"topic": "חתולים", "sentence": "", "attempts": 0})
    print(f"\nסיימנו אחרי {result['attempts']} ניסיונות.")
    print("המשפט הסופי:", result["sentence"])


if __name__ == "__main__":
    main()
