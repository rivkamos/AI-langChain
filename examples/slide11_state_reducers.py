"""שקף 11: State ו-Reducers.

משווים שני גרפים זהים. בראשון שדה messages בלי Reducer, ובשני עם add_messages.
שאלה לכיתה: מה קורה לשיחה בלי add_messages?
"""
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
llm = ChatOpenAI(model=MODEL, temperature=0)


# State בלי Reducer: ערך חדש דורס את הישן
class StateNoReducer(TypedDict):
    messages: list[AnyMessage]


# State עם Reducer: ההודעות החדשות מתווספות לרשימה
class StateWithReducer(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# אותו Node בדיוק בשני הגרפים. מחזיר עדכון חלקי: רק messages
def answer(state) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def build(state_schema):
    builder = StateGraph(state_schema)
    builder.add_node("answer", answer)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    return builder.compile()


def show(title: str, result: dict):
    print(f"\n=== {title} ===")
    print("מספר הודעות ב-State:", len(result["messages"]))
    for m in result["messages"]:
        print(f"  [{m.type}] {m.content[:70]}")


def main():
    initial = {"messages": [HumanMessage(content="במשפט אחד, מהו Reducer ב-LangGraph?")]}

    show("בלי Reducer (דריסה)", build(StateNoReducer).invoke(initial))
    show("עם add_messages (הוספה)", build(StateWithReducer).invoke(initial))

    print("\nמסקנה: בלי add_messages שאלת המשתמש נעלמה, ונשארה רק תשובת המודל.")


if __name__ == "__main__":
    main()
