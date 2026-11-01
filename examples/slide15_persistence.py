"""שקף 15: Persistence - Checkpointer, thread_id ו-Store.

הדגמה מומלצת: אותו thread_id פעמיים (הגרף זוכר), ואז thread_id חדש (שיחה נקייה).
"""
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.store.memory import InMemoryStore

load_dotenv()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
llm = ChatOpenAI(model=MODEL, temperature=0)


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def chat(state: State) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}


def checkpointer_demo():
    print("=== Checkpointer: זיכרון בתוך שיחה ===")
    builder = StateGraph(State)
    builder.add_node("chat", chat)
    builder.add_edge(START, "chat")
    builder.add_edge("chat", END)
    # InMemorySaver לפיתוח. בייצור משתמשים ב-PostgresSaver
    graph = builder.compile(checkpointer=InMemorySaver())

    thread_a = {"configurable": {"thread_id": "thread-A"}}
    thread_b = {"configurable": {"thread_id": "thread-B"}}

    graph.invoke({"messages": [HumanMessage(content="היי, קוראים לי דנה.")]}, thread_a)
    r = graph.invoke({"messages": [HumanMessage(content="איך קוראים לי?")]}, thread_a)
    print("thread-A (אותה שיחה):", r["messages"][-1].content)

    r = graph.invoke({"messages": [HumanMessage(content="איך קוראים לי?")]}, thread_b)
    print("thread-B (שיחה חדשה):", r["messages"][-1].content)

    state = graph.get_state(thread_a)
    print("\nמספר הודעות שנשמרו ב-thread-A:", len(state.values["messages"]))


def store_demo():
    # Store שומר מידע על משתמש בין שיחות שונות. כאן רק מראים את הממשק הבסיסי
    print("\n=== Store: שמירה וקריאה ===")
    store = InMemoryStore()
    store.put(("users", "u1"), "profile", {"name": "דנה", "language": "עברית"})
    item = store.get(("users", "u1"), "profile")
    print("נשמר:", item.value)


def main():
    checkpointer_demo()
    store_demo()


if __name__ == "__main__":
    main()
