"""שקף 15: Persistence - Checkpointer, thread_id ו-Store.

הדגמה מומלצת: אותו thread_id פעמיים (הגרף זוכר), ואז thread_id חדש (שיחה נקייה).
בסוף: Store ששומר מידע על משתמש בין שיחות שונות.
"""
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.store.base import BaseStore
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

    graph.invoke({"messages": [HumanMessage(content="Hi, my name is Dana.")]}, thread_a)
    r = graph.invoke({"messages": [HumanMessage(content="What is my name?")]}, thread_a)
    print("thread-A:", r["messages"][-1].content)

    r = graph.invoke({"messages": [HumanMessage(content="What is my name?")]}, thread_b)
    print("thread-B (חדש):", r["messages"][-1].content)

    state = graph.get_state(thread_a)
    print("\nמספר הודעות שנשמרו ב-thread-A:", len(state.values["messages"]))


def store_demo():
    print("\n=== Store: זיכרון על משתמש בין שיחות ===")
    store = InMemoryStore()
    # שומרים פרופיל למשתמש u1
    store.put(("users", "u1"), "profile", {"name": "Dana", "language": "Hebrew"})

    def chat_with_profile(state: State, config, *, store: BaseStore) -> dict:
        user_id = config["configurable"]["user_id"]
        item = store.get(("users", user_id), "profile")
        profile = item.value if item else {}
        system = SystemMessage(content=f"User profile: {profile}. Greet them by name, in one short sentence.")
        return {"messages": [llm.invoke([system] + state["messages"])]}

    builder = StateGraph(State)
    builder.add_node("chat", chat_with_profile)
    builder.add_edge(START, "chat")
    builder.add_edge("chat", END)
    graph = builder.compile(checkpointer=InMemorySaver(), store=store)

    # שתי שיחות (thread) שונות, אותו משתמש
    for thread in ["conv-1", "conv-2"]:
        cfg = {"configurable": {"thread_id": thread, "user_id": "u1"}}
        r = graph.invoke({"messages": [HumanMessage(content="Hello!")]}, cfg)
        print(f"{thread}:", r["messages"][-1].content)


def main():
    checkpointer_demo()
    store_demo()


if __name__ == "__main__":
    main()
