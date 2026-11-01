"""שקף 14: ענפים ולולאות - הלב של סוכן.

agent (מודל) -> route: האם ביקש כלי? כן -> tools -> חזרה ל-agent. לא -> END.
"""
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


@tool
def get_weather(city: str) -> str:
    """מחזיר מזג אוויר (מדומה) בעיר."""
    fake = {"paris": "18C, cloudy", "tel aviv": "27C, sunny", "london": "14C, rainy"}
    return fake.get(city.lower(), "20C, clear")


@tool
def multiply(a: float, b: float) -> float:
    """מכפיל שני מספרים."""
    return a * b


tools = [get_weather, multiply]
llm_with_tools = ChatOpenAI(model=MODEL, temperature=0).bind_tools(tools)


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# Node שקורא למודל
def agent(state: State) -> dict:
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# פונקציית ניתוב: האם המודל ביקש להפעיל כלי?
def route(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if last.tool_calls else "end"


def main():
    builder = StateGraph(State)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route, {"tools": "tools", "end": END})  # ענף
    builder.add_edge("tools", "agent")  # סוגר את הלולאה
    graph = builder.compile()

    question = "What's the weather in Paris, and what is 12.5 times 8?"
    print("שאלה:", question, "\n")

    # stream מראה כל צעד בלולאה: agent -> tools -> agent
    for step in graph.stream({"messages": [HumanMessage(content=question)]}, stream_mode="updates"):
        for node, update in step.items():
            msg = update["messages"][-1]
            if getattr(msg, "tool_calls", None):
                print(f"[{node}] מבקש כלים:", [(c["name"], c["args"]) for c in msg.tool_calls])
            else:
                print(f"[{node}] {msg.content}")


if __name__ == "__main__":
    main()
