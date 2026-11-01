"""שקף 12: גרף ראשון - ארבעה צעדים: State, add_node, add_edge, compile."""
import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

load_dotenv()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
llm = ChatOpenAI(model=MODEL, temperature=0)


# צעד 1: מגדירים State
class State(TypedDict):
    question: str
    answer: str


# Node: מקבל State, מחזיר עדכון חלקי
def answer_node(state: State) -> dict:
    response = llm.invoke(state["question"])
    return {"answer": response.content}


def main():
    builder = StateGraph(State)

    # צעד 2: מוסיפים Nodes
    builder.add_node("answer", answer_node)

    # צעד 3: מחברים Edges
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)

    # צעד 4: compile. כאן נבדקים חיבורי הגרף, ורק אחריו אפשר להריץ
    graph = builder.compile()

    result = graph.invoke({"question": "What is LangGraph, in one sentence?"})
    print("ה-State שחזר מהגרף:")
    print(result)

    print("\nאו רק התשובה:")
    print(result["answer"])


if __name__ == "__main__":
    main()
