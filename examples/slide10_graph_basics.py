"""שקף 10: תהליך כגרף - Nodes, Edges ו-State.

גרף קטן עם שלושה Nodes. בלי מודל בכלל: Nodes הם סתם פונקציות פייתון.
כדאי להריץ ולצייר ביחד על הלוח.
"""
from typing import TypedDict

from langgraph.graph import END, START, StateGraph


# State: המידע המשותף שעובר בגרף
class State(TypedDict):
    text: str


# Nodes: פונקציות שעושות את העבודה. כל אחת מחזירה עדכון חלקי ל-State
def clean(state: State) -> dict:
    print("[clean] מקבל:", repr(state["text"]))
    return {"text": state["text"].strip().lower()}


def shout(state: State) -> dict:
    print("[shout] מקבל:", repr(state["text"]))
    return {"text": state["text"].upper() + "!"}


def sign(state: State) -> dict:
    print("[sign]  מקבל:", repr(state["text"]))
    return {"text": state["text"] + "  -- LangGraph"}


def main():
    builder = StateGraph(State)
    builder.add_node("clean", clean)
    builder.add_node("shout", shout)
    builder.add_node("sign", sign)

    # Edges: קובעים מה ירוץ אחרי מה
    builder.add_edge(START, "clean")
    builder.add_edge("clean", "shout")
    builder.add_edge("shout", "sign")
    builder.add_edge("sign", END)

    graph = builder.compile()

    result = graph.invoke({"text": "   Hello Graph   "})
    print("\nה-State הסופי:", result)

    # תרשים הגרף בפורמט Mermaid. אפשר להדביק ב-https://mermaid.live
    print("\n" + graph.get_graph().draw_mermaid())


if __name__ == "__main__":
    main()
