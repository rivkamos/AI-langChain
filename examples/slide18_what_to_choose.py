"""שקף 18: מה לבחור ומתי - שלוש רמות, מהפשוטה לגמישה.

python slide18_what_to_choose.py chain   # תהליך קבוע: LCEL
python slide18_what_to_choose.py agent   # סוכן מוכן: create_agent
python slide18_what_to_choose.py graph   # ענפים, State ושליטה מלאה: LangGraph

כלל אצבע: מתחילים מהפתרון הפשוט ביותר שעונה על הצורך.
"""
import os
import sys
from typing import TypedDict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

load_dotenv()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
llm = ChatOpenAI(model=MODEL, temperature=0)

ARTICLE = (
    "ב-LangChain 1.0 הוצג create_agent כדרך הסטנדרטית לבנות סוכנים. "
    "מתחת לפני השטח הוא רץ על LangGraph, ולכן סוכנים מקבלים שמירה ו-Streaming בחינם."
)


# ---- 1. תהליך קבוע וצפוי -> LCEL Chain ----
def run_chain():
    chain = ChatPromptTemplate.from_template("סכם במשפט אחד:\n{text}") | llm | StrOutputParser()
    print(chain.invoke({"text": ARTICLE}))


# ---- 2. סוכן מוכן -> create_agent ----
def run_agent():
    agent = create_agent(model=llm, system_prompt="אתה עוזר ידידותי. ענה בעברית ובקצרה.")
    result = agent.invoke({"messages": [{"role": "user", "content": "מהו LangGraph? ענה במשפט אחד."}]})
    print(result["messages"][-1].content)


# ---- 3. ענפים, State ושליטה מלאה -> LangGraph ----
class State(TypedDict):
    text: str
    kind: str
    output: str


def run_graph():
    def classify(state: State) -> dict:
        label = llm.invoke(
            f"סווג את ההודעה כ'תלונה' או 'שבח'. ענה במילה אחת בלבד.\n{state['text']}"
        ).content.strip()
        return {"kind": "complaint" if "תלונה" in label else "praise"}

    def handle_complaint(state: State) -> dict:
        return {"output": llm.invoke(f"כתוב תשובה מתנצלת במשפט אחד להודעה: {state['text']}").content}

    def handle_praise(state: State) -> dict:
        return {"output": llm.invoke(f"כתוב תשובה מודה במשפט אחד להודעה: {state['text']}").content}

    builder = StateGraph(State)
    builder.add_node("classify", classify)
    builder.add_node("complaint", handle_complaint)
    builder.add_node("praise", handle_praise)
    builder.add_edge(START, "classify")
    builder.add_conditional_edges("classify", lambda s: s["kind"], {"complaint": "complaint", "praise": "praise"})
    builder.add_edge("complaint", END)
    builder.add_edge("praise", END)
    graph = builder.compile()

    for text in ["החבילה הגיעה שבורה.", "משלוח מהיר ושירות מעולה!"]:
        r = graph.invoke({"text": text, "kind": "", "output": ""})
        print(f"[{r['kind']}] {r['output']}")


def main():
    modes = {"chain": run_chain, "agent": run_agent, "graph": run_graph}
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "all":
        for name, fn in modes.items():
            print(f"\n=== {name} ===")
            fn()
    elif mode in modes:
        modes[mode]()
    else:
        print("שימוש: python slide18_what_to_choose.py [chain|agent|graph]")


if __name__ == "__main__":
    main()
