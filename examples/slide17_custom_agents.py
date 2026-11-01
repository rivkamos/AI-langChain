"""שקף 17: סוכנים בהתאמה אישית - create_agent.

הדוגמה הכי פשוטה: בלי כלים ובלי Middleware, רק מודל והוראה (system_prompt).
create_agent הוא הדרך הסטנדרטית ב-LangChain 1.0 לבנות סוכן. הוא רץ על LangGraph,
ולכן מספיק להוסיף checkpointer כדי לקבל זיכרון בין הודעות, כמו בשקף 15.

כשצריך יותר שליטה: מוסיפים tools ו-middleware, או בונים את הגרף בעצמנו (שקפים 10-16).
"""
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def main():
    agent = create_agent(
        model=ChatOpenAI(model=MODEL, temperature=0),
        system_prompt="אתה עוזר ידידותי. ענה תמיד בעברית ובקצרה.",
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "chat-1"}}

    r = agent.invoke({"messages": [{"role": "user", "content": "היי, קוראים לי דנה."}]}, config)
    print("סוכן:", r["messages"][-1].content)

    # אותו thread_id: הסוכן זוכר
    r = agent.invoke({"messages": [{"role": "user", "content": "איך קוראים לי?"}]}, config)
    print("סוכן:", r["messages"][-1].content)


if __name__ == "__main__":
    main()
