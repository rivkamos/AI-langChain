"""שקף 3: מפת הכלים - LangChain, LangGraph ו-LangSmith.

מראה ש-create_agent של LangChain הוא בעצם גרף של LangGraph שרץ מתחת.
"""
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


@tool
def word_length(word: str) -> int:
    """מחזיר את מספר האותיות במילה."""
    return len(word)


def main():
    llm = ChatOpenAI(model=MODEL, temperature=0)

    # LangChain: דרך אחת וקצרה לבנות סוכן (גרסה 1.0)
    agent = create_agent(
        model=llm,
        tools=[word_length],
        system_prompt="You are a helpful assistant. Use tools when needed.",
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "How many letters are in the word 'framework'?"}]}
    )
    print("תשובה:", result["messages"][-1].content)

    # LangGraph: מתחת לסוכן יש גרף עם Nodes. אפשר להסתכל עליו
    print("\nסוג האובייקט:", type(agent).__name__)
    print("Nodes בגרף של הסוכן:", list(agent.get_graph().nodes.keys()))

    # LangSmith: מספיק להגדיר LANGSMITH_TRACING=true ו-LANGSMITH_API_KEY ב-.env,
    # וכל ההרצה מופיעה באתר עם כל שלב ושלב. אין צורך לשנות את הקוד.
    print("\nLangSmith פעיל:", os.getenv("LANGSMITH_TRACING", "false"))


if __name__ == "__main__":
    main()
