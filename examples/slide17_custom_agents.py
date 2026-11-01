"""שקף 17: סוכנים בהתאמה אישית - create_agent עם Middleware.

create_agent הוא הדרך הסטנדרטית ב-LangChain 1.0 (create_react_agent של LangGraph הוצא משימוש).
HumanInTheLoopMiddleware עוצר לאישור לפני כלי רגיש. הסוכן רץ על LangGraph, ולכן
יש לו Persistence בחינם (checkpointer), וזה מה שמאפשר לעצור ולחדש.
"""
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

load_dotenv()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


@tool
def search_flights(destination: str) -> str:
    """מחפש טיסות ליעד (מדומה). כלי בטוח, לא צריך אישור."""
    return f"Found 2 flights to {destination}: LY001 at $420, LY002 at $390."


@tool
def book_flight(flight_number: str) -> str:
    """מזמין טיסה (מדומה). כלי רגיש, דורש אישור אדם."""
    return f"Flight {flight_number} booked successfully."


def main():
    agent = create_agent(
        model=ChatOpenAI(model=MODEL, temperature=0),
        tools=[search_flights, book_flight],
        system_prompt="You are a travel assistant. Search first, then book the cheapest flight.",
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "book_flight": True,      # עוצר לאישור
                    "search_flights": False,  # רץ ישר
                }
            )
        ],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "trip-1"}}

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Book me the cheapest flight to Rome."}]}, config
    )

    if "__interrupt__" in result:
        request = result["__interrupt__"][0].value
        print("הסוכן עצר ומבקש אישור:")
        print(request)

        choice = input("\nלאשר? (approve/reject): ").strip().lower() or "reject"
        if choice == "approve":
            decision = {"type": "approve"}
        else:
            decision = {"type": "reject", "message": "The user declined the booking."}

        result = agent.invoke(Command(resume={"decisions": [decision]}), config)

    print("\nתשובה סופית:", result["messages"][-1].content)


if __name__ == "__main__":
    main()
