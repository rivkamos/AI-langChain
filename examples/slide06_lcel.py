"""שקף 6: LCEL - מחברים Runnables עם סימן ה-pipe (|).

אותו Chain עובד גם עם invoke וגם עם stream, בלי לשנות אותו.
"""
import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def main():
    prompt = ChatPromptTemplate.from_template("Explain {topic} to a 10 year old in two sentences.")
    model = ChatOpenAI(model=MODEL, temperature=0)
    parser = StrOutputParser()

    # Prompt -> Model -> Parser
    chain = prompt | model | parser

    print("סוג ה-Chain:", type(chain).__name__)  # RunnableSequence, וגם הוא Runnable

    print("\n--- invoke ---")
    print(chain.invoke({"topic": "gravity"}))

    print("\n--- stream (אותו Chain, בלי שינוי) ---")
    for chunk in chain.stream({"topic": "gravity"}):
        print(chunk, end="", flush=True)
    print()

    print("\n--- batch ---")
    for answer in chain.batch([{"topic": "rainbows"}, {"topic": "magnets"}]):
        print("*", answer)


if __name__ == "__main__":
    main()
