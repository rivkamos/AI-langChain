"""שקף 8: עוד Runnables - Passthrough, Lambda, Branch, Assign, with_fallbacks, with_retry."""
import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_openai import ChatOpenAI

load_dotenv()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

llm = ChatOpenAI(model=MODEL, temperature=0)
parser = StrOutputParser()


def passthrough_demo():
    print("--- RunnablePassthrough: מעביר את הקלט כמו שהוא ---")
    chain = RunnableParallel(original=RunnablePassthrough(), upper=RunnableLambda(str.upper))
    print(chain.invoke("hello"))


def lambda_demo():
    print("\n--- RunnableLambda: עוטף פונקציה רגילה ---")
    word_count = RunnableLambda(lambda text: len(text.split()))
    print(word_count.invoke("one two three four"))


def assign_demo():
    print("\n--- RunnableAssign: מוסיף מפתחות למילון הקלט ---")
    chain = RunnablePassthrough.assign(
        word_count=lambda x: len(x["text"].split()),
        shout=lambda x: x["text"].upper(),
    )
    print(chain.invoke({"text": "langchain is fun"}))


def branch_demo():
    print("\n--- RunnableBranch: ניתוב לפי תנאי ---")
    question_chain = ChatPromptTemplate.from_template("ענה בקצרה: {text}") | llm | parser
    statement_chain = ChatPromptTemplate.from_template("הגב במשפט ידידותי אחד: {text}") | llm | parser
    branch = RunnableBranch(
        (lambda x: x["text"].strip().endswith("?"), question_chain),
        statement_chain,  # ברירת מחדל
    )
    print(branch.invoke({"text": "מהי בירת איטליה?"}))
    print(branch.invoke({"text": "סיימתי את האפליקציה הראשונה שלי ב-LangChain."}))


def fallbacks_and_retry_demo():
    print("\n--- with_fallbacks + with_retry ---")
    broken = ChatOpenAI(model="this-model-does-not-exist")
    safe = broken.with_fallbacks([llm.with_retry(stop_after_attempt=3)])
    print(safe.invoke("אמור 'בסדר' ותו לא.").content)


def main():
    passthrough_demo()
    lambda_demo()
    assign_demo()
    branch_demo()
    fallbacks_and_retry_demo()


if __name__ == "__main__":
    main()
