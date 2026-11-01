"""שקף 5: ממשק Runnable - invoke, stream, batch, הגרסאות האסינכרוניות, retry ו-fallbacks."""
import asyncio
import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

llm = ChatOpenAI(model=MODEL, temperature=0)
prompt = ChatPromptTemplate.from_template("ספר עובדה מעניינת במשפט אחד על {topic}.")
chain = prompt | llm | StrOutputParser()


def sync_methods():
    print("--- invoke: קלט אחד, פלט אחד ---")
    print(chain.invoke({"topic": "תמנונים"}))

    print("\n--- stream: הפלט מגיע בהדרגה ---")
    for chunk in chain.stream({"topic": "הירח"}):
        print(chunk, end="", flush=True)
    print()

    print("\n--- batch: כמה קלטים במקביל ---")
    results = chain.batch([{"topic": "דבורים"}, {"topic": "הרי געש"}, {"topic": "קפה"}])
    for r in results:
        print("*", r)


async def async_methods():
    print("\n--- ainvoke / abatch / astream ---")
    print(await chain.ainvoke({"topic": "פינגווינים"}))
    print(await chain.abatch([{"topic": "תה"}, {"topic": "שחמט"}]))
    async for chunk in chain.astream({"topic": "מאדים"}):
        print(chunk, end="", flush=True)
    print()


def retry_and_fallbacks():
    # retry: אם הקריאה נכשלת, מנסים שוב (עד 3 פעמים)
    robust_llm = llm.with_retry(stop_after_attempt=3)
    print("\n--- with_retry ---")
    print(robust_llm.invoke("אמור שלום במילה אחת.").content)

    # fallbacks: מודל עם שם לא קיים ייכשל, ואז עוברים למודל התקין
    broken_llm = ChatOpenAI(model="this-model-does-not-exist")
    safe_llm = broken_llm.with_fallbacks([llm])
    print("\n--- with_fallbacks ---")
    print(safe_llm.invoke("אמור שלום במילה אחת.").content)


def main():
    sync_methods()
    asyncio.run(async_methods())
    retry_and_fallbacks()


if __name__ == "__main__":
    main()
