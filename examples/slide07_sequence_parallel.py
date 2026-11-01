"""שקף 7: RunnableSequence ו-RunnableParallel.

Sequence: רכיבים בזה אחר זה. Parallel: כמה רכיבים בו-זמנית על אותו קלט.
וגם הקיצורים: מילון הופך ל-Parallel, ופונקציה הופכת ל-Lambda.
"""
import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnableSequence
from langchain_openai import ChatOpenAI

load_dotenv()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

llm = ChatOpenAI(model=MODEL, temperature=0)
parser = StrOutputParser()

TEXT = (
    "LangGraph הוא כלי לבניית תהליכים עם זיכרון. "
    "הוא מתאר תהליך כגרף של צמתים וחיצים, שומר את המצב בין הרצות, "
    "ומאפשר לאדם לאשר פעולה באמצע ההרצה."
)

summary_chain = ChatPromptTemplate.from_template("סכם במשפט אחד:\n{text}") | llm | parser
keywords_chain = ChatPromptTemplate.from_template("חלץ 4 מילות מפתח, מופרדות בפסיקים:\n{text}") | llm | parser


def main():
    # RunnableSequence מפורש. זה בדיוק מה שה-pipe בונה מאחורי הקלעים
    sequence = RunnableSequence(
        ChatPromptTemplate.from_template("תרגם לצרפתית: {text}"), llm, parser
    )
    print("--- RunnableSequence ---")
    print(sequence.invoke({"text": "בוקר טוב"}))

    # RunnableParallel מפורש: שני Chains על אותו קלט, תוצאה כמילון
    parallel = RunnableParallel(summary=summary_chain, keywords=keywords_chain)
    print("\n--- RunnableParallel ---")
    print(parallel.invoke({"text": TEXT}))

    # קיצור: בתוך ביטוי LCEL, מילון הופך אוטומטית ל-RunnableParallel,
    # ופונקציה רגילה הופכת ל-RunnableLambda.
    # תנאי: לפחות אחד מצדי ה-pipe כבר חייב להיות Runnable.
    def format_report(d: dict) -> str:
        return f"סיכום: {d['summary']}\nמילות מפתח: {d['keywords']}"

    # מילון (בצד שמאל) | Runnable (בצד ימין) -> המילון הופך ל-RunnableParallel
    parallel_shortcut = {"summary": summary_chain, "keywords": keywords_chain} | RunnableLambda(format_report)
    print("\n--- מילון הופך ל-RunnableParallel ---")
    print(parallel_shortcut.invoke({"text": TEXT}))

    # Runnable (בצד שמאל) | פונקציה (בצד ימין) -> הפונקציה הופכת ל-RunnableLambda
    shout_chain = summary_chain | (lambda s: s.upper() + "!")
    print("\n--- פונקציה הופכת ל-RunnableLambda ---")
    print(shout_chain.invoke({"text": TEXT}))


if __name__ == "__main__":
    main()
