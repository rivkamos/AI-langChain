"""שקף 4: אבני הבניין - Prompt Template, Chat Model, Output Parser.

מריצים כל רכיב בנפרד, ידנית, כדי לראות מה כל אחד מקבל ומה הוא מחזיר.
"""
import os

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def main():
    # 1. Prompt Template: תבנית עם משתנים -> הודעות
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a concise teacher."),
            ("human", "Explain {concept} in one sentence."),
        ]
    )
    messages = prompt.invoke({"concept": "a Runnable"})
    print("1) Prompt הפך לקלט של הודעות:")
    print(messages, "\n")

    # 2. Chat Model: מקבל הודעות, מחזיר הודעת AI
    llm = ChatOpenAI(model=MODEL, temperature=0)
    ai_message = llm.invoke(messages)
    print("2) המודל החזיר:", type(ai_message).__name__)
    print(ai_message.content, "\n")

    # 3. Output Parser: ממיר את התשובה למבנה שנוח לעבוד איתו
    text = StrOutputParser().invoke(ai_message)
    print("3) טקסט נקי:", repr(text), "\n")

    # Parser מסוג JSON: מקבלים מילון פייתון ולא מחרוזת
    json_prompt = ChatPromptTemplate.from_template(
        "Return a JSON object with the keys 'name' and 'year' for the programming "
        "language {language}. Return only JSON."
    )
    json_llm = llm.bind(response_format={"type": "json_object"})
    data = (json_prompt | json_llm | JsonOutputParser()).invoke({"language": "Python"})
    print("JSON כמילון:", data, "| סוג:", type(data).__name__)


if __name__ == "__main__":
    main()
