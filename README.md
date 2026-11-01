# LangChain ו-LangGraph - דוגמאות קוד למפגש 6

ריפו ליווי להרצאה "LangChain ו-LangGraph (מפגש 6)". לכל שקף שיש בו מה להדגים בקוד יש קובץ פייתון נפרד
בתיקייה `examples/`. כל הקבצים עובדים מול ה-API של OpenAI, וכל קובץ עומד בפני עצמו ואפשר להריץ אותו לבד.

## התקנה

```bash
python -m venv .venv
source .venv/bin/activate        # ב-Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env             # ואז עורכים את .env ומכניסים OPENAI_API_KEY
```

## הרצה

```bash
python examples/slide06_lcel.py
```

המודל נקבע ב-`OPENAI_MODEL` בקובץ `.env` (ברירת מחדל: `gpt-4o-mini`).

## הרצה עם ממשק גרפי

```bash
streamlit run app.py             # ב-Windows: py -m streamlit run app.py
```

בסרגל הצד בוחרים שקף, רואים את הקוד, ולוחצים על Execute. הקבצים רצים בתהליך נפרד והפלט מוצג בדף.
בשקפים 16, 17 ו-19 יש תיבה להזנת קלט המקלדת (stdin), ובשקף 18 בוחרים את הארגומנט (`chain`, `agent`, `graph`).

## מפת השקפים והקבצים

| שקף | נושא | קובץ | מה רואים בהרצה |
|---|---|---|---|
| 1 | כותרת | - | אין קוד |
| 2 | מה נלמד היום | - | אין קוד |
| 3 | מפת הכלים | `slide03_tool_map.py` | `create_agent` ושהוא גרף של LangGraph. הערה על LangSmith |
| 4 | אבני הבניין | `slide04_building_blocks.py` | Prompt, Model ו-Parser, כל אחד לבד |
| 5 | ממשק Runnable | `slide05_runnable_interface.py` | invoke, stream, batch, גרסאות async, retry, fallbacks |
| 6 | LCEL | `slide06_lcel.py` | `prompt \| model \| parser` ואותו Chain ב-invoke, stream ו-batch |
| 7 | Sequence ו-Parallel | `slide07_sequence_parallel.py` | סיכום ומילות מפתח במקביל, מילון כ-Parallel |
| 8 | עוד Runnables | `slide08_more_runnables.py` | Passthrough, Lambda, Assign, Branch, fallbacks, retry |
| 9 | מתי Chain לא מספיק | `slide09_chain_vs_graph.py` | Chain קבוע, ואז אותו Chain בתוך גרף עם לולאה |
| 10 | תהליך כגרף | `slide10_graph_basics.py` | שלושה Nodes בלי מודל, ותרשים Mermaid |
| 11 | State ו-Reducers | `slide11_state_reducers.py` | השוואה: בלי Reducer ועם `add_messages` |
| 12 | גרף ראשון | `slide12_first_graph.py` | ארבעת הצעדים, ו-State שחוזר מ-`invoke` |
| 13 | סוגי מעבר | `slide13_edge_types.py` | `add_edge`, `add_conditional_edges`, `Send`, `Command` |
| 14 | ענפים ולולאות | `slide14_branches_loops.py` | סוכן עם כלים: agent, route, tools וחזרה |
| 15 | Persistence | `slide15_persistence.py` | `thread_id` זהה מול חדש, ו-Store בין שיחות |
| 16 | Human-in-the-loop | `slide16_human_in_the_loop.py` | `interrupt` ו-`Command(resume=...)`, ה-Node רץ שוב |
| 17 | סוכנים בהתאמה אישית | `slide17_custom_agents.py` | `create_agent` עם `HumanInTheLoopMiddleware` |
| 18 | מה לבחור ומתי | `slide18_what_to_choose.py` | `chain`, `agent` או `graph` לפי ארגומנט |
| 19 | תרגיל מסכם | `slide19_exercise_solution.py` | פתרון לדוגמה: plan, search, answer עם לולאה, שמירה ואישור |
| 20 | מה לקחנו | - | אין קוד |
| 21 | מקורות | - | אין קוד |

## הערות למרצה

- שקפים 16, 17 ו-19 מבקשים קלט מהמקלדת (אישור) באמצע ההרצה.
- שקף 5 ושקף 8 כוללים בכוונה מודל עם שם לא קיים כדי להראות `with_fallbacks` בפעולה.
- שקף 19 הוא פתרון התרגיל. כדאי לא לחלק אותו לפני שהסטודנטיות סיימו.
- ה-API של LangChain ושל LangGraph מתעדכן. אם משהו נשבר אחרי עדכון גרסה, בודקים בתיעוד הרשמי.
