"""ממשק Streamlit להרצת דוגמאות LangChain / LangGraph של מפגש 6."""
import os
import subprocess
import sys
from pathlib import Path

import streamlit as st
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent
EXAMPLES = ROOT / "examples"
TIMEOUT_SEC = 120

# (מספר שקף, שם קובץ או None אם אין קוד, כותרת, האם דרוש OPENAI_API_KEY)
SLIDES = [
    (1, None, "כותרת", False),
    (2, None, "מה נלמד היום", False),
    (3, "slide03_tool_map.py", "מפת הכלים", True),
    (4, "slide04_building_blocks.py", "אבני הבניין", True),
    (5, "slide05_runnable_interface.py", "ממשק Runnable", True),
    (6, "slide06_lcel.py", "LCEL", True),
    (7, "slide07_sequence_parallel.py", "Sequence ו-Parallel", True),
    (8, "slide08_more_runnables.py", "עוד Runnables", True),
    (9, "slide09_chain_vs_graph.py", "מתי Chain לא מספיק", True),
    (10, "slide10_graph_basics.py", "תהליך כגרף", False),
    (11, "slide11_state_reducers.py", "State ו-Reducers", True),
    (12, "slide12_first_graph.py", "גרף ראשון", True),
    (13, "slide13_edge_types.py", "סוגי מעבר", True),
    (14, "slide14_branches_loops.py", "ענפים ולולאות", True),
    (15, "slide15_persistence.py", "Persistence", True),
    (16, "slide16_human_in_the_loop.py", "Human-in-the-loop", True),
    (17, "slide17_custom_agents.py", "סוכנים בהתאמה אישית", True),
    (18, "slide18_what_to_choose.py", "מה לבחור ומתי", True),
    (19, "slide19_exercise_solution.py", "תרגיל מסכם", True),
    (20, None, "מה לקחנו", False),
    (21, None, "מקורות", False),
]
SLIDE_BY_NUM = {s[0]: s for s in SLIDES}

# שקפים שקוראים input(): ברירת מחדל ל-stdin
STDIN_DEFAULTS = {16: "yes", 17: "approve", 19: "yes"}
# שקף 18: ארגומנט אופציונלי (ריק = מריצים הכול)
SLIDE18_ARGS = {"הכול": None, "chain": "chain", "agent": "agent", "graph": "graph"}

st.set_page_config(layout="wide", page_title="LangChain - Session 6")

# RTL לטקסט הממשק, קוד ופלט נשארים LTR
st.markdown(
    """
    <style>
    .stApp, [data-testid="stSidebar"] { direction: rtl; text-align: right; }
    pre, code, [data-testid="stCode"], [data-testid="stCodeBlock"] {
        direction: ltr !important; text-align: left !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def label(num: int) -> str:
    _, filename, title, _ = SLIDE_BY_NUM[num]
    return f"שקף {num}: {title}" + ("" if filename else " (אין קוד)")


def has_api_key() -> bool:
    """מפתח בסביבה או בקובץ .env בשורש הפרויקט."""
    return bool(os.getenv("OPENAI_API_KEY") or dotenv_values(ROOT / ".env").get("OPENAI_API_KEY"))


def run_script(path: Path, args: list[str], stdin_text: str) -> tuple[int | None, str]:
    """מריץ קובץ בתהליך נפרד. מחזיר (קוד יציאה או None אם נגמר הזמן, פלט משולב)."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    try:
        proc = subprocess.run(
            [sys.executable, "-u", str(path), *args],
            cwd=ROOT,
            env=env,
            input=stdin_text,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # stdout ו-stderr יחד
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=TIMEOUT_SEC,
        )
        return proc.returncode, proc.stdout
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        return None, out


# --- Sidebar ---
st.sidebar.title("שקפים")
choice = st.sidebar.radio(
    "בחרי שקף",
    [s[0] for s in SLIDES],
    format_func=label,
    label_visibility="collapsed",
)

# --- Main ---
num = choice
_, filename, title, needs_key = SLIDE_BY_NUM[num]
st.header(f"שקף {num}: {title}")

if filename is None:
    st.info("אין קוד לשקף הזה")
    st.stop()

path = EXAMPLES / filename
st.code(path.read_text(encoding="utf-8"), language="python")

stdin_text = ""
if num in STDIN_DEFAULTS:
    stdin_text = st.text_input(
        "Keyboard input (stdin)", value=STDIN_DEFAULTS[num], key=f"stdin_{num}"
    )

args: list[str] = []
if num == 18:
    arg_label = st.selectbox("ארגומנט", list(SLIDE18_ARGS), key="arg_18")
    if SLIDE18_ARGS[arg_label]:
        args = [SLIDE18_ARGS[arg_label]]

missing_key = needs_key and not has_api_key()
if missing_key:
    st.warning("לא נמצא OPENAI_API_KEY (לא ב-.env ולא במשתני הסביבה). ההרצה כנראה תיכשל.")

col_run, col_clear, _ = st.columns([1, 1, 6])
run_clicked = col_run.button("Execute", type="primary", key=f"run_{num}")
clear_clicked = col_clear.button("Clear output", key=f"clear_{num}")

# התוצאה נשמרת ב-session_state לפי שקף, כדי שתישאר בין רינדורים
if clear_clicked:
    st.session_state.pop(f"result_{num}", None)

if run_clicked:
    with st.spinner("רץ..."):
        # stdin מסתיים ב-newline כדי ש-input() יחזור
        st.session_state[f"result_{num}"] = run_script(path, args, stdin_text + "\n")

result = st.session_state.get(f"result_{num}")
if result is not None:
    code, output = result
    if code is None:
        st.error(f"ההרצה הופסקה אחרי {TIMEOUT_SEC} שניות (timeout).")
    elif code != 0:
        st.error(f"הריצה נכשלה (exit code {code}). הפלט כולל את ה-traceback:")
    st.code(output or "(אין פלט)", language="text")
