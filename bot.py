"""Data-analyst Telegram bot — TDS Project 1.

An LLM agent that answers data-analysis questions sent over Telegram.
Replies to every message with exactly one JSON object:
    {"answer": <shaped as the question asks>, "log_url": "<public JSONL log>"}

Architecture:
  - FastAPI app serves /health and /run.jsonl (the public agent log).
  - A background thread long-polls Telegram getUpdates.
  - Each incoming message runs an agentic loop (OpenAI-compatible chat with a
    run_python tool) until the model produces the final JSON answer.
  - A keep-warm thread pings our own public URL so the free host never idles out.
"""

import io
import json
import os
import re
import threading
import time
import traceback
import contextlib
import ast
import math
import statistics
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import requests
from fastapi import FastAPI
from fastapi.responses import FileResponse, PlainTextResponse

from dotenv import load_dotenv
load_dotenv()
# ---------------------------------------------------------------- config
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN", "")
MODEL = os.environ.get("MODEL", "gpt-4o-mini")
MODEL_BASE_URL = os.environ.get("MODEL_BASE_URL", "https://aipipe.org/openai/v1")
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000").rstrip("/")
LOG_PATH = "run.jsonl"
LOG_URL = f"{BASE_URL}/run.jsonl"
TG_API = f"https://api.telegram.org/bot{BOT_TOKEN}"

MAX_AGENT_STEPS = 10
PY_TIMEOUT = 60  # seconds for one run_python call
ANSWER_BUDGET = 210  # wall-clock seconds before we force a final answer
MAX_RECOVERY_ATTEMPTS = 2
MAX_CODE_CHARS = 20000
MAX_OUTPUT_CHARS = 8000

_log_lock = threading.Lock()
_histories: dict[int, list[dict]] = {}  # chat_id -> chat-completion messages
_hist_lock = threading.Lock()


# ---------------------------------------------------------------- logging
def log_event(**fields):
    fields["ts"] = datetime.now(timezone.utc).isoformat()
    line = json.dumps(fields, ensure_ascii=False, default=str)
    with _log_lock:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ---------------------------------------------------------------- tools
ALLOWED_IMPORTS = {
    "pandas", "numpy", "requests", "bs4", "openpyxl", "io", "json",
    "math", "statistics", "re", "datetime", "collections"
}
BLOCKED_CALLS = {
    "eval", "exec", "compile", "__import__", "input", "breakpoint",
    "help", "open", "globals", "locals", "vars", "dir", "getattr",
    "setattr", "delattr"
}
BLOCKED_NAMES = {
    "os", "sys", "subprocess", "socket", "shutil", "pathlib", "builtins",
    "importlib", "ctypes", "signal", "pickle", "marshal", "resource"
}


def validate_python(code: str) -> tuple[bool, str]:
    """Validate model-generated Python before execution.

    This is an application-level guardrail, NOT a security sandbox. It blocks
    common filesystem/process/introspection primitives while preserving the
    data-analysis/network workflow required by the project.
    """
    if not isinstance(code, str) or not code.strip():
        return False, "empty Python code"
    if len(code) > MAX_CODE_CHARS:
        return False, f"code exceeds {MAX_CODE_CHARS} characters"
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        return False, f"syntax error: {e}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    return False, f"blocked import: {root}"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_IMPORTS:
                return False, f"blocked import: {root}"
        elif isinstance(node, ast.Name):
            if node.id in BLOCKED_NAMES:
                return False, f"blocked name: {node.id}"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_CALLS:
                return False, f"blocked call: {node.func.id}"
            if isinstance(node.func, ast.Attribute) and node.func.attr in BLOCKED_CALLS:
                return False, f"blocked attribute call: {node.func.attr}"
        elif isinstance(node, ast.Attribute):
            if node.attr in {"__globals__", "__code__", "__builtins__", "__subclasses__", "__mro__", "__bases__"}:
                return False, f"blocked attribute: {node.attr}"
    return True, "ok"


def profile_dataframe(df):
    """Return a compact deterministic profile for a pandas DataFrame."""
    numeric = df.select_dtypes(include="number").columns.tolist()
    categorical = df.select_dtypes(exclude="number").columns.tolist()
    missing = {str(k): int(v) for k, v in df.isna().sum().items() if int(v) > 0}
    return {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": [str(c) for c in df.columns],
        "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
        "missing": missing,
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": [str(c) for c in numeric],
        "categorical_columns": [str(c) for c in categorical],
        "numeric_summary": df[numeric].describe().round(6).to_dict() if numeric else {},
    }


def run_python(code: str) -> str:
    """Validate and execute analysis code, returning captured stdout/errors."""
    valid, reason = validate_python(code)
    if not valid:
        return f"GUARDRAIL_ERROR: {reason}"

    out = io.StringIO()
    result: dict = {}

    def target():
        env = {"__name__": "__main__", "profile_dataframe": profile_dataframe}
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                exec(code, env)
            result["ok"] = True
        except Exception:
            result["ok"] = False
            out.write("\\n" + traceback.format_exc(limit=4))

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(PY_TIMEOUT)
    if t.is_alive():
        return "ERROR: code timed out after %ss" % PY_TIMEOUT
    text = out.getvalue()
    return text[-MAX_OUTPUT_CHARS:] if text else "(no output — use print())"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Run Python code on the server and get its printed output. "
                "pandas, numpy, requests, bs4, openpyxl are installed and the "
                "network is available (download public datasets with requests). "
                "Always print() what you need to see."
            ),
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Python source to execute"}},
                "required": ["code"],
            },
        },
    }
]

SYSTEM_PROMPT = """You are an expert data-analyst agent answering questions sent to a Telegram bot.

Rules:
1. Work out the answer to the user's LATEST message. Earlier messages in the chat are context for multi-turn tasks.
2. The message may embed data inline, or reference a public dataset (MOSPI, data.gov.in, etc.). Use the run_python tool to fetch data and compute — do not guess numeric results you can compute. For well-known published statistics (e.g. "which state has the highest maternal mortality rate per MOSPI/SRS"), you may answer from reliable knowledge if fetching fails.
3. The message usually spells out the exact JSON shape it wants, e.g. Reply with ONLY {"answer": {"state": "<state>"}, "log_url": "..."}.
4. When you are ready to answer, reply with ONLY that JSON object — no prose, no markdown fences. Use a placeholder like "LOG_URL" for the log_url value; the harness substitutes the real URL. Match the requested shape for "answer" EXACTLY (keys, nesting, types: numbers as numbers unless a string is asked for).
5. If the message does not specify a shape, reply {"answer": <your concise answer>, "log_url": "LOG_URL"}.
6. If a mid-conversation message is only setup/context ("I will send data next"), still reply with {"answer": "ok", "log_url": "LOG_URL"} unless it asks something.
7. Round numbers as instructed; if unspecified, give reasonable precision. Never add keys that were not asked for inside "answer".
"""


# ---------------------------------------------------------------- llm
def chat_completion(messages, use_tools=True):
    body = {"model": MODEL, "messages": messages, "temperature": 0}
    if use_tools:
        body["tools"] = TOOLS
    r = requests.post(
        f"{MODEL_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {AIPIPE_TOKEN}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (data-analyst-bot)",
        },
        json=body,
        timeout=180,
    )
    
    r.raise_for_status()
    return r.json()["choices"][0]["message"]


def extract_json(text: str):
    """Pull the first balanced JSON object out of model text."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.M)
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if esc:
            esc = False
            continue
        if c == "\\":
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def solve(chat_id: int, question: str) -> str:
    """Run the agent loop; return the final JSON reply text."""
    run_started = time.perf_counter()
    with _hist_lock:
        history = _histories.setdefault(chat_id, [])
        history.append({"role": "user", "content": question})
        del history[:-20]
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + list(history)

    log_event(event="question", chat_id=chat_id, text=question)
    final_text = None
    deadline = time.time() + ANSWER_BUDGET
    recovery_attempts = 0
    tool_calls_count = 0

    for step in range(MAX_AGENT_STEPS):
        out_of_time = time.time() > deadline
        if out_of_time:
            messages.append({"role": "user", "content": "Time is up. Reply NOW with only your best final JSON object."})
        try:
            llm_started = time.perf_counter()
            msg = chat_completion(messages, use_tools=not out_of_time)
            llm_latency = time.perf_counter() - llm_started
            log_event(event="llm_response", chat_id=chat_id, step=step, latency_s=round(llm_latency, 4))
        except Exception as e:
            log_event(event="llm_error", chat_id=chat_id, step=step, error=str(e))
            time.sleep(2)
            try:
                llm_started = time.perf_counter()
                msg = chat_completion(messages, use_tools=True)
                log_event(event="llm_retry", chat_id=chat_id, step=step, latency_s=round(time.perf_counter()-llm_started, 4))
            except Exception as e2:
                log_event(event="llm_error_final", chat_id=chat_id, error=str(e2))
                break

        tool_calls = msg.get("tool_calls")
        if tool_calls:
            messages.append(msg)
            for tc in tool_calls:
                tool_calls_count += 1
                try:
                    code = json.loads(tc["function"]["arguments"]).get("code", "")
                except json.JSONDecodeError:
                    code = tc["function"]["arguments"]
                log_event(event="tool_call", chat_id=chat_id, step=step, code=code[:4000])
                tool_started = time.perf_counter()
                output = run_python(code)
                tool_latency = time.perf_counter() - tool_started
                log_event(event="tool_result", chat_id=chat_id, step=step, output=output[:4000], latency_s=round(tool_latency, 4))
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": output})

                if output.startswith("GUARDRAIL_ERROR:") or output.startswith("ERROR:"):
                    if recovery_attempts < MAX_RECOVERY_ATTEMPTS:
                        recovery_attempts += 1
                        messages.append({
                            "role": "user",
                            "content": (
                                "The previous Python tool call failed. Fix the code and retry. "
                                f"This is recovery attempt {recovery_attempts}/{MAX_RECOVERY_ATTEMPTS}. "
                                "Do not repeat blocked operations; use an allowed data-analysis approach."
                            ),
                        })
                        log_event(event="recovery_attempt", chat_id=chat_id, step=step, attempt=recovery_attempts)
            continue

        final_text = msg.get("content") or ""
        break

    obj = extract_json(final_text) if final_text else None
    if obj is None:
        obj = {"answer": (final_text or "unable to determine").strip()[:1000]}
    if "answer" not in obj:
        obj = {"answer": obj}
    obj["log_url"] = LOG_URL
    reply = json.dumps(obj, ensure_ascii=False)

    total_latency = time.perf_counter() - run_started
    log_event(
        event="answer", chat_id=chat_id, reply=reply,
        total_latency_s=round(total_latency, 4),
        tool_calls=tool_calls_count,
        recovery_attempts=recovery_attempts,
    )
    with _hist_lock:
        _histories.setdefault(chat_id, []).append({"role": "assistant", "content": reply})
    return reply


# ---------------------------------------------------------------- telegram
def tg(method, **params):
    r = requests.post(f"{TG_API}/{method}", json=params, timeout=65)

    print("SEND STATUS:", r.status_code)
    print("SEND BODY:", r.text)

    return r.json()


def handle_update(upd):
    msg = upd.get("message") or upd.get("edited_message")
    if not msg:
        return

    text = msg.get("text") or msg.get("caption") or ""
    chat_id = msg["chat"]["id"]

    if not text:
        return

    try:
        reply = solve(chat_id, text)
        
    except Exception:
        print(traceback.format_exc())      # <-- ADD THIS
        log_event(event="agent_crash", chat_id=chat_id, error=traceback.format_exc())
        reply = json.dumps({"answer": "internal error", "log_url": LOG_URL})

    tg("sendMessage", chat_id=chat_id, text=reply)


def poll_loop():
    log_event(event="startup", base_url=BASE_URL, model=MODEL)
    offset = 0
    pool = ThreadPoolExecutor(max_workers=6)
    while True:
        try:
            r = requests.get(
                f"{TG_API}/getUpdates",
                params={"offset": offset, "timeout": 50},
                timeout=65,
            )

            

            resp = r.json()
            for upd in resp.get("result", []):
                offset = upd["update_id"] + 1
                pool.submit(handle_update, upd)
        except Exception as e:
            log_event(event="poll_error", error=str(e))
            time.sleep(5)


def keepwarm_loop():
    """Ping our own public URL so a free host never spins down."""
    while True:
        time.sleep(600)
        try:
            requests.get(f"{BASE_URL}/health", timeout=30)
        except Exception:
            pass


# ---------------------------------------------------------------- web app
app = FastAPI()


@app.on_event("startup")
def _start():
    if not os.path.exists(LOG_PATH):
        log_event(event="log_created")
    threading.Thread(target=poll_loop, daemon=True).start()
    threading.Thread(target=keepwarm_loop, daemon=True).start()


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL, "log_url": LOG_URL}


@app.get("/run.jsonl")
def run_log():
    if os.path.exists(LOG_PATH):
        return FileResponse(LOG_PATH, media_type="application/jsonl; charset=utf-8", filename="run.jsonl")
    return PlainTextResponse("", media_type="application/jsonl")


@app.get("/")
def root():
    return {"service": "data-analyst-telegram-bot", "log_url": LOG_URL}
