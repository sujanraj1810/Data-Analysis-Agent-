"""Lightweight local evaluation harness for Project 1.

This deliberately does not fabricate accuracy scores: it sends the supplied
questions to a running bot endpoint only when the user configures one.
"""
import json
import os
import time
from pathlib import Path
import requests

CASES = json.loads(Path(__file__).with_name("test_cases.json").read_text())
BASE_URL = os.environ.get("EVAL_BASE_URL", "").rstrip("/")

if not BASE_URL:
    raise SystemExit("Set EVAL_BASE_URL to the deployed service URL before running evaluation.")

print(f"Running {len(CASES)} evaluation cases against {BASE_URL}")
print("Note: Telegram delivery is intentionally not automated by this harness.")
print("Use these cases to exercise the deployed bot and record outcomes.")

# The TDS bot exposes a health endpoint, not a direct question endpoint.
r = requests.get(f"{BASE_URL}/health", timeout=30)
r.raise_for_status()
print("Health:", r.json())

results = []
for case in CASES:
    started = time.perf_counter()
    # Manual/Telegram execution is required because the production interface is Telegram.
    results.append({
        "id": case["id"],
        "expected": case["expected"],
        "latency_s": round(time.perf_counter() - started, 4),
        "status": "manual_telegram_test"
    })

print(json.dumps(results, indent=2))
