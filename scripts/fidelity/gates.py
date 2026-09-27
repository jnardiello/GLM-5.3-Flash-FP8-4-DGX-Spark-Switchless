#!/usr/bin/env python3
"""Run the two post-boot functional gates of docs/operations.md and check their pass rules.

Gate 1: coherent response with thinking off (names Rome). Gate 2: structured tool call
(`get_weather` with JSON arguments containing Milan). Also records /health and the elapsed
time since the caller-supplied health timestamp, because both gates must pass within two
minutes of /health 200.

    python3 scripts/fidelity/gates.py --base-url http://HOST:8000 --out gates.json
"""
import argparse
import json
import sys
import time
import urllib.request

GATE1 = {"model": "glm-5.3-flash", "temperature": 0, "max_tokens": 64,
         "chat_template_kwargs": {"enable_thinking": False},
         "messages": [{"role": "user", "content": "What is the capital of Italy? Reply with one sentence."}]}
GATE2 = {"model": "glm-5.3-flash", "max_tokens": 256,
         "messages": [{"role": "user", "content": "What is the weather in Milan?"}],
         "tools": [{"type": "function", "function": {
             "name": "get_weather", "description": "Get weather for a city",
             "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                            "required": ["city"]}}}],
         "tool_choice": "auto"}


def call(base, path, body=None, timeout=300):
    req = urllib.request.Request(base.rstrip("/") + path,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        return r.status, (json.loads(raw) if raw and body is not None else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="glm-5.3-flash")
    ap.add_argument("--health-since", type=float, help="unix time of the first /health 200")
    ap.add_argument("--out")
    a = ap.parse_args()
    t0 = time.time()
    rep = {"started": t0}
    rep["health"], _ = call(a.base_url, "/health")
    g1, g2 = dict(GATE1, model=a.model), dict(GATE2, model=a.model)
    _, r1 = call(a.base_url, "/v1/chat/completions", g1)
    content = (r1["choices"][0]["message"].get("content") or "")
    rep["gate1"] = {"pass": "rome" in content.lower() or "roma" in content.lower(),
                    "content": content[:200], "finish_reason": r1["choices"][0].get("finish_reason")}
    _, r2 = call(a.base_url, "/v1/chat/completions", g2)
    calls = r2["choices"][0]["message"].get("tool_calls") or []
    ok2, args = False, None
    if calls:
        fn = calls[0]["function"]
        try:
            args = json.loads(fn["arguments"])
            ok2 = fn["name"] == "get_weather" and "milan" in json.dumps(args).lower()
        except (ValueError, TypeError):
            ok2 = False
    rep["gate2"] = {"pass": ok2, "tool": calls[0]["function"]["name"] if calls else None, "arguments": args}
    rep["elapsed_s"] = round(time.time() - t0, 1)
    if a.health_since:
        rep["since_health_s"] = round(time.time() - a.health_since, 1)
        rep["within_two_minutes"] = rep["since_health_s"] <= 120
    rep["pass"] = rep["health"] == 200 and rep["gate1"]["pass"] and rep["gate2"]["pass"] \
        and rep.get("within_two_minutes", True)
    text = json.dumps(rep, indent=2)
    print(text)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
