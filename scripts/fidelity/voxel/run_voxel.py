#!/usr/bin/env python3
"""Send a saved voxel-pagoda prompt (bytes unchanged, one user message) to a
local vLLM arm or z.ai, one greedy run plus three sampled runs, and extract
the returned HTML document.

Uses only the stdlib for the HTTP/arm logic (same shape as
scripts/fidelity/tasks/run_tasks.py); reasoning-token counting falls back to
the tokenizer under data/fidelity/tokenizer/ via the `tokenizers` package
when usage does not report reasoning tokens directly (not required by the
offline stdlib test, so this optional import is deferred).

Output:
  docs/fidelity/voxel/<arm>/<prompt>/<run>.html   the extracted HTML document
  data/fidelity/voxel/raw/<arm>/<prompt>/<run>.json   the full raw response
"""
import argparse
import json
import os
import pathlib
import re
import sys
import time
import urllib.error
import urllib.request

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
PROMPTS_DIR = REPO_ROOT / "data" / "fidelity" / "voxel" / "prompts"
RAW_DIR = REPO_ROOT / "data" / "fidelity" / "voxel" / "raw"
HTML_DIR = REPO_ROOT / "docs" / "fidelity" / "voxel"
TOKENIZER_DIR = REPO_ROOT / "data" / "fidelity" / "tokenizer"

ZAI_URL = "https://api.z.ai/api/paas/v4/chat/completions"
ZAI_MODEL = "glm-5.3-flash"

DEFAULT_LOCAL_MAX_TOKENS = 65536
DEFAULT_TIMEOUT = 3600


def eprint(*a, **k):
    print(*a, file=sys.stderr, **k)


# --------------------------------------------------------------- extraction

_HTML_FENCE_RE = re.compile(r"```(?:html)?\s*\n(.*?)```", re.S)
_DOCTYPE_RE = re.compile(r"<!DOCTYPE html>.*?</html>", re.S | re.I)


def extract_html(text):
    """Prefer a fenced ```html block; else the <!DOCTYPE html>...</html> span."""
    fences = _HTML_FENCE_RE.findall(text)
    for candidate in sorted(fences, key=len, reverse=True):
        if "<!DOCTYPE" in candidate or "<html" in candidate.lower():
            return candidate.strip()
    m = _DOCTYPE_RE.search(text)
    if m:
        return m.group(0).strip()
    if fences:
        return max(fences, key=len).strip()
    return None


# ------------------------------------------------------------------ tokenizer

_tokenizer_cache = {}


def count_tokens(text):
    if not text:
        return 0
    if "tok" not in _tokenizer_cache:
        try:
            from tokenizers import Tokenizer
            _tokenizer_cache["tok"] = Tokenizer.from_file(str(TOKENIZER_DIR / "tokenizer.json"))
        except Exception as exc:
            eprint(f"tokenizer unavailable ({exc!r}); falling back to a chars/4 estimate")
            _tokenizer_cache["tok"] = None
    tok = _tokenizer_cache["tok"]
    if tok is None:
        return max(1, len(text) // 4)
    return len(tok.encode(text).ids)


# --------------------------------------------------------------------- prompts

def load_prompt(name):
    path = PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"no saved prompt at {path}; run fetch_prompts.py first")
    return path.read_bytes()


# ------------------------------------------------------------------- request

def build_request(arm, model, prompt_text, mode, effort, max_tokens):
    messages = [{"role": "user", "content": prompt_text}]
    if arm == "local":
        body = {"model": model, "messages": messages, "max_tokens": max_tokens, "stream": False,
                "chat_template_kwargs": {"reasoning_effort": effort}}
        body["temperature"] = 0 if mode == "greedy" else 1.0
        if mode == "sampled":
            body["top_p"] = 0.95
        return body
    body = {"model": ZAI_MODEL, "messages": messages, "max_tokens": max_tokens,
            "reasoning_effort": effort}
    body["do_sample"] = mode == "sampled"
    if mode == "sampled":
        body["temperature"] = 1.0
        body["top_p"] = 0.95
    return body


def http_post_json(url, body, headers, timeout):
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def resolve_local_model(base_url, explicit_model, timeout=30):
    if explicit_model:
        return explicit_model
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/models")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode())
    models = data.get("data") or []
    if not models:
        raise RuntimeError("GET /v1/models returned no models; pass --model")
    return models[0]["id"]


def run_one(args, prompt_name, prompt_text, mode, run_idx, headers, base_url, model):
    max_tokens = args.max_tokens
    body = build_request(args.arm, model, prompt_text, mode, args.reasoning_effort, max_tokens)
    url = ZAI_URL if args.arm == "zai" else base_url.rstrip("/") + "/v1/chat/completions"

    t0 = time.time()
    resp = http_post_json(url, body, headers, args.timeout)
    wall = time.time() - t0

    choice = resp["choices"][0]
    msg = choice["message"]
    usage = resp.get("usage", {}) or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""

    reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    if reasoning_tokens is None:
        reasoning_tokens = count_tokens(reasoning)

    record = {
        "ok": True, "arm": args.arm, "label": args.label, "prompt": prompt_name, "mode": mode, "run": run_idx,
        "request": {k: v for k, v in body.items() if k != "messages"},
        "response": {"content": content, "reasoning_content": reasoning,
                      "finish_reason": choice.get("finish_reason")},
        "usage": usage,
        "output_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": reasoning_tokens,
        "wall_s": round(wall, 3),
    }

    raw_dir = RAW_DIR / args.label / prompt_name
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{mode}-run{run_idx}.json").write_text(json.dumps(record, indent=1, ensure_ascii=False))

    html_doc = extract_html(content)
    html_dir = HTML_DIR / args.label / prompt_name
    html_dir.mkdir(parents=True, exist_ok=True)
    html_path = html_dir / f"{mode}-run{run_idx}.html"
    if html_doc:
        html_path.write_text(html_doc, encoding="utf-8")
    else:
        eprint(f"  WARNING: no HTML document extracted for {prompt_name} {mode} run{run_idx}")

    print(f"  {prompt_name} {mode} run{run_idx}: {record['output_tokens']} out tok, "
          f"{reasoning_tokens} reasoning tok, {wall:.1f}s, finish={record['response']['finish_reason']}, "
          f"html_extracted={bool(html_doc)}")
    return record


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", action="append", required=True,
                     help="prompt name under data/fidelity/voxel/prompts/ (repeatable)")
    ap.add_argument("--base-url")
    ap.add_argument("--zai", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--reasoning-effort", default="high")
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    ap.add_argument("--label", help="arm name for output paths (e.g. Cp, R0); default: local/zai")
    ap.add_argument("--sampled-runs", type=int, default=3, help="sampled runs per prompt after the greedy one")
    args = ap.parse_args(argv)

    if args.zai and args.base_url:
        ap.error("pass either --base-url or --zai, not both")
    if not args.zai and not args.base_url:
        ap.error("one of --base-url or --zai is required")
    args.arm = "zai" if args.zai else "local"
    args.label = args.label or args.arm

    headers = {}
    model = None
    if args.arm == "zai":
        key = os.environ.get("ZAI_API_KEY")
        if not key:
            ap.error("ZAI_API_KEY must be set in the environment for --zai")
        headers = {"Authorization": f"Bearer {key}"}
        # z.ai caps max_tokens; leave unset (server default) unless overridden.
        if args.max_tokens is None:
            args.max_tokens = 8192
    else:
        model = resolve_local_model(args.base_url, args.model)
        if args.max_tokens is None:
            args.max_tokens = DEFAULT_LOCAL_MAX_TOKENS

    for prompt_name in args.prompt:
        prompt_text = load_prompt(prompt_name).decode("utf-8")
        print(f"=== {prompt_name} ({len(prompt_text)} chars) ===")
        run_one(args, prompt_name, prompt_text, "greedy", 1, headers, args.base_url or "", model)
        for run_idx in range(1, args.sampled_runs + 1):
            run_one(args, prompt_name, prompt_text, "sampled", run_idx, headers, args.base_url or "", model)

    return 0


if __name__ == "__main__":
    sys.exit(main())
