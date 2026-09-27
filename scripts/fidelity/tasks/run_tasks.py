#!/usr/bin/env python3
"""Endpoint layer for the vendored knapcio qeval/hardset task sets.

This module owns the HTTP/arm/mode/cost logic. It imports task definitions
and graders from the vendored, unmodified third_party/knapcio-bench modules
and never edits them; it does not reuse their CLIs, only qeval_tasks.TASKS
(id/category/thinking/max_tokens/prompt/checker) and hardset.PROMPTS
(id/category/prompt).

Arms:
  local  --base-url http://HOST:PORT   POST /v1/chat/completions
         model discovered from GET /v1/models unless --model is given.
         chat_template_kwargs: {"reasoning_effort": EFFORT}
  zai    --zai                         POST https://api.z.ai/api/paas/v4/chat/completions
         model "glm-5.3-flash", top-level "reasoning_effort", do_sample:false
         for greedy. Authorization: Bearer $ZAI_API_KEY (env only, never
         logged/written). --max-usd is REQUIRED for this arm.

Modes:
  greedy   local: temperature 0, 1 run.  zai: do_sample false, 3 repeats.
  sampled  temperature 1.0, top_p 0.95, 5 runs both arms. Local requests
           carry a deterministic seed per (item, run).

Settings (deviate from the vendored task set's own per-task budgets and are
recorded per request as both the override and the task's own default):
  reasoning_effort default "high" for every item.
  max_tokens default 16384 for every item.

Output: data/fidelity/tasks/<arm>/<set>/<mode>/run<k>/<item>.json

Security: qeval checkers execute model-generated Python with the caller's permissions.
Run qeval in a disposable container, VM or unprivileged account without credentials
or private files.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import pathlib
import sys
import threading
import time
import urllib.error
import urllib.request

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
VENDOR_DIR = REPO_ROOT / "third_party" / "knapcio-bench" / "bench"
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

import qeval_tasks as qt  # noqa: E402  (vendored, unmodified)
import hardset as hs  # noqa: E402  (vendored, unmodified)

ZAI_URL = "https://api.z.ai/api/paas/v4/chat/completions"
ZAI_MODEL = "glm-5.3-flash"
# $/token. "cached_input" applies to usage.prompt_tokens_details.cached_tokens
# when the API reports it; reasoning bills as output per the brief.
ZAI_PRICE = {"input": 0.15e-6, "cached_input": 0.03e-6, "output": 0.50e-6}

DEFAULT_EFFORT = "high"
DEFAULT_MAX_TOKENS = 16384
DEFAULT_TIMEOUT = 1800
DEFAULT_RETRIES = 5


def eprint(*a, **k):
    print(*a, file=sys.stderr, **k)
    sys.stderr.flush()


# --------------------------------------------------------------------- items

def load_items(set_name):
    """Return a uniform list of dicts: id, category, prompt, and (qeval only)
    task_default_max_tokens/task_default_effort/checker."""
    if set_name == "qeval":
        out = []
        for t in qt.TASKS:
            out.append({
                "id": t["id"],
                "category": t["category"],
                "prompt": t["prompt"],
                "checker": t["checker"],
                "task_default_max_tokens": t["max_tokens"],
                "task_default_effort": "high" if t["thinking"] else "low",
            })
        return out
    if set_name == "hardset":
        out = []
        for pid, cat, prompt in hs.PROMPTS:
            out.append({
                "id": pid,
                "category": cat,
                "prompt": prompt,
                "checker": None,
                "task_default_max_tokens": 4096,
                "task_default_effort": "high",
            })
        return out
    raise ValueError(f"unknown --set {set_name!r}")


# -------------------------------------------------------------------- spend

class SpendLedger:
    """Tracks z.ai USD spend and refuses a request that could exceed the cap
    assuming worst case (max_tokens output, no cached input)."""

    def __init__(self, path, max_usd):
        self.path = path
        self.max_usd = max_usd
        self.lock = threading.Lock()
        self.spent = 0.0
        self.entries = []
        if path.exists():
            try:
                data = json.loads(path.read_text())
                self.spent = data.get("spent_usd", 0.0)
                self.entries = data.get("entries", [])
            except Exception:
                pass

    @staticmethod
    def worst_case_cost(prompt_tokens, max_tokens):
        return prompt_tokens * ZAI_PRICE["input"] + max_tokens * ZAI_PRICE["output"]

    @staticmethod
    def actual_cost(usage):
        pt = usage.get("prompt_tokens", 0) or 0
        ct = usage.get("completion_tokens", 0) or 0
        cached = 0
        details = usage.get("prompt_tokens_details") or {}
        if isinstance(details, dict):
            cached = details.get("cached_tokens", 0) or 0
        cached = min(cached, pt)
        uncached = pt - cached
        return uncached * ZAI_PRICE["input"] + cached * ZAI_PRICE["cached_input"] + ct * ZAI_PRICE["output"]

    def would_exceed(self, prompt_tokens, max_tokens):
        with self.lock:
            return self.spent + self.worst_case_cost(prompt_tokens, max_tokens) > self.max_usd

    def record(self, item_id, run_idx, usage):
        cost = self.actual_cost(usage)
        with self.lock:
            self.spent += cost
            self.entries.append({
                "item": item_id, "run": run_idx, "usage": usage,
                "cost_usd": round(cost, 6), "cumulative_usd": round(self.spent, 6),
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
            self._flush()
        return cost

    def _flush(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({
            "spent_usd": round(self.spent, 6), "max_usd": self.max_usd, "entries": self.entries,
        }, indent=1))
        tmp.replace(self.path)


# ------------------------------------------------------------------- request

def approx_tokens(text):
    """Cheap, dependency-free estimate used only for the pre-flight cost
    guard on prompt tokens before the first real usage is known."""
    return max(1, len(text) // 4)


def deterministic_seed(item_id, run_idx):
    h = hashlib.sha256(f"{item_id}:{run_idx}".encode()).digest()
    return int.from_bytes(h[:4], "big")


def build_request(arm, model, prompt, mode, effort, max_tokens, item_id, run_idx):
    messages = [{"role": "user", "content": prompt}]
    if arm == "local":
        body = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
            "chat_template_kwargs": {"reasoning_effort": effort},
        }
        if mode == "greedy":
            body["temperature"] = 0
        else:
            body["temperature"] = 1.0
            body["top_p"] = 0.95
            body["seed"] = deterministic_seed(item_id, run_idx)
        return body
    # zai
    body = {
        "model": ZAI_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "reasoning_effort": effort,
    }
    if mode == "greedy":
        body["do_sample"] = False
    else:
        body["do_sample"] = True
        body["temperature"] = 1.0
        body["top_p"] = 0.95
    return body


def http_post_json(url, body, headers, timeout, retries=DEFAULT_RETRIES):
    data = json.dumps(body).encode()
    last_exc = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code == 429 or 500 <= exc.code < 600:
                backoff = min(60, 2 ** attempt)
                eprint(f"  retry after HTTP {exc.code} in {backoff}s (attempt {attempt + 1}/{retries})")
                time.sleep(backoff)
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as exc:
            last_exc = exc
            backoff = min(60, 2 ** attempt)
            eprint(f"  retry after {exc!r} in {backoff}s (attempt {attempt + 1}/{retries})")
            time.sleep(backoff)
            continue
    raise last_exc


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


# --------------------------------------------------------------- run 1 item

def result_is_valid(path):
    if not path.exists():
        return False
    try:
        d = json.loads(path.read_text())
    except Exception:
        return False
    return d.get("ok") is True and "response" in d


def run_one(args, item, mode, run_idx, out_dir, ledger, headers, base_url, model):
    out_path = out_dir / f"{item['id']}.json"
    if args.resume and result_is_valid(out_path):
        return "skipped", item["id"]

    effort = args.reasoning_effort
    max_tokens = args.max_tokens
    body = build_request(args.arm, model, item["prompt"], mode, effort, max_tokens, item["id"], run_idx)

    if args.arm == "zai":
        prompt_tok_est = approx_tokens(item["prompt"])
        if ledger.would_exceed(prompt_tok_est, max_tokens):
            record = {
                "ok": False, "arm": args.arm, "set": args.set, "mode": mode, "run": run_idx,
                "item": item["id"], "category": item["category"],
                "request": {k: v for k, v in body.items() if k != "messages"} | {"prompt_chars": len(item["prompt"])},
                "reasoning_effort_used": effort, "max_tokens_used": max_tokens,
                "task_default_effort": item["task_default_effort"],
                "task_default_max_tokens": item["task_default_max_tokens"],
                "error": "cost-cap: worst-case cost would exceed --max-usd; request skipped",
            }
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(record, indent=1))
            return "cost-capped", item["id"]

    url = ZAI_URL if args.arm == "zai" else base_url.rstrip("/") + "/v1/chat/completions"
    t0 = time.time()
    try:
        resp = http_post_json(url, body, headers, args.timeout, args.retries)
        wall = time.time() - t0
    except Exception as exc:
        record = {
            "ok": False, "arm": args.arm, "set": args.set, "mode": mode, "run": run_idx,
            "item": item["id"], "category": item["category"],
            "request": {k: v for k, v in body.items() if k != "messages"} | {"prompt_chars": len(item["prompt"])},
            "reasoning_effort_used": effort, "max_tokens_used": max_tokens,
            "task_default_effort": item["task_default_effort"],
            "task_default_max_tokens": item["task_default_max_tokens"],
            "wall_s": round(time.time() - t0, 3),
            "error": f"{exc!r}",
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(record, indent=1))
        return "error", item["id"]

    choice = resp["choices"][0]
    msg = choice["message"]
    usage = resp.get("usage", {}) or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""

    if args.arm == "zai":
        ledger.record(item["id"], run_idx, usage)

    record = {
        "ok": True, "arm": args.arm, "set": args.set, "mode": mode, "run": run_idx,
        "item": item["id"], "category": item["category"],
        "request": {k: v for k, v in body.items() if k != "messages"} | {"prompt_chars": len(item["prompt"])},
        "reasoning_effort_used": effort, "max_tokens_used": max_tokens,
        "task_default_effort": item["task_default_effort"],
        "task_default_max_tokens": item["task_default_max_tokens"],
        "response": {
            "content": content,
            "reasoning_content": reasoning,
            "finish_reason": choice.get("finish_reason"),
        },
        "usage": usage,
        "wall_s": round(wall, 3),
    }
    if args.set == "qeval" and item["checker"] is not None:
        try:
            ok, why = item["checker"](content)
        except Exception as exc:
            ok, why = False, f"checker raised {exc!r}"
        record["grader"] = {"pass": bool(ok), "why": "" if ok else str(why)[:200]}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=1, ensure_ascii=False))
    return ("pass" if record.get("grader", {}).get("pass") else "done"), item["id"]


# ------------------------------------------------------------------- driver

def runs_for(arm, mode, override):
    if override:
        return override
    if mode == "greedy":
        return 3 if arm == "zai" else 1
    return 5


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=["qeval", "hardset"], required=True)
    ap.add_argument("--mode", choices=["greedy", "sampled"], required=True)
    ap.add_argument("--base-url", help="local vLLM arm, e.g. http://127.0.0.1:8000")
    ap.add_argument("--zai", action="store_true", help="use the z.ai arm instead of --base-url")
    ap.add_argument("--label", help="arm name for output paths (e.g. Cp, R0); default: local/zai")
    ap.add_argument("--model", default=None, help="override model id (local: else from /v1/models)")
    ap.add_argument("--max-usd", type=float, default=None, help="REQUIRED with --zai: hard spend cap")
    ap.add_argument("--reasoning-effort", default=DEFAULT_EFFORT)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--runs", type=int, default=0, help="override the mode/arm default run count")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    ap.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    ap.add_argument("--out-root", default=str(REPO_ROOT / "data" / "fidelity" / "tasks"))
    ap.add_argument("--resume", dest="resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--limit", type=int, default=0, help="first N items (smoke test)")
    args = ap.parse_args(argv)

    if args.zai and args.base_url:
        ap.error("pass either --base-url or --zai, not both")
    if not args.zai and not args.base_url:
        ap.error("one of --base-url or --zai is required")
    if args.zai and args.max_usd is None:
        ap.error("--max-usd is REQUIRED with --zai")

    args.arm = "zai" if args.zai else "local"
    args.label = args.label or args.arm

    headers = {}
    model = None
    if args.arm == "zai":
        key = os.environ.get("ZAI_API_KEY")
        if not key:
            ap.error("ZAI_API_KEY must be set in the environment for --zai")
        headers = {"Authorization": f"Bearer {key}"}
    else:
        model = resolve_local_model(args.base_url, args.model)

    items = load_items(args.set)
    if args.limit:
        items = items[: args.limit]

    n_runs = runs_for(args.arm, args.mode, args.runs)
    out_dir_root = pathlib.Path(args.out_root) / args.label / args.set / args.mode
    ledger_path = pathlib.Path(args.out_root) / args.label / "spend-ledger.json"
    ledger = SpendLedger(ledger_path, args.max_usd) if args.arm == "zai" else None

    print(f"set={args.set} arm={args.arm} mode={args.mode} runs={n_runs} items={len(items)} "
          f"effort={args.reasoning_effort} max_tokens={args.max_tokens} concurrency={args.concurrency}")

    jobs = []
    for run_idx in range(1, n_runs + 1):
        out_dir = out_dir_root / f"run{run_idx}"
        for item in items:
            jobs.append((item, run_idx, out_dir))

    counts = {}
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(run_one, args, item, args.mode, run_idx, out_dir, ledger, headers, args.base_url or "", model)
                for item, run_idx, out_dir in jobs]
        for fut in cf.as_completed(futs):
            status, item_id = fut.result()
            counts[status] = counts.get(status, 0) + 1
            print(f"  {status:<12} {item_id}")

    print(f"\ndone: {counts}")
    if ledger is not None:
        print(f"zai spend: ${ledger.spent:.4f} / ${ledger.max_usd:.2f} cap -> {ledger_path}")
    # Collection failures and cost-capped skips fail the run; grader failures are results.
    return 1 if counts.get("error") or counts.get("cost-capped") else 0


if __name__ == "__main__":
    sys.exit(main())
