#!/usr/bin/env python3
"""Fetch, at run time, the two voxel-pagoda showcase prompts and save verbatim
bytes plus provenance metadata under data/fidelity/voxel/prompts/. Stdlib only
(urllib). Public pages only; no site credentials.

(a) The Artificial Analysis Pagoda Bench prompt: HTML-scraped from the public
    page's rendered "Prompt" panel (a <p> whose raw HTML content is the
    verbatim prompt with HTML entities decoded; the page embeds real
    newlines, not JSON-escaped ones).
(b) The classic community voxel-pagoda prompt from a since-gone (HTTP 402)
    x.com tweet: tried through public tweet-JSON mirrors in order
    (fxtwitter, vxtwitter, the Twitter syndication endpoint), recording
    which source worked. If none works, the failure is recorded and no text
    is invented.
"""
import argparse
import hashlib
import html
import json
import pathlib
import re
import sys
import time
import urllib.error
import urllib.request

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "data" / "fidelity" / "voxel" / "prompts"

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"

PAGODA_BENCH_URL = "https://artificialanalysis.ai/microevals/pagoda-bench-1788018029907"
TWEET_ID = "2082061169158660504"
TWEET_USER = "MiaAI_lab"
TWEET_MIRRORS = [
    ("fxtwitter", f"https://api.fxtwitter.com/{TWEET_USER}/status/{TWEET_ID}"),
    ("vxtwitter", f"https://api.vxtwitter.com/{TWEET_USER}/status/{TWEET_ID}"),
    ("twitter-syndication", f"https://cdn.syndication.twimg.com/tweet-result?id={TWEET_ID}&lang=en"),
]


def fetch(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def save(name, text_bytes, meta):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{name}.txt").write_bytes(text_bytes)
    meta = dict(meta)
    meta["sha256"] = hashlib.sha256(text_bytes).hexdigest()
    meta["bytes"] = len(text_bytes)
    (OUT_DIR / f"{name}.json").write_text(json.dumps(meta, indent=1))
    return meta


# ------------------------------------------------------------- (a) pagoda bench

def extract_pagoda_bench_prompt(page_html):
    """The rendered "Prompt" panel: <p class="...">Prompt</p><p>VERBATIM</p>.
    Raises ValueError if the page structure no longer matches (never invents
    text)."""
    marker = re.search(r'<p[^>]*>Prompt</p><p>', page_html)
    if not marker:
        raise ValueError("could not locate the rendered Prompt panel <p> markers")
    start = marker.end()
    end = page_html.find("</p>", start)
    if end < 0:
        raise ValueError("could not find the closing </p> for the Prompt panel")
    raw = page_html[start:end]
    text = html.unescape(raw)
    if "window.__VOXEL__" not in text or "index.html" not in text:
        raise ValueError("extracted text does not look like the expected pagoda-bench prompt")
    return text


def do_pagoda_bench(args):
    try:
        raw_html = fetch(PAGODA_BENCH_URL, timeout=args.timeout).decode("utf-8", errors="replace")
    except Exception as exc:
        return {"ok": False, "url": PAGODA_BENCH_URL, "fetched_at": now_iso(),
                "method": "html-scrape", "error": f"{exc!r}"}
    try:
        text = extract_pagoda_bench_prompt(raw_html)
    except Exception as exc:
        return {"ok": False, "url": PAGODA_BENCH_URL, "fetched_at": now_iso(),
                "method": "html-scrape", "error": f"{exc!r}"}
    meta = save("pagoda-bench-artificialanalysis", text.encode("utf-8"), {
        "url": PAGODA_BENCH_URL, "fetched_at": now_iso(),
        "method": "html-scrape: rendered <p>Prompt</p><p>VERBATIM</p> panel, HTML-entity-decoded",
    })
    meta["ok"] = True
    return meta


# ------------------------------------------------------------------ (b) tweet

def do_tweet(args):
    last_error = None
    for source_name, url in TWEET_MIRRORS:
        try:
            raw = fetch(url, timeout=args.timeout)
        except Exception as exc:
            last_error = f"{source_name}: {exc!r}"
            continue
        try:
            data = json.loads(raw.decode("utf-8", errors="replace"))
        except Exception as exc:
            last_error = f"{source_name}: not JSON ({exc!r})"
            continue
        text = None
        if source_name in ("fxtwitter", "vxtwitter"):
            tweet = data.get("tweet") or {}
            text = tweet.get("text") or (tweet.get("raw_text") or {}).get("text")
        elif source_name == "twitter-syndication":
            text = data.get("text")
        if not text:
            last_error = f"{source_name}: no tweet text field in response"
            continue
        meta = save("voxel-pagoda-miaai-x", text.encode("utf-8"), {
            "url": f"https://x.com/{TWEET_USER}/status/{TWEET_ID}",
            "mirror_url": url,
            "fetched_at": now_iso(),
            "method": f"tweet-json mirror ({source_name}), verbatim tweet text field",
        })
        meta["ok"] = True
        return meta
    return {"ok": False, "url": f"https://x.com/{TWEET_USER}/status/{TWEET_ID}",
            "fetched_at": now_iso(), "method": "tweet-json mirrors (all failed)",
            "error": last_error or "no mirrors attempted",
            "mirrors_tried": [name for name, _ in TWEET_MIRRORS]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=30)
    ap.add_argument("--which", choices=["pagoda-bench", "tweet", "both"], default="both")
    args = ap.parse_args(argv)

    results = {}
    if args.which in ("pagoda-bench", "both"):
        results["pagoda-bench"] = do_pagoda_bench(args)
    if args.which in ("tweet", "both"):
        results["tweet"] = do_tweet(args)

    for name, r in results.items():
        status = "OK" if r.get("ok") else "FAILED"
        print(f"{name}: {status}  sha256={r.get('sha256', '-')}  bytes={r.get('bytes', '-')}")
        if not r.get("ok"):
            print(f"  error: {r.get('error')}")

    ok = all(r.get("ok") for r in results.values())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
