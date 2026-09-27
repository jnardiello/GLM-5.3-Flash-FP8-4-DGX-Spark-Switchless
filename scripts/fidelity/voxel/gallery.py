#!/usr/bin/env python3
"""Build a self-contained, anonymised gallery from render.py's checks.json:
shuffled sample labels (seeded), screenshots, and a link to each HTML file.
The label -> arm/prompt/mode/run key is written separately (private) so the
gallery itself stays blind. Stdlib only.

Output:
  docs/fidelity/voxel/gallery.html        the gallery page (light/dark, phone-width friendly)
  data/fidelity/voxel/gallery-key.json    label -> {arm, prompt, mode, run, html_path, screenshot}
"""
import argparse
import html
import json
import pathlib
import random
import re
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
VOXEL_DOCS_DIR = REPO_ROOT / "docs" / "fidelity" / "voxel"
KEY_PATH = REPO_ROOT / "data" / "fidelity" / "voxel" / "gallery-key.json"

_RUN_RE = re.compile(r"^(greedy|sampled)-run(\d+)$")


def parse_entry(html_rel_path):
    """docs/fidelity/voxel/<arm>/<prompt>/<mode>-run<k>.html -> dict, or None."""
    parts = pathlib.Path(html_rel_path).parts
    # parts: ("docs","fidelity","voxel", arm, prompt, "<mode>-run<k>.html")
    try:
        idx = parts.index("voxel")
    except ValueError:
        return None
    tail = parts[idx + 1:]
    if len(tail) != 3:
        return None
    arm, prompt, filename = tail
    m = _RUN_RE.match(pathlib.Path(filename).stem)
    if not m:
        return None
    return {"arm": arm, "prompt": prompt, "mode": m.group(1), "run": int(m.group(2))}


def load_checks(checks_path):
    data = json.loads(checks_path.read_text())
    return data.get("results", [])


def build_entries(results):
    entries = []
    for r in results:
        parsed = parse_entry(r["html_path"])
        if not parsed:
            continue
        entry = dict(parsed)
        entry["html_path"] = r["html_path"]
        entry["screenshot"] = r.get("screenshot")
        entry["console_error_count"] = len(r.get("console_errors") or [])
        entry["measured_fps_relative"] = r.get("measured_fps_relative")
        constraints = r.get("constraints")
        if constraints:
            entry["constraints_ok"] = all(v for k, v in constraints.items() if k.endswith("_ok"))
        entries.append(entry)
    return entries


GALLERY_CSS = """
:root { --bg:#f7f7f8; --fg:#1a1a1a; --card:#ffffff; --border:#e2e2e6; --accent:#5b6cff; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#14141a; --fg:#eaeaf0; --card:#1e1e26; --border:#2c2c36; --accent:#8b9aff; }
}
* { box-sizing: border-box; }
body { margin:0; padding:1.5rem; background:var(--bg); color:var(--fg);
       font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
h1 { font-size:1.4rem; margin:0 0 .25rem; }
p.sub { color:#8888; margin:0 0 1.5rem; }
.grid { display:grid; gap:1rem; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); }
.card { background:var(--card); border:1px solid var(--border); border-radius:10px; overflow:hidden; }
.card img { width:100%; display:block; background:#000; aspect-ratio: 1280/800; object-fit:cover; }
.card .body { padding:.75rem .9rem; }
.card .label { font-weight:600; }
.card .meta { font-size:.8rem; color:#8888; margin-top:.25rem; }
.card a { color:var(--accent); text-decoration:none; font-size:.85rem; }
.card a:hover { text-decoration:underline; }
.badge { display:inline-block; font-size:.7rem; padding:.1rem .4rem; border-radius:6px; margin-left:.4rem; }
.badge.ok { background:#1c7a3a22; color:#1c7a3a; }
.badge.bad { background:#a3232322; color:#a32323; }
@media (max-width: 480px) {
  body { padding:.75rem; }
  .grid { grid-template-columns: 1fr; }
}
"""


def render_card(label, entry):
    img = html.escape(entry["screenshot"]) if entry.get("screenshot") else None
    link = html.escape(entry["html_path"])
    rel_img = f"../../{img}" if img else None  # gallery.html lives at docs/fidelity/voxel/
    rel_link = f"../../{link}"
    badge = ""
    if "constraints_ok" in entry:
        badge = (f'<span class="badge {"ok" if entry["constraints_ok"] else "bad"}">'
                  f'{"constraints OK" if entry["constraints_ok"] else "constraints FAILED"}</span>')
    img_tag = f'<img src="{rel_img}" alt="{html.escape(label)}" loading="lazy">' if rel_img else ""
    fps = entry.get("measured_fps_relative")
    errs = entry.get("console_error_count", 0)
    return f"""
    <div class="card">
      {img_tag}
      <div class="body">
        <div class="label">{html.escape(label)}{badge}</div>
        <div class="meta">measured fps (relative): {fps if fps is not None else '-'} &middot; console errors: {errs}</div>
        <div class="meta"><a href="{rel_link}" target="_blank" rel="noopener">open HTML</a></div>
      </div>
    </div>"""


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checks", default=str(VOXEL_DOCS_DIR / "checks.json"))
    ap.add_argument("--out", default=str(VOXEL_DOCS_DIR / "gallery.html"))
    ap.add_argument("--key-out", default=str(KEY_PATH))
    ap.add_argument("--seed", type=int, default=20260927)
    args = ap.parse_args(argv)

    checks_path = pathlib.Path(args.checks)
    if not checks_path.exists():
        print(f"no checks file at {checks_path}; run render.py first", file=sys.stderr)
        return 1

    entries = build_entries(load_checks(checks_path))
    rng = random.Random(args.seed)
    order = list(range(len(entries)))
    rng.shuffle(order)

    key = {}
    cards_html = []
    for i, idx in enumerate(order):
        label = f"Sample {chr(ord('A') + i)}" if i < 26 else f"Sample {i + 1}"
        entry = entries[idx]
        key[label] = entry
        cards_html.append(render_card(label, entry))

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Voxel pagoda showcase gallery (anonymised)</title>
<style>{GALLERY_CSS}</style>
</head>
<body>
  <h1>Voxel pagoda showcase gallery</h1>
  <p class="sub">Anonymised, shuffled (seed {args.seed}). {len(entries)} samples. The label &rarr; arm key is kept separately and privately.</p>
  <div class="grid">
    {''.join(cards_html)}
  </div>
</body>
</html>
"""
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")

    key_path = pathlib.Path(args.key_out)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(json.dumps({"seed": args.seed, "key": key}, indent=1))

    print(f"wrote {out_path} ({len(entries)} samples)")
    print(f"wrote {key_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
