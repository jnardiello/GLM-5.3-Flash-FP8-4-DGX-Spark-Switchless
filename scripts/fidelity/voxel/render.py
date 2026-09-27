#!/usr/bin/env python3
"""Render every extracted voxel HTML file with headless Chromium (Playwright)
at 1280x800, capture console errors and a 5s screenshot, count frames over
5s with an injected requestAnimationFrame counter (relative only: this is
software WebGL under headless Chromium, not a GPU-backed measurement), and
for Pagoda Bench prompts read window.__VOXEL__ and check the stated
constraints. Writes docs/fidelity/voxel/checks.json.

Requires the `playwright` package and a Chromium install
(`python -m playwright install chromium`); not part of the stdlib offline
test suite for that reason.
"""
import argparse
import json
import pathlib
import re
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
VOXEL_DOCS_DIR = REPO_ROOT / "docs" / "fidelity" / "voxel"

RAF_INIT_SCRIPT = """
window.__frameCount = 0;
window.__rafErrors = [];
(function loop(){
  window.__frameCount++;
  requestAnimationFrame(loop);
})();
window.addEventListener('error', function(e){ window.__rafErrors.push(String(e.message)); });
"""

MAX_BYTES = 700 * 1024
VOXEL_COUNT_RANGE = (15000, 30000)
EXPECTED_PAGODA_FLOORS = 5
MAX_DRAW_CALLS = 12
MIN_FPS = 55

_LOCAL_REF_RE = re.compile(r'(?:src|href)\s*=\s*"([^"]+)"', re.I)


def is_pagoda_bench(name):
    return "pagoda-bench" in name or "pagoda_bench" in name


def check_single_file(html_path, html_text):
    """No local external references: every src/href is either http(s), a
    data: URI, or an in-page anchor (#...)."""
    offenders = []
    for m in _LOCAL_REF_RE.finditer(html_text):
        ref = m.group(1)
        if ref.startswith(("http://", "https://", "data:", "#")):
            continue
        offenders.append(ref)
    size = html_path.stat().st_size
    return {
        "size_bytes": size,
        "size_ok": size < MAX_BYTES,
        "local_external_refs": offenders,
        "single_file_ok": not offenders,
    }


def render_one(playwright_module, html_path):
    from playwright.sync_api import Error as PlaywrightError

    console_errors = []
    result = {"html_path": str(html_path.relative_to(REPO_ROOT))}

    browser = playwright_module.chromium.launch(headless=True)
    try:
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda exc: console_errors.append(str(exc)))
        page.add_init_script(RAF_INIT_SCRIPT)
        try:
            page.goto(html_path.as_uri(), wait_until="load", timeout=30000)
        except PlaywrightError as exc:
            result["load_error"] = str(exc)
            return result

        page.wait_for_timeout(5000)
        frames = page.evaluate("window.__frameCount") or 0
        measured_fps = round(frames / 5.0, 1)

        screenshot_path = html_path.with_suffix(".png")
        page.screenshot(path=str(screenshot_path))

        result["console_errors"] = console_errors
        result["measured_frames_5s"] = frames
        result["measured_fps_relative"] = measured_fps
        result["screenshot"] = str(screenshot_path.relative_to(REPO_ROOT))

        if is_pagoda_bench(html_path.stem) or is_pagoda_bench(str(html_path.parent)):
            voxel = page.evaluate("window.__VOXEL__ || null")
            result["voxel_state"] = voxel
            if voxel:
                vc = voxel.get("voxelCount")
                result["constraints"] = {
                    "voxel_count": vc,
                    "voxel_count_ok": bool(vc is not None and VOXEL_COUNT_RANGE[0] <= vc <= VOXEL_COUNT_RANGE[1]),
                    "pagoda_floors": voxel.get("pagodaFloors"),
                    "pagoda_floors_ok": voxel.get("pagodaFloors") == EXPECTED_PAGODA_FLOORS,
                    "reported_fps": voxel.get("fps"),
                    "reported_fps_ok": bool(voxel.get("fps") is not None and voxel.get("fps") > MIN_FPS),
                    "time_of_day": voxel.get("timeOfDay"),
                    # Draw-call count is not exposed by __VOXEL__; this is
                    # informational only, not enforced from the outside.
                    "draw_calls_note": f"not exposed by window.__VOXEL__; spec caps at {MAX_DRAW_CALLS}",
                }
    finally:
        browser.close()

    html_text = html_path.read_text(encoding="utf-8", errors="replace")
    result["file_checks"] = check_single_file(html_path, html_text)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(VOXEL_DOCS_DIR),
                     help="directory tree to search for *.html files")
    ap.add_argument("--out", default=str(VOXEL_DOCS_DIR / "checks.json"))
    args = ap.parse_args(argv)

    root = pathlib.Path(args.root)
    html_files = sorted(root.rglob("*.html"))
    if not html_files:
        print(f"no .html files found under {root}", file=sys.stderr)

    from playwright.sync_api import sync_playwright

    results = []
    with sync_playwright() as p:
        for html_path in html_files:
            print(f"rendering {html_path.relative_to(REPO_ROOT)} ...")
            results.append(render_one(p, html_path))

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"results": results}, indent=1))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
