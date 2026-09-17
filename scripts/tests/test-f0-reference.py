#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import stat
import subprocess
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("f0_reference", REPO / "scripts/f0-reference.py")
assert SPEC and SPEC.loader
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


compile(tool.REMOTE_COLLECTOR, "REMOTE_COLLECTOR", "exec")
assert tool.public_reference_problems() == []
assert 'elif rc in accepted: status="ok"' in tool.REMOTE_COLLECTOR
assert 'unsupported=bool(re.search(br"(?i)(operation not supported|not supported|unknown option|unknown command)",err))' in tool.REMOTE_COLLECTOR
assert '["sudo","-n","devlink","dev","eswitch","show",dev]' in tool.REMOTE_COLLECTOR
normalizer_globals = {"json": json, "re": __import__("re")}
exec(tool.REMOTE_NORMALIZER, normalizer_globals)
stable_stdout = normalizer_globals["stable_stdout"]
firewall_a = b"# Generated at t1\n:INPUT ACCEPT [1:2]\n-A INPUT -m comment --comment keep -j ACCEPT\n# Completed at t1\n"
firewall_b = b"# Generated at t2\n:INPUT ACCEPT [8:9]\n-A INPUT -m comment --comment keep -j ACCEPT\n# Completed at t2\n"
assert stable_stdout("firewall-v4", firewall_a) == stable_stdout("firewall-v4", firewall_b)
assert stable_stdout("firewall-v4", firewall_a) != stable_stdout(
    "firewall-v4", firewall_b.replace(b"INPUT ACCEPT", b"INPUT DROP"))
assert stable_stdout("firewall-v4", firewall_a) != stable_stdout(
    "firewall-v4", firewall_b.replace(b"-A INPUT", b"-A OUTPUT"))
ip_a = json.dumps([{"ifname": "fabric0", "mtu": 9000,
                    "linkinfo": {"info_data": {"gc_timer": 1, "mode": "x"}},
                    "addr_info": [{"local": "10.0.0.1", "valid_life_time": 10,
                                   "preferred_life_time": 5}]}]).encode()
ip_b = ip_a.replace(b'"gc_timer": 1', b'"gc_timer": 2').replace(
    b'"valid_life_time": 10', b'"valid_life_time": 9').replace(
    b'"preferred_life_time": 5', b'"preferred_life_time": 4')
assert stable_stdout("ip-address", ip_a) == stable_stdout("ip-address", ip_b)
assert stable_stdout("ip-address", ip_a) != stable_stdout(
    "ip-address", ip_b.replace(b'"mtu": 9000', b'"mtu": 1500'))
assert "HF_TOKEN" not in tool.rank_payload(0, {
    "container": "c", "launcher": "l", "model_dir": "$HOME/m", "draft_dir": "$HOME/d",
    "model_rev": "r", "nccl_dir": "$HOME/n", "cache_dir": "$HOME/c",
    "extra_docker_env": "-v $HOME/moe-configs/x:/x:ro",
    "fabric_ifaces_0": "f0 f1",
}, 4096, 8192)["safe_env"]
assert 'retain=False' in tool.REMOTE_COLLECTOR


with tempfile.TemporaryDirectory(prefix="f0-reference-test-") as temp:
    root = Path(temp)
    isolated_scripts = root / "sealed/scripts"
    isolated_scripts.mkdir(parents=True)
    for name in ("f0-reference.py", "check-f0.py"):
        (isolated_scripts / name).write_bytes((REPO / "scripts" / name).read_bytes())
    sealed_cli = subprocess.run(["python3", str(isolated_scripts / "f0-reference.py"), "--help"],
                                capture_output=True, text=True, check=False)
    assert sealed_cli.returncode == 0, sealed_cli.stderr
    assert not (isolated_scripts / "__pycache__").exists()

    ordinary = root / "ordinary.env"
    ordinary.write_text("SPEC_TOKENS=5\nBATCHED_TOKENS=8192\n")
    assert tool.safe_config(ordinary).startswith(b"SPEC_TOKENS")
    secret = root / "secret.env"
    secret.write_text("API_TOKEN=do-not-copy\n")
    try:
        tool.safe_config(secret)
    except tool.ReferenceError:
        pass
    else:
        raise AssertionError("credential-like configuration was accepted")

    try:
        tool.outside_repo(REPO / "private-archive")
    except tool.ReferenceError:
        pass
    else:
        raise AssertionError("archive inside checkout was accepted")

    existing = root / "existing"
    existing.mkdir(mode=0o700)
    args = tool.parse_args(["capture", "--archive", str(existing),
                            "--prechange-source", str(root / "unused-source"),
                            "--evidence-dir", str(root / "unused-evidence")])
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        assert tool.capture(args) == 1
    assert "already exists" in output.getvalue()

    source = root / "source"
    source.mkdir(); executable = source / "run.sh"; executable.write_text("#!/bin/sh\n")
    executable.chmod(0o600)
    source_records = [{"path": "run.sh", "type": "file", "mode": 0o755,
                       "uid": os.getuid(), "gid": os.getgid(), "size": executable.stat().st_size,
                       "sha256": tool.sha256_file(executable)}]
    staged = root / "staged"
    tool.materialize_source_tree(source, source_records, staged)
    assert stat.S_IMODE((staged / "run.sh").stat().st_mode) == 0o755
    try:
        tool.materialize_source_tree(source, source_records, staged)
    except tool.ReferenceError:
        pass
    else:
        raise AssertionError("source staging overwrote an existing destination")

    merged = root / "merged.env"
    merged.write_bytes(tool.complete_f0_cluster(b"NODES='a b c d'\n", b"SPEC_TOKENS=5\n",
                                                b"MGMT_IF_BY_RANK=(m m m m)\n"))
    assert subprocess.run(["bash", "-n", str(merged)], check=False).returncode == 0

    archive = root / "archive"
    def put(relative: str, value: object = {}) -> Path:
        path = archive / relative; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n" if not isinstance(value, str) else value)
        path.chmod(0o600); return path

    state = {"status": "COMPLETE", "rank_count": 4, "runtime_signatures": ["same"] * 4,
             "operational_readiness": {"status": "FAIL", "reasons": ["NeedDaemonReload=yes"]}}
    put("archive.json", state); put(tool.ARCHIVE_MANIFEST)
    put("private/site/cluster.env", "NODES='a b c d'\n")
    put("private/site/effective-recipe.json", {"hosts": "n0 n1 n2 n3"})
    put("private/site/f0-site-resolved.env", "MGMT_IF_BY_RANK=(m m m m)\n")
    put("private/site/f0-cluster-resolved.env", "NODES='a b c d'\n")
    put("private/prepared/tp4-autostart.service.d/20-f0-reference.conf", "[Service]\n")
    prechange_file = put("source/pre-change/source/x", "x")
    put("source/pre-change/manifest.json", {"entry_count": 1, "entries": [{
        "path": "x", "type": "file", "size": prechange_file.stat().st_size,
        "sha256": tool.sha256_file(prechange_file), "mode": 0o644}]})
    put("source/pre-change/git-status.txt", " M x\n")
    put("source/pre-change-ownership.json", {"entries": [{"path": "x", "uid": 1, "gid": 1}]})
    completed_file = put("source/completed-iac/x", "x")
    completed = [{"path": "x", "type": "file", "mode": 0o644, "uid": os.getuid(),
                  "gid": os.getgid(), "size": completed_file.stat().st_size,
                  "sha256": tool.sha256_file(completed_file)}]
    put("source/completed-iac-manifest.json", completed)
    put("artifacts/libnccl.so.2", "")
    put("artifacts/nccl-copy.json", {"status": "ok"})
    put("rendered/render.json", {"status": "PASS", "generated": []})
    put("evidence/initial/report.json", {"status": "FAIL", "problems": ["readiness"]})
    for rank in range(4):
        put(f"ranks/rank-{rank}/receipt.json", {"rank": rank, "capture_status": "complete",
            "identity_stable": True, "problems": []})
    tool.secure_tree(archive)
    old_size, old_sha = tool.EXPECTED_NCCL_SIZE, tool.EXPECTED_NCCL_SHA
    tool.EXPECTED_NCCL_SIZE = 0; tool.EXPECTED_NCCL_SHA = tool.sha256_file(archive / "artifacts/libnccl.so.2")
    entries = tool.archive_hashes(archive)
    put(tool.ARCHIVE_MANIFEST, {"schema": 1, "entries": entries})
    archive_problems = tool.validate_archive(archive, rerender=False)
    assert archive_problems == [], archive_problems
    (archive / "archive.json").write_text(json.dumps({**state, "changed": True}))
    assert "archive hash: archive.json" in tool.validate_archive(archive, rerender=False)
    put(tool.ARCHIVE_MANIFEST, {"schema": 1, "entries": [{**entries[0], "path": "../escape"}, *entries[1:]]})
    assert "unsafe SHA manifest path" in tool.validate_archive(archive, rerender=False)
    tool.EXPECTED_NCCL_SIZE, tool.EXPECTED_NCCL_SHA = old_size, old_sha

    live_archive = root / "live"
    (live_archive / "private/site").mkdir(parents=True)
    recipe = {"hosts": "n0 n1 n2 n3"}
    (live_archive / "private/site/effective-recipe.json").write_text(json.dumps(recipe))
    (live_archive / "archive.json").write_text(json.dumps({"runtime_signatures": ["same"] * 4}))
    live_args = tool.parse_args(["verify", "--live", "--archive", str(live_archive)])

    def unreachable(rank, _host, _recipe, _timeout, _command_limit, _total_limit):
        return {"rank": rank, "capture_status": "ssh_error", "problems": ["rank capture SSH failure"]}

    live_problems, _ = tool.live_compare(live_archive, live_args, unreachable)
    assert any("rank capture SSH failure" in item for item in live_problems)

    def changed(rank, _host, _recipe, _timeout, _command_limit, _total_limit):
        return {"rank": rank, "capture_status": "complete", "runtime_signature": "changed", "problems": []}

    changed_problems, _ = tool.live_compare(live_archive, live_args, changed)
    assert sum("runtime/configuration identity changed" in item for item in changed_problems) == 4

    report_parent = root / "reports"
    plan_args = tool.parse_args(["plan-restore", "--archive", str(live_archive),
                                 "--report-root", str(report_parent)])
    with contextlib.redirect_stdout(io.StringIO()):
        assert tool.plan_restore(plan_args) == 1
    reports = list(report_parent.glob("f0-reference-report-*/comparison.json"))
    assert len(reports) == 1
    assert json.loads(reports[0].read_text())["restore_commands_executed"] is False
    assert stat.S_IMODE(reports[0].stat().st_mode) == 0o600

    stage_cli = subprocess.run([
        "python3", str(REPO / "scripts/f0-reference.py"), "stage-source",
        "--archive", str(root / "missing-archive"),
        "--destination", str(root / "unused-stage"),
    ], capture_output=True, text=True, check=False)
    assert stage_cli.returncode == 1, stage_cli.stderr
    assert "capture bounds are too small" not in stage_cli.stderr

with contextlib.redirect_stderr(io.StringIO()):
    try:
        tool.parse_args(["verify", "--archive", "/tmp/a"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("verify accepted neither --offline nor --live")

    try:
        tool.parse_args(["verify", "--offline", "--live", "--archive", "/tmp/a"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("verify accepted both --offline and --live")

print("f0-reference tests: PASS")
