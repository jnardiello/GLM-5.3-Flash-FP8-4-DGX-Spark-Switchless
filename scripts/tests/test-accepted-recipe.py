#!/usr/bin/env python3
"""Verify accepted defaults against the measured overlay, rollback and saved receipts."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import tempfile

REPO = Path(__file__).resolve().parents[2]
REFERENCE = REPO / "docs/historical_benchmarks/baselines/2026-09-28-e31/baseline.json"
record = json.loads(REFERENCE.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
for item in record["provenance"].values():
    assert sha(REPO / item["path"]) == item["sha256"], item["path"]
for path, digest in record["system"]["source_files_sha256"].items():
    assert sha(REPO / path) == digest, path
for item in record["receipts"].values():
    extract = item["portable_extract"]
    assert sha(REPO / extract["path"]) == extract["sha256"]
    data = json.loads((REPO / extract["path"]).read_text())
    assert data["receipt_sha256"] == item["native_receipt_sha256"]
    assert data["request_count"] == data["completed_streams"] == data["visible_responses"] == 54
# E31 was accepted from one suite of the promoted arm (owner screening practice), with the
# same-load E29-equivalent arm as its comparison; the record states n = 1 explicitly.
runs = record["performance"]["included_run_count"]
assert runs == 1 and record["performance"]["accepted_run"] in record["receipts"]
assert record["functional"]["measured_requests"]["count"] == 54 * runs
for row in record["performance"]["metrics"]:
    # A metric may exclude a run by recorded owner decision; the value and reason stay in the record.
    assert len(row["per_run"]) + len(row.get("excluded_per_run", [])) == runs, row["key"]
    reference = row["same_load_reference"]
    assert reference["arm"] == "E" and reference["value"] > 0
    assert abs((row["median"] - reference["value"]) / reference["value"] * 100 - reference["change_pct"]) < 1e-9
    assert all(item["reason"] for item in row.get("excluded_per_run", [])), row["key"]
    assert statistics.median(row["per_run"]) == row["median"], row["key"]

# The current E31-MB reference: two default-arm suites of the E32 series (n = 2). The operational
# identity is pinned at freeze time only, because it follows every template change.
MB = json.loads((REPO / "docs/historical_benchmarks/baselines/2026-09-30-e31-mb/baseline.json").read_text())
assert MB["name"] == "E31-MB" and MB["status"] == "frozen_baseline"
for item in MB["provenance"].values():
    if "sha256" in item:
        assert sha(REPO / item["path"]) == item["sha256"], item["path"]
assert MB["provenance"]["previous_baseline"]["path"] == str(REFERENCE.relative_to(REPO))
for path, digest in MB["system"]["source_files_sha256"].items():
    assert sha(REPO / path) == digest, path
mb_values = {}
for run in MB["performance"]["accepted_runs"]:
    item = MB["receipts"][run]
    extract = item["portable_extract"]
    assert sha(REPO / extract["path"]) == extract["sha256"], run
    data = json.loads((REPO / extract["path"]).read_text())
    assert data["receipt_sha256"] == item["native_receipt_sha256"] and data["arm"] == "A"
    assert data["request_count"] == data["completed_streams"] == data["visible_responses"] == 54
    assert data["prefill_token_matches"] == data["prefill_requests"] == 18
    mb_values[run] = {row["key"]: row["value"] for row in data["metrics"]}
assert MB["performance"]["included_run_count"] == len(mb_values) == 2
assert MB["functional"]["measured_requests"]["count"] == 54 * len(mb_values)
assert {row["key"] for row in MB["performance"]["metrics"]} == {
    row["key"] for row in record["performance"]["metrics"]}
for row in MB["performance"]["metrics"]:
    assert row["per_run"] == [mb_values[run][row["key"]] for run in MB["performance"]["accepted_runs"]]
    assert statistics.median(row["per_run"]) == row["median"], row["key"]

e03 = REPO / "scripts/node/experiments/e03"
previous = (REPO / "scripts/node/reference/baseline-20260919.env").read_text()
e03_rollback = (REPO / "scripts/node/reference/baseline-20260919-e03.env").read_text()
e21_rollback = (REPO / "scripts/node/reference/baseline-20260923-e21.env").read_text()
e22b_rollback = (REPO / "scripts/node/reference/baseline-20260924-e22b.env").read_text()
e27_rollback = (REPO / "scripts/node/reference/baseline-20260924-e27.env").read_text()
e27c_rollback = (REPO / "scripts/node/reference/baseline-20260925-e27c.env").read_text()
e28b_rollback = (REPO / "scripts/node/reference/baseline-20260925-e28b.env").read_text()
e29_rollback = (REPO / "scripts/node/reference/baseline-20260925-e29.env").read_text()
e31_rollback = (REPO / "scripts/node/reference/baseline-20260928-e31.env").read_text()
# The historical E03 measurement: previous base plus the three E03 deltas.
historical_e03 = "\n".join([previous, (e03 / "candidate.env").read_text(),
                            (e03 / "replay-views/delta.env").read_text(),
                            (e03 / "draft-budget/delta.env").read_text()])
# The historical E21 measurement: the complete E03 recipe plus the BF16-residue delta.
historical_e21 = "\n".join([e03_rollback, (e03 / "bf16-residue/delta.env").read_text()])
# The measured E22b load: the complete E21 recipe plus the drafter context-BF16 delta.
historical_e22b = "\n".join([e21_rollback, (e03 / "drafter-w8a16/delta-context-bf16.env").read_text()])
# The measured E27 load: the E22b default plus the native prefill cadence. Its overlay
# (sha256 recorded in the E27 promotion record) added exactly this one engine argument.
historical_e27 = "\n".join([e22b_rollback, 'EXTRA_VLLM_ARGS+=" --prefill-schedule-interval 8"'])
# The measured E27c load: the E27 default plus the queued-cadence overlay.
historical_e27c = "\n".join([e27_rollback, (e03 / "queued-cadence/delta.env").read_text()])
# The measured E28b load: the E28 recipe (E27c plus the draft-depth-7 overlay, promoted
# locally before the window) plus the 16 GiB KV overlay.
historical_e28b = "\n".join([e27c_rollback, (e03 / "draft-depth-7/delta.env").read_text(),
                             (e03 / "kv-16gib/delta.env").read_text()])
# The measured E29 candidate (load B): the complete E28b recipe plus the end-drain overlay.
historical_e29 = "\n".join([e28b_rollback, (e03 / "end-drain/delta-b.env").read_text()])
# The measured E31 load: the complete E29 recipe plus the E31 indexer overlay. The
# protected operational default is that exact engine recipe plus the RAM-budget delta.
historical_e31 = "\n".join([e29_rollback, (e03 / "e31-indexer/delta.env").read_text()])
protected_candidate = "\n".join(
    [historical_e31, (e03 / "sparkcache-ram-budget/delta.env").read_text()])
bounded_candidate = "\n".join([
    protected_candidate,
    (e03 / "bounded-admission/production.env").read_text(),
])
# E31-MB: the memory-bounded command plus the SparkCache disk-capacity pair. The measured E35
# load: E31-MB plus the E35 overlay, whose policy the window wrote at runtime.
e31mb_candidate = "\n".join([bounded_candidate, 'EXTRA_DOCKER_ENV+=" -e SPARK_CONTEXT_CACHE_MAX_BYTES='
                              '214748364800 -e SPARK_CONTEXT_CACHE_LOW_WATERMARK_BYTES=171798691840"'])
e35_measured = "\n".join([e31mb_candidate, (e03 / "e35-runner-k/delta.env").read_text()])
# The E35 default: the measured E35 load plus its read-only policy mount. The measured E36
# load: that default plus the E36 overlay, which the current default must reproduce exactly.
e35_default = "\n".join([e35_measured, "EXTRA_DOCKER_ENV+=' -v $HOME/tp4/experiments/e03/e35-runner-k/"
                          "policy.flag:/tmp/glm53-e35-policy:ro'"])
e36_measured = "\n".join([e35_default, (e03 / "e36-lm-head-w8a16/delta.env").read_text()])
default_identity = json.loads((
    REPO / "docs/operational-identities/2026-09-30-e36-lm-head.json").read_text())
DISK_WORDS = ["-e", "SPARK_CONTEXT_CACHE_MAX_BYTES=214748364800",
              "-e", "SPARK_CONTEXT_CACHE_LOW_WATERMARK_BYTES=171798691840"]
POLICY_WORDS = ["-v", str(Path.home()) + "/tp4/experiments/e03/e35-runner-k/policy.flag:/tmp/glm53-e35-policy:ro"]
NVIDIA = "/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/"
E31_ADDED = [str(Path.home()) + "/tp4/experiments/e03/e31-indexer/pooled_indexer.py:" + NVIDIA + "pooled_indexer.py:ro",
             str(Path.home()) + "/tp4/experiments/e03/e31-indexer/glm_kpool.py:" + NVIDIA + "ops/glm_kpool.py:ro",
             "-e", "VLLM_GLM53_INDEXER_GATE_TC_FLAG=/tmp/glm53-indexer-gate-tc",
             "-e", "VLLM_GLM53_KPOOL_TAIL_RING_FLAG=/tmp/glm53-kpool-tail-ring"]
E31_REMOVED = [str(Path.home()) + "/tp4/overrides/vllm/models/glm5next/nvidia/pooled_indexer.py:" + NVIDIA + "pooled_indexer.py:ro",
               str(Path.home()) + "/tp4/overrides/vllm/models/glm5next/nvidia/ops/glm_kpool.py:" + NVIDIA + "ops/glm_kpool.py:ro"]
SCHED = "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py:ro"
E29_ADDED = [str(Path.home()) + "/tp4/experiments/e03/end-drain/scheduler.py:" + SCHED,
             "-v", str(Path.home()) + "/tp4/experiments/e03/end-drain/core.py:"
             "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py:ro",
             "-e", "VLLM_E29_END_DRAIN=1", "-e", "VLLM_E29_IDLE_COALESCE_MS=4", "-e", "VLLM_E29_TRACE=0"]
E29_REMOVED = [str(Path.home()) + "/tp4/experiments/e03/queued-cadence/scheduler.py:" + SCHED]
E28B_ADDED = ['{"method":"dflash","model":"/draft","num_speculative_tokens":7,'
              '"num_speculative_tokens_per_batch_size":[[1,1,7],[2,6,3]],"kv_cache_dtype":"fp8_e4m3"}',
              "-e", "VLLM_ADAPTIVE_K_HI=7", '--compilation-config={"max_cudagraph_capture_size":72}',
              "--kv-cache-memory-bytes=17179869184"]
E28B_REMOVED = ['{"method":"dflash","model":"/draft","num_speculative_tokens":5,'
                '"num_speculative_tokens_per_batch_size":[[1,1,5],[2,6,3]],"kv_cache_dtype":"fp8_e4m3"}',
                "--kv-cache-memory-bytes=16106127360"]
E27C_TOKENS = ["-v", str(Path.home()) + "/tp4/experiments/e03/queued-cadence/scheduler.py:"
               "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py:ro",
               "-e", "VLLM_E27B_SHORT_PREFILL_TOKENS=2048", "-e", "VLLM_E27C_CADENCE_WHEN_QUEUED=1"]
with tempfile.TemporaryDirectory(prefix="tp4-accepted-recipe-") as temp:
    root = Path(temp)
    shutil.copyfile(REPO / "scripts/launcher/launch-glm53-tp4.sh", root / "launch.sh")
    config = (REPO / "cluster.env.example").read_text() + '''
NODES="n0 n1 n2 n3"
MGMT_IPS="192.0.2.21 192.0.2.22 192.0.2.23 192.0.2.24"
MASTER_IP=192.0.2.21
RELAY_DEST=operator@192.0.2.23
'''
    (root / "cluster.env").write_text(config)
    (root / "protected.env").write_text(protected_candidate)
    (root / "bounded.env").write_text(bounded_candidate)
    (root / "e31mb.env").write_text(e31mb_candidate)
    (root / "e35-measured.env").write_text(e35_measured)
    (root / "e35-default.env").write_text(e35_default)
    (root / "e36-measured.env").write_text(e36_measured)
    (root / "operational-e35.env").write_text((
        REPO / "scripts/node/reference/operational-20260930-e35.env").read_text())
    (root / "operational-e31-mb.env").write_text((
        REPO / "scripts/node/reference/operational-20260930-e31-mb.env").read_text())
    (root / "operational-memory-bounded.env").write_text((
        REPO / "scripts/node/reference/operational-20260929-memory-bounded.env").read_text())
    (root / "operational-protected.env").write_text((
        REPO / "scripts/node/reference/operational-20260929-sparkcache-protected.env").read_text())
    (root / "rollback.env").write_text(previous)
    (root / "rollback-e03.env").write_text(e03_rollback)
    (root / "rollback-e21.env").write_text(e21_rollback)
    (root / "rollback-e22b.env").write_text(e22b_rollback)
    (root / "rollback-e27.env").write_text(e27_rollback)
    (root / "rollback-e27c.env").write_text(e27c_rollback)
    (root / "rollback-e28b.env").write_text(e28b_rollback)
    (root / "rollback-e29.env").write_text(e29_rollback)
    (root / "rollback-e31.env").write_text(e31_rollback)
    (root / "historical-e31.env").write_text(historical_e31)
    (root / "historical-e29.env").write_text(historical_e29)
    (root / "historical-e28b.env").write_text(historical_e28b)
    (root / "historical-e27c.env").write_text(historical_e27c)
    (root / "historical-e27.env").write_text(historical_e27)
    (root / "historical-e22b.env").write_text(historical_e22b)
    (root / "historical-e21.env").write_text(historical_e21)
    (root / "historical-e03.env").write_text(historical_e03)
    env = dict(os.environ, TP4_DRY_RUN="1")
    env.pop("TP4_ENV", None)
    forbidden = root / "forbidden.log"
    bindir = root / "bin"
    bindir.mkdir()
    for name in ("sudo", "docker", "ssh", "systemctl", "curl", "ip", "sysctl"):
        path = bindir / name
        path.write_text('#!/bin/sh\nprintf "%s\\n" "$0" >> "$TP4_FORBIDDEN"\nexit 97\n')
        path.chmod(0o700)
    env.update(PATH=str(bindir) + os.pathsep + env["PATH"], TP4_FORBIDDEN=str(forbidden))

    def launch(rank, overlay=None):
        selected = dict(env, TP4_ENV=overlay) if overlay else env
        result = subprocess.run(["bash", str(root / "launch.sh"), str(rank)],
                                env=selected, capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stderr
        assert not forbidden.exists(), "Dry-run attempted an external action"
        return [line[2:] for line in result.stdout.splitlines() if line.startswith("  ")]

    for rank in range(4):
        current = launch(rank)
        # The default is the measured E35 command plus only the read-only policy mount.
        # The default is exactly the measured E36 command; the one-step return reaches E35, which
        # is the measured E35 command plus only the read-only policy mount.
        assert current == launch(rank, "e36-measured.env"), f"rank {rank}: E36 default drifted"
        e35 = launch(rank, "e35-default.env")
        assert launch(rank, "operational-e35.env") == e35, f"rank {rank}: E35 return drifted"
        measured = launch(rank, "e35-measured.env")
        start = next(i for i in range(len(e35)) if e35[i:i + 2] == POLICY_WORDS)
        assert e35[:start] + e35[start + 2:] == measured, f"rank {rank}: E35 default drifted"
        # E31-MB is the memory-bounded command plus the SparkCache disk-capacity pair; the
        # one-step return reaches it, and the memory-bounded return removes E35 and the pair.
        e31mb = launch(rank, "e31mb.env")
        bounded = launch(rank, "bounded.env")
        assert launch(rank, "operational-e31-mb.env") == e31mb, f"rank {rank}: E31-MB return drifted"
        assert launch(rank, "operational-memory-bounded.env") == bounded, (
            f"rank {rank}: memory-bounded return drifted")
        start = next(i for i in range(len(e31mb)) if e31mb[i:i + 4] == DISK_WORDS)
        assert e31mb[:start] + e31mb[start + 4:] == bounded, f"rank {rank}: bounded default drifted"
        mounts = dict(arg.split(":")[1::-1] for i, arg in enumerate(current) if i and current[i-1] == "-v")
        mount_count = Counter(arg.split(":")[1] for i, arg in enumerate(current) if i and current[i-1] == "-v")
        assert all(n == 1 for n in mount_count.values()), "Duplicate mount target"
        payloads = record["system"]["payload_packaging"]["operator_payloads"]
        private_targets = {item["container_path"] for item in payloads.values()}
        runtime_hashes = dict(record["system"]["operational_identity"]["container_file_sha256"])
        runtime_hashes.update(default_identity["runtime_identity_overrides"]["container_file_sha256"])
        for target, digest in runtime_hashes.items():
            assert target in mounts, target
            if target in private_targets:
                continue
            source = mounts[target]
            assert source.startswith(str(Path.home()) + "/tp4/")
            local = REPO / "scripts/node" / source.split("/tp4/", 1)[1]
            assert sha(local) == digest, local
        # Immediate operational rollback: protected SparkCache with the 16 GiB E31 engine.
        protected_restored = launch(rank, "operational-protected.env")
        assert protected_restored == launch(rank, "protected.env"), (
            f"rank {rank}: protected operational rollback drifted")
        assert "--kv-cache-memory-bytes=17179869184" in protected_restored
        assert "--middleware" not in protected_restored
        # Historical rollback: exactly the measured E31 command and previous cache behavior.
        e31_restored = launch(rank, "rollback-e31.env")
        assert e31_restored == launch(rank, "historical-e31.env"), f"rank {rank}: E31 rollback drifted"
        changes = [(old, new) for old, new in zip(e31_restored, protected_restored) if old != new]
        assert len(e31_restored) == len(protected_restored) and len(changes) == 2, (rank, changes)
        assert any("spark_context_cache_connector-ram-budget.py" in new for old, new in changes)
        assert any("sparkcache-ram-budget/kv-transfer-config.json" in new for old, new in changes)
        # Older E29 rollback: exactly the measured E29 command, with the production indexer.
        e29_restored = launch(rank, "rollback-e29.env")
        assert e29_restored == launch(rank, "historical-e29.env"), f"rank {rank}: E29 rollback drifted"
        assert Counter(e31_restored) - Counter(e29_restored) == Counter(E31_ADDED), f"rank {rank}: E31 delta"
        assert Counter(e29_restored) - Counter(e31_restored) == Counter(E31_REMOVED), f"rank {rank}: E31 removed"
        # Older E28b return: exactly the measured E28b command, with the E27c scheduler.
        e28b_restored = launch(rank, "rollback-e28b.env")
        assert e28b_restored == launch(rank, "historical-e28b.env"), f"rank {rank}: E28b rollback drifted"
        assert Counter(e29_restored) - Counter(e28b_restored) == Counter(E29_ADDED), f"rank {rank}: E29 delta"
        assert Counter(e28b_restored) - Counter(e29_restored) == Counter(E29_REMOVED), f"rank {rank}: E29 removed"
        # Older E27c return: exactly the measured E27c command, five draft tokens and 15 GiB.
        e27c_restored = launch(rank, "rollback-e27c.env")
        assert e27c_restored == launch(rank, "historical-e27c.env"), f"rank {rank}: E27c rollback drifted"
        assert Counter(e28b_restored) - Counter(e27c_restored) == Counter(E28B_ADDED), f"rank {rank}: E28b delta"
        assert Counter(e27c_restored) - Counter(e28b_restored) == Counter(E28B_REMOVED), f"rank {rank}: E28b removed"
        # Older E27 return: exactly the measured E27 command, without the E27c scheduler.
        e27_restored = launch(rank, "rollback-e27.env")
        assert e27_restored == launch(rank, "historical-e27.env"), f"rank {rank}: E27 rollback drifted"
        assert not any("queued-cadence" in item or item.startswith("VLLM_E27") for item in e27_restored)
        assert Counter(e27c_restored) - Counter(e27_restored) == Counter(E27C_TOKENS), f"rank {rank}: E27c delta"
        assert Counter(e27_restored) - Counter(e27c_restored) == Counter(), f"rank {rank}: E27c removed items"
        # Older E22b return: exactly the measured E22b command, without the E27 cadence.
        e22b_restored = launch(rank, "rollback-e22b.env")
        assert e22b_restored == launch(rank, "historical-e22b.env"), f"rank {rank}: E22b rollback drifted"
        assert "--prefill-schedule-interval" not in e22b_restored
        assert e27_restored == e22b_restored + ["--prefill-schedule-interval", "8"], f"rank {rank}: E27 delta"
        # Older E21 return: exactly the measured E21 command, without any E22 element.
        e21_restored = launch(rank, "rollback-e21.env")
        assert e21_restored == launch(rank, "historical-e21.env"), f"rank {rank}: E21 rollback drifted"
        assert not any("drafter-w8a16" in item or "qwen3_dflash2" in item for item in e21_restored)
        assert not any(item.startswith("VLLM_E22_") for item in e21_restored)
        assert "VLLM_E21_BF16_RESIDUE_W8A16=1" in e21_restored
        assert "VLLM_E22_DRAFTER_W8A16=1" in current and "VLLM_E22_CONTEXT_KV_W8A16=0" in current
        # Older E03 return: exactly the measured E03 command, without any E21 element.
        e03_restored = launch(rank, "rollback-e03.env")
        assert e03_restored == launch(rank, "historical-e03.env"), f"rank {rank}: E03 rollback drifted"
        assert not any("e21_bf16_residue" in item for item in e03_restored)
        assert "VLLM_E21_BF16_RESIDUE_W8A16=1" not in e03_restored
        assert any(item.endswith("/tp4/overrides/vllm/models/glm5next/nvidia/e20_kda_w8a16.py:"
                                 "/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/"
                                 "e20_kda_w8a16.py:ro") for item in e03_restored)
        assert "VLLM_E21_BF16_RESIDUE_W8A16=1" in current
        # Older pre-E03 base remains a complete return as well.
        restored = launch(rank, "rollback.env")
        assert "SPARK_MHC_PREFILL_SHARD=0" in restored
        assert "VLLM_ADAPTIVE_K_RESPECT_DRAFT_BUDGET=1" not in restored
        assert not any("/experiments/e03/" in item for item in restored)
        assert not any("connector-e03-replay-views" in item for item in restored)
        assert "--kv-cache-memory-bytes=16106127360" in restored

    # Both returns refuse any base without exactly the default E35 selection (and, for the
    # memory-bounded return, the default capacity pair).
    e35_changes = (
        ("e35-twice", 'EXTRA_DOCKER_ENV+=" -e VLLM_E35_ENABLE=1"\n'),
        ("e35-policy", "EXTRA_DOCKER_ENV=${EXTRA_DOCKER_ENV/policy.flag/other.flag}\n"),
        ("e35-scheduler", "EXTRA_DOCKER_ENV=${EXTRA_DOCKER_ENV/e35-runner-k\\/adaptive/draft-budget\\/adaptive}\n"),
        ("renamed-e36-module", 'EXTRA_DOCKER_ENV+=" -v /tmp/renamed.py:/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/e36_lm_head_w8a16.py:ro"\n'),
        ("env-file", 'EXTRA_DOCKER_ENV+=" --env-file /x.env"\n'),
        ("volume-form", 'EXTRA_DOCKER_ENV+=" --volume=/a:/b"\n'),
        ("second-scheduler", 'EXTRA_DOCKER_ENV+=" -v /other:/opt/tp4/adaptive_k_scheduler.py:ro"\n'),
    )
    e36_changes = (
        ("e36-twice", 'EXTRA_DOCKER_ENV+=" -e VLLM_E36_KEEP_BF16=0"\n'),
        ("e36-module", "EXTRA_DOCKER_ENV=${EXTRA_DOCKER_ENV/e36_lm_head_w8a16.py:ro/e36_other.py:ro}\n"),
        ("env-file", 'EXTRA_DOCKER_ENV+=" --env-file /x.env"\n'),
        ("second-runner", 'EXTRA_DOCKER_ENV+=" -v /o:/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/model_runner.py:ro"\n'),
    )
    for reference, message, cases in (
        ("operational-20260930-e35.env",
         "requires exactly the default E36 runner, module and variables", e36_changes),
        ("operational-20260929-memory-bounded.env",
         "requires exactly the default E35 selection and SparkCache disk-capacity pair", (
             ("changed-value", "EXTRA_DOCKER_ENV=${EXTRA_DOCKER_ENV/MAX_BYTES=214748364800/MAX_BYTES=1}\n"),
             ("duplicate-pair", 'EXTRA_DOCKER_ENV+=" ' + " ".join(DISK_WORDS) + '"\n'),
             ("ttl", 'EXTRA_DOCKER_ENV+=" -e SPARK_CONTEXT_CACHE_TTL_SECONDS=60"\n'), *e35_changes)),
        ("operational-20260930-e31-mb.env",
         "requires exactly the default E35 scheduler, runner, speculator and policy selection",
         e35_changes),
    ):
        return_body = (REPO / "scripts/node/reference" / reference).read_text()
        for name, prefix in (("applied-twice", return_body + "\n"), *cases):
            (root / f"return-{name}.env").write_text(prefix + return_body)
            refused = subprocess.run(["bash", str(root / "launch.sh"), "0"],
                                     env=dict(env, TP4_ENV=f"return-{name}.env"),
                                     capture_output=True, text=True, timeout=10)
            assert refused.returncode != 0, (reference, name)
            assert message in refused.stderr, (reference, name, refused.stderr)
            assert not forbidden.exists(), "Dry-run attempted an external action"

print("test-accepted-recipe: PASS (four-rank E36 default = measured E36; E35 = measured E35 + policy mount; E31-MB, memory-bounded, protected16, E31, E29, E28b, E27c, E27, E22b, E21, E03 and pre-E03 rollbacks)")
