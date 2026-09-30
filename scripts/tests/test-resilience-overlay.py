#!/usr/bin/env python3
"""Offline tests for generation of the private resilience overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/resilience"))
import common  # noqa: E402
import prepare_overlay as prepare  # noqa: E402


BASE_CONFIG = REPO / "scripts/node/experiments/e03/sparkcache-ram-budget/kv-transfer-config.json"
BASE_CONNECTOR = REPO / "third_party/sparkcache/spark_context_cache_connector-ram-budget.py"


class OverlayTests(unittest.TestCase):
    def test_generated_delta_preserves_base_bytes_and_changes_namespace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts/resilience/.campaign").mkdir(parents=True)
            args = argparse.Namespace(campaign_id="unit-overlay", base_config=str(BASE_CONFIG),
                                      base_connector=str(BASE_CONNECTOR), force=False)
            with patch.object(prepare, "REPO", root):
                output = prepare.prepare(args)
            base = json.loads(BASE_CONFIG.read_text())
            generated = json.loads((output / "kv-transfer-config.json").read_text())
            expected = json.loads(json.dumps(base))
            expected["kv_connector_extra_config"]["spark_cache_root"] = (
                "/cache/jit/tp4-resilience-unit-overlay"
            )
            self.assertEqual(generated, expected)
            delta = (output / "delta.env").read_text()
            self.assertIn(common.sha256_file(BASE_CONNECTOR), delta)
            self.assertIn("TP4_RESILIENCE_MAX_BYTES=8589934592", delta)
            self.assertIn("SPARK_CONTEXT_CACHE_MAX_BYTES=3221225472", delta)
            self.assertIn("SPARK_CONTEXT_CACHE_LOW_WATERMARK_BYTES=2147483648", delta)
            self.assertNotIn(str(BASE_CONNECTOR), delta)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertNotIn("bounded_admission", manifest)
            self.assertFalse((output / prepare.ADMISSION_STAGED_NAME).exists())
            self.assertNotIn(prepare.ADMISSION_IMPORT, delta)
            self.assertEqual((output / "manifest.json").stat().st_mode & 0o077, 0)

    def test_requested_variants_are_composed_deterministically_and_manifest_pinned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts/resilience/.campaign").mkdir(parents=True)
            trim_source = root / "scripts/node/experiments/e03/prefill-cache-trim/delta.env"
            step_source = root / "scripts/node/experiments/e03/prefill-step-cap/delta.env"
            trim_source.parent.mkdir(parents=True)
            step_source.parent.mkdir(parents=True)
            trim_source.write_text("# trim component\nTRIM=1\n")
            step_source.write_text("# step component\nCAP=6912\n")

            def generate(campaign_id, variant, force=False):
                args = argparse.Namespace(
                    campaign_id=campaign_id, base_config=str(BASE_CONFIG),
                    base_connector=str(BASE_CONNECTOR), runtime_variant=variant,
                    force=force)
                with patch.object(prepare, "REPO", root):
                    return prepare.prepare(args)

            trim = generate("unit-trim", "prefill-cache-trim")
            trim_bytes = (trim / "delta-prefill-trim.env").read_bytes()
            self.assertEqual(trim_bytes,
                             (trim / "delta.env").read_bytes() + trim_source.read_bytes())
            trim_manifest = json.loads((trim / "manifest.json").read_text())
            self.assertEqual(trim_manifest["runtime_variant"], "prefill-cache-trim")
            self.assertEqual(trim_manifest["selected_delta"], "delta-prefill-trim.env")
            self.assertEqual(trim_manifest["files"]["delta-prefill-trim.env"],
                             hashlib.sha256(trim_bytes).hexdigest())

            first_manifest = (trim / "manifest.json").read_bytes()
            generate("unit-trim", "prefill-cache-trim", force=True)
            self.assertEqual((trim / "delta-prefill-trim.env").read_bytes(), trim_bytes)
            self.assertEqual((trim / "manifest.json").read_bytes(), first_manifest)

            step = generate("unit-step", "prefill-step-cap")
            step_bytes = (step / "delta-prefill-step-cap.env").read_bytes()
            self.assertEqual(step_bytes, (step / "delta.env").read_bytes()
                             + trim_source.read_bytes() + step_source.read_bytes())
            step_manifest = json.loads((step / "manifest.json").read_text())
            self.assertEqual(step_manifest["runtime_variant"], "prefill-step-cap")
            self.assertEqual([item["path"] for item in step_manifest["variant_components"]], [
                "scripts/node/experiments/e03/prefill-cache-trim/delta.env",
                "scripts/node/experiments/e03/prefill-step-cap/delta.env",
            ])
            self.assertEqual((step / "delta-prefill-step-cap.env").stat().st_mode & 0o077, 0)

    def test_optional_kv_override_changes_only_four_rank_kv_argument(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts/resilience/.campaign").mkdir(parents=True)
            for relative in (
                "scripts/node/experiments/e03/prefill-cache-trim/delta.env",
                "scripts/node/experiments/e03/prefill-step-cap/delta.env",
            ):
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(REPO / relative, destination)
            cluster = root / "cluster.env"
            cluster.write_text(
                (REPO / "cluster.env.example").read_text()
                + "\n"
                + (REPO / "scripts/node/reference/operational-20260929-sparkcache-protected.env").read_text()
                + '\nNODES="n0 n1 n2 n3"\n'
                + 'MGMT_IPS="192.0.2.21 192.0.2.22 192.0.2.23 192.0.2.24"\n'
                + 'MASTER_IP=192.0.2.21\nRELAY_DEST=operator@192.0.2.23\n')
            launcher = root / "launch-glm53-tp4.sh"
            shutil.copyfile(REPO / "scripts/launcher/launch-glm53-tp4.sh", launcher)
            campaign_id = "unit-kv14"
            relative_delta = (
                f"scripts/resilience/.campaign/{campaign_id}/delta-prefill-step-cap.env"
            )

            def generate(kv_bytes=Ellipsis, force=False):
                values = dict(
                    campaign_id=campaign_id, base_config=str(BASE_CONFIG),
                    base_connector=str(BASE_CONNECTOR), runtime_variant="prefill-step-cap",
                    force=force)
                if kv_bytes is not Ellipsis:
                    values["kv_cache_memory_bytes"] = kv_bytes
                with patch.object(prepare, "REPO", root):
                    return prepare.prepare(argparse.Namespace(**values))

            def dry_runs():
                outputs = []
                for rank in range(4):
                    completed = __import__("subprocess").run(
                        ["bash", str(launcher), str(rank)],
                        env={**os.environ, "HOME": str(root / "node-home"),
                             "TP4_DRY_RUN": "1", "TP4_ENV": relative_delta},
                        capture_output=True, text=True)
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    outputs.append(completed.stdout.splitlines()[-1])
                return outputs

            output = generate()
            selected = output / "delta-prefill-step-cap.env"
            baseline_payload = selected.read_bytes()
            baseline_manifest = json.loads((output / "manifest.json").read_text())
            self.assertNotIn("kv_cache_memory_bytes", baseline_manifest)
            baseline_commands = dry_runs()

            output = generate(None, force=True)
            self.assertEqual(selected.read_bytes(), baseline_payload)
            self.assertEqual(json.loads((output / "manifest.json").read_text()),
                             baseline_manifest)

            output = generate(14 << 30, force=True)
            candidate_payload = selected.read_bytes()
            candidate_manifest = json.loads((output / "manifest.json").read_text())
            self.assertTrue(candidate_payload.startswith(baseline_payload))
            self.assertEqual(candidate_manifest["kv_cache_memory_bytes"], 14 << 30)
            for name, digest in baseline_manifest["files"].items():
                if name != "delta-prefill-step-cap.env":
                    self.assertEqual(candidate_manifest["files"][name], digest)
            candidate_commands = dry_runs()
            old = "--kv-cache-memory-bytes=17179869184"
            new = "--kv-cache-memory-bytes=15032385536"
            for baseline, candidate in zip(baseline_commands, candidate_commands):
                self.assertEqual(candidate.count(new), 1)
                self.assertEqual(baseline.count(old), 1)
                self.assertEqual(candidate.replace(new, old), baseline)
                self.assertIn("--max-model-len 262144", candidate)
                self.assertIn("--max-num-seqs 6", candidate)

    def test_kv_override_rejects_bad_values_and_ambiguous_arguments(self):
        for value in (True, False, 0, -1, 1.0, "0", "not-an-integer"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                prepare.positive_int(value)
        self.assertEqual(prepare.positive_int("15032385536"), 14 << 30)
        parsed = prepare.parser().parse_args([
            "--campaign-id", "unit-cli", "--base-config", str(BASE_CONFIG),
            "--base-connector", str(BASE_CONNECTOR),
            "--kv-cache-memory-bytes=15032385536",
        ])
        self.assertEqual(parsed.kv_cache_memory_bytes, 14 << 30)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kv.env"
            path.write_text(prepare.kv_cache_override_text(14 << 30))

            def source(arguments):
                return __import__("subprocess").run(
                    ["bash", "-c", 'set -e; source "$1"; printf "%s\\n" "$EXTRA_VLLM_ARGS"',
                     "kv-test", str(path)],
                    env={**os.environ, "EXTRA_VLLM_ARGS": arguments},
                    capture_output=True, text=True)

            original = ("--kv-cache-memory-bytes=17179869184 --attention-backend B12X "
                        "--max-logprobs 100")
            completed = source(original)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stdout.strip(), original.replace(
                "--kv-cache-memory-bytes=17179869184",
                "--kv-cache-memory-bytes=15032385536"))
            for ambiguous in (
                "--attention-backend B12X",
                "--kv-cache-memory-bytes=15032385536 --attention-backend B12X",
                "--kv-cache-memory-bytes=17179869184 --kv-cache-memory-bytes=17179869184",
                "--kv-cache-memory-bytes=17179869184 --kv-cache-memory=17179869184",
                "--kv-cache-memory-bytes=17179869184 --kv-cache-memory 15032385536",
                "--kv-cache-memory-bytes=17179869184 --kv-cache-memory-bytes 15032385536",
            ):
                with self.subTest(arguments=ambiguous):
                    completed = source(ambiguous)
                    self.assertNotEqual(completed.returncode, 0)

            args = argparse.Namespace(
                campaign_id="unit-invalid", base_config=str(BASE_CONFIG),
                base_connector=str(BASE_CONNECTOR), runtime_variant="default", force=False,
                kv_cache_memory_bytes=True)
            with patch.object(prepare, "REPO", Path(directory)), \
                    self.assertRaises(common.ContractError):
                prepare.prepare(args)

    def test_optional_bounded_admission_is_pinned_staged_and_composed_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts/resilience/.campaign").mkdir(parents=True)
            source = root / prepare.ADMISSION_MANIFEST
            source.parent.mkdir(parents=True)
            middleware = source.parent / "middleware.py"
            middleware.write_text("# bounded admission unit candidate\n")
            digest = hashlib.sha256(middleware.read_bytes()).hexdigest()
            source.write_text(json.dumps({
                "schema": "tp4-bounded-admission-candidate-v1",
                "candidate": {
                    "path": str(middleware.relative_to(root)),
                    "sha256": digest,
                    "mount_target": prepare.ADMISSION_TARGET,
                    "import_string": prepare.ADMISSION_IMPORT,
                },
                "selection": {"environment": prepare.ADMISSION_ENVIRONMENT},
                "guard": {
                    "mode": prepare.ADMISSION_GUARD_MODE,
                    "bypass_paths": sorted(prepare.ADMISSION_BYPASS_PATHS),
                    "bypass_pattern": prepare.ADMISSION_BYPASS_PATTERN,
                },
            }))
            args = argparse.Namespace(
                campaign_id="unit-admission", base_config=str(BASE_CONFIG),
                base_connector=str(BASE_CONNECTOR), runtime_variant="default",
                bounded_admission=True, force=False)
            with patch.object(prepare, "REPO", root):
                output = prepare.prepare(args)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["bounded_admission"]["source_sha256"], digest)
            self.assertEqual(manifest["files"][prepare.ADMISSION_STAGED_NAME], digest)
            self.assertEqual((output / prepare.ADMISSION_STAGED_NAME).read_bytes(),
                             middleware.read_bytes())
            self.assertEqual((output / prepare.ADMISSION_STAGED_NAME).stat().st_mode & 0o077,
                             0)
            delta = (output / "delta.env").read_text()
            self.assertIn(prepare.ADMISSION_IMPORT, delta)
            self.assertIn(prepare.ADMISSION_TARGET, delta)

            fragment = root / "admission.env"
            fragment.write_text(prepare.admission_override_text("unit-admission"))

            def source_fragment(docker: str, vllm: str):
                return __import__("subprocess").run(
                    ["bash", "-c",
                     'set -e; source "$1"; printf "%s\\n%s\\n" '
                     '"$EXTRA_DOCKER_ENV" "$EXTRA_VLLM_ARGS"',
                     "admission-test", str(fragment)],
                    env={**os.environ, "EXTRA_DOCKER_ENV": docker,
                         "EXTRA_VLLM_ARGS": vllm}, capture_output=True, text=True)

            completed = source_fragment("-v /tmp/base:/opt/base:ro -e EXISTING=1",
                                        "--max-model-len 262144")
            self.assertEqual(completed.returncode, 0, completed.stderr)
            docker, vllm = completed.stdout.splitlines()
            self.assertIn("-v /tmp/base:/opt/base:ro", docker)
            self.assertIn(f":{prepare.ADMISSION_TARGET}:ro", docker)
            for key, value in prepare.ADMISSION_ENVIRONMENT.items():
                self.assertEqual(docker.split().count(f"{key}={value}"), 1)
            self.assertEqual(vllm.split().count("--middleware"), 1)
            self.assertEqual(vllm.split().count(prepare.ADMISSION_IMPORT), 1)

            conflicts = [
                (f"-v /tmp/x:{prepare.ADMISSION_TARGET}:ro", ""),
                (f"-e {next(iter(prepare.ADMISSION_ENVIRONMENT))}=1", ""),
                ("", f"--middleware={prepare.ADMISSION_IMPORT}"),
                ("", f"--middleware {prepare.ADMISSION_IMPORT}"),
            ]
            for docker, vllm in conflicts:
                with self.subTest(docker=docker, vllm=vllm):
                    self.assertNotEqual(source_fragment(docker, vllm).returncode, 0)

            parsed = prepare.parser().parse_args([
                "--campaign-id", "unit-cli", "--base-config", str(BASE_CONFIG),
                "--base-connector", str(BASE_CONNECTOR), "--bounded-admission"])
            self.assertTrue(parsed.bounded_admission)
            bad_manifest = json.loads(source.read_text())
            bad_manifest["guard"]["mode"] = "named_paths"
            source.write_text(json.dumps(bad_manifest))
            args.force = True
            with patch.object(prepare, "REPO", root), self.assertRaisesRegex(
                    common.ContractError, "unexpected contract"):
                prepare.prepare(args)

    def test_variant_generation_refuses_missing_components_and_unknown_variant(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts/resilience/.campaign").mkdir(parents=True)
            args = argparse.Namespace(
                campaign_id="unit-missing", base_config=str(BASE_CONFIG),
                base_connector=str(BASE_CONNECTOR), runtime_variant="prefill-cache-trim",
                force=False)
            with patch.object(prepare, "REPO", root), self.assertRaises(common.ContractError):
                prepare.prepare(args)
            args.runtime_variant = "made-up"
            with patch.object(prepare, "REPO", root), self.assertRaises(common.ContractError):
                prepare.prepare(args)

    def test_base_must_be_bounded_recompute_config(self):
        value = json.loads(BASE_CONFIG.read_text())
        value["kv_connector_extra_config"]["spark_cache_cpu_budget_bytes"] = 0
        with self.assertRaises(common.ContractError):
            prepare.validate_base_config(value)
        for key in ("spark_cache_max_bytes", "spark_cache_low_watermark_bytes"):
            with self.subTest(key=key):
                value = json.loads(BASE_CONFIG.read_text())
                value["kv_connector_extra_config"][key] = 1
                with self.assertRaises(common.ContractError):
                    prepare.validate_base_config(value)

    def test_wrapper_overrides_only_the_intended_real_connector_hooks(self):
        prepare.validate_wrapper_overrides(BASE_CONNECTOR)
        with tempfile.TemporaryDirectory() as directory:
            wrapper = Path(directory) / "connector_wrapper.py"
            source = (REPO / "scripts/resilience/connector_wrapper.py").read_text()
            altered = source.replace(
                "additional_digests=(), error=None):", "error=None):", 1)
            self.assertNotEqual(altered, source)
            wrapper.write_text(altered)
            with self.assertRaises(common.ContractError):
                prepare.validate_wrapper_overrides(BASE_CONNECTOR, wrapper)
        value = json.loads(BASE_CONFIG.read_text())
        value["kv_load_failure_policy"] = "fail"
        with self.assertRaises(common.ContractError):
            prepare.validate_base_config(value)

    def test_overlay_shell_has_balanced_syntax(self):
        text = prepare.overlay_text("unit-overlay", "a" * 64, "b" * 64, "c" * 64)
        with tempfile.NamedTemporaryFile("w", suffix=".env") as stream:
            stream.write(text); stream.flush()
            completed = __import__("subprocess").run(["bash", "-n", stream.name])
        self.assertEqual(completed.returncode, 0)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "delta.env"
            base.write_text(text)
            for variant in ("prefill-cache-trim", "prefill-step-cap"):
                _, components = prepare.RUNTIME_VARIANTS[variant]
                payload, _ = prepare.compose_variant(base, components)
                composed = Path(directory) / f"{variant}.env"
                composed.write_bytes(payload)
                completed = __import__("subprocess").run(
                    ["bash", "-n", str(composed)], capture_output=True, text=True)
                self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_runtime_mount_keeps_home_literal_until_native_launcher(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            overlay = root / "scripts/resilience/.campaign/unit-overlay/delta.env"
            overlay.parent.mkdir(parents=True)
            overlay.write_text(prepare.overlay_text(
                "unit-overlay", "a" * 64, "b" * 64, "c" * 64))
            cluster = root / "cluster.env"
            cluster.write_text(
                (REPO / "cluster.env.example").read_text()
                + '\nNODES="n0 n1 n2 n3"\n'
                + 'MGMT_IPS="192.0.2.21 192.0.2.22 192.0.2.23 192.0.2.24"\n'
                + 'MASTER_IP=192.0.2.21\nRELAY_DEST=operator@192.0.2.23\n')
            launcher = root / "launch-glm53-tp4.sh"
            shutil.copyfile(REPO / "scripts/launcher/launch-glm53-tp4.sh", launcher)
            controller_home = root / "controller-home"
            node_home = root / "node-home"
            source_command = (
                'source "$1"; source "$2"; printf "%s\\n" "$EXTRA_DOCKER_ENV"')
            sourced = __import__("subprocess").run(
                ["bash", "-c", source_command, "overlay-test", str(cluster), str(overlay)],
                env={**os.environ, "HOME": str(controller_home)},
                capture_output=True, text=True)
            self.assertEqual(sourced.returncode, 0, sourced.stderr)
            runtime_mount = (
                "$HOME/tp4/scripts/resilience/.campaign/unit-overlay/runtime.py:"
                "/opt/tp4-resilience/runtime.py:ro")
            self.assertIn(runtime_mount, sourced.stdout.split())
            self.assertNotIn(str(controller_home), sourced.stdout)
            # The default's disk capacity is replaced, never duplicated, by the campaign's.
            self.assertIn("SPARK_CONTEXT_CACHE_MAX_BYTES=214748364800",
                          (REPO / "cluster.env.example").read_text())
            self.assertEqual(
                [word for word in sourced.stdout.split() if "SPARK_CONTEXT_CACHE_" in word],
                ["SPARK_CONTEXT_CACHE_MAX_BYTES=3221225472",
                 "SPARK_CONTEXT_CACHE_LOW_WATERMARK_BYTES=2147483648"])
            self.assertNotIn("-e -e", sourced.stdout)
            for name, extra in (
                    ("duplicate", ' -e SPARK_CONTEXT_CACHE_MAX_BYTES=1'
                                  ' -e SPARK_CONTEXT_CACHE_LOW_WATERMARK_BYTES=1'),
                    ("unpaired", ' -e SPARK_CONTEXT_CACHE_MAX_BYTES=1'),
                    ("long-form", ' --env SPARK_CONTEXT_CACHE_TTL_SECONDS=60')):
                ambiguous = root / f"{name}.env"
                ambiguous.write_text(cluster.read_text()
                                     + f'EXTRA_DOCKER_ENV+="{extra}"\n')
                refused = __import__("subprocess").run(
                    ["bash", "-c", 'source "$1"; source "$2" || exit 1', "overlay-test",
                     str(ambiguous), str(overlay)],
                    env={**os.environ, "HOME": str(controller_home)},
                    capture_output=True, text=True)
                self.assertNotEqual(refused.returncode, 0, name)
                self.assertIn("at most one plain SparkCache disk-capacity pair",
                              refused.stderr, name)

            launched = __import__("subprocess").run(
                ["bash", str(launcher), "0"],
                env={**os.environ, "HOME": str(node_home), "TP4_DRY_RUN": "1",
                     "TP4_ENV": "scripts/resilience/.campaign/unit-overlay/delta.env"},
                capture_output=True, text=True)
        self.assertEqual(launched.returncode, 0, launched.stderr)
        expected = str(node_home) + runtime_mount[len("$HOME"):]
        self.assertIn("[dry-run] would check mount source: "
                      + expected.split(":", 1)[0], launched.stdout)
        self.assertIn(expected, launched.stdout)
        self.assertNotIn(str(controller_home), launched.stdout)

    def test_remote_init_expands_literal_home_cache_dir(self):
        command = prepare.remote_init_command("unit-overlay")
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            (home / "tp4/scripts/resilience/.campaign/unit-overlay").mkdir(parents=True)
            (home / "tp4/cluster.env").write_text("CACHE_DIR='$HOME/vllm-cache'\n")
            fake = home / "tp4/scripts/resilience/.campaign/unit-overlay/faultctl.py"
            fake.write_text("import json,sys; print(json.dumps(sys.argv[1:]))\n")
            bin_dir = home / "bin"
            bin_dir.mkdir()
            sudo = bin_dir / "sudo"
            sudo.write_text(
                '#!/bin/sh\n[ "$1" = -n ] && shift\n'
                'printf \'%s\\n\' "$*" >> "$HOME/sudo-calls"\nexec "$@"\n')
            sudo.chmod(0o700)
            environment = {**__import__("os").environ, "HOME": str(home),
                           "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"]}
            completed = __import__("subprocess").run(["bash", "-c", command], env=environment,
                text=True, stdout=__import__("subprocess").PIPE, stderr=__import__("subprocess").PIPE)
            sudo_calls = (home / "sudo-calls").read_text().splitlines()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(len(sudo_calls), 2)
        self.assertTrue(sudo_calls[0].startswith("install -d "), sudo_calls)
        self.assertTrue(sudo_calls[1].startswith("python3 "), sudo_calls)
        arguments = json.loads(completed.stdout)
        self.assertEqual(arguments[arguments.index("--root") + 1],
                         str(home / "vllm-cache/jit/tp4-resilience-unit-overlay"))
        self.assertIn("--root " + str(home / "vllm-cache/jit/tp4-resilience-unit-overlay"),
                      sudo_calls[1])

    def test_publication_events_report_the_base_finish_result_and_survive_telemetry_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.py"
            runtime = root / "runtime.py"
            base.write_text(textwrap.dedent('''
                import threading
                def _require_positive_int(name, value): pass
                class MemoryReservation:
                    def __init__(self, budget, peak):
                        self._budget, self._peak_bytes, self._released = budget, peak, False
                class MemoryBudget:
                    def __init__(self, *args, **kwargs):
                        self._lock, self._reserved_bytes = threading.Lock(), 0
                class SparkContextCacheConnector:
                    def __init__(self): self.finishes = []
                    def _capture_stream_snapshot(self, plan, *args, **kwargs):
                        plan.advance(); return 'capture'
                    def _restore_stream_snapshot(self, lookup, plan, *args, **kwargs):
                        plan.advance(); return True
                    def _commit_store_snapshot(self, snapshot):
                        snapshot.advance()
                        if snapshot.mode == 'failed':
                            self._finish_store(snapshot.plan.digest, committed=False,
                                               error=ValueError('commit failed'))
                        elif snapshot.mode == 'evicted':
                            self._finish_store(snapshot.plan.digest, committed=False, evicted=True)
                        else:
                            self._finish_store(snapshot.plan.digest, committed=True,
                                               additional_digests=('alias',))
                    def _finish_store(self, digest, *, committed, evicted=False,
                                      additional_digests=(), error=None):
                        self.finishes.append((digest, committed, evicted,
                                             list(additional_digests),
                                             type(error).__name__ if error else None))
                    def shutdown(self): return 'shutdown'
            '''))
            runtime.write_text(textwrap.dedent('''
                events = []
                pause_error = False
                emit_error = False
                case_id = 'initial'
                def install(): pass
                def write_reservation(*args, **kwargs): pass
                def current_case_id(): return case_id
                def begin_stage(stage, operation_id, **fields):
                    if pause_error: raise TimeoutError('pause expired')
                    fields.setdefault('case_id', case_id)
                    events.append({'kind': 'stage_begin', 'stage': stage, **fields})
                    return fields['case_id']
                def emit_event(kind, **fields):
                    if emit_error: raise OSError('telemetry failed')
                    fields.setdefault('case_id', case_id)
                    events.append({'kind': kind, **fields})
            '''))
            source = textwrap.dedent('''
                import json
                from types import SimpleNamespace
                import connector_wrapper as wrapper
                connector = wrapper.SparkContextCacheConnector()
                def advance(next_case):
                    return lambda: setattr(wrapper._runtime, 'case_id', next_case)
                def snapshot(digest, mode, next_case):
                    return SimpleNamespace(plan=SimpleNamespace(digest=digest), mode=mode,
                                           advance=advance(next_case))
                wrapper._runtime.case_id = 'publication-a'
                connector._commit_store_snapshot(snapshot('a' * 64, 'good', 'after-a'))
                wrapper._runtime.case_id = 'publication-b'
                connector._commit_store_snapshot(snapshot('b' * 64, 'failed', 'after-b'))
                wrapper._runtime.case_id = 'publication-c'
                connector._commit_store_snapshot(snapshot('c' * 64, 'evicted', 'after-c'))
                wrapper._runtime.case_id = 'capture-f'
                capture_plan = SimpleNamespace(digest='f' * 64, advance=advance('after-f'))
                connector._capture_stream_snapshot(capture_plan)
                wrapper._runtime.case_id = 'restore-g'
                restore_plan = SimpleNamespace(digest='g' * 64, advance=advance('after-g'))
                connector._restore_stream_snapshot(SimpleNamespace(is_hit=True), restore_plan)
                wrapper._runtime.case_id = 'publication-d'
                wrapper._runtime.pause_error = True
                connector._commit_store_snapshot(snapshot('d' * 64, 'good', 'after-d'))
                wrapper._runtime.pause_error = False
                wrapper._runtime.case_id = 'publication-e'
                wrapper._runtime.emit_error = True
                connector._commit_store_snapshot(snapshot('e' * 64, 'good', 'after-e'))
                results = [event for event in wrapper._runtime.events
                           if event['kind'] == 'stage_end']
                print(json.dumps({'finishes': connector.finishes, 'results': results}))
            ''')
            environment = os.environ.copy()
            environment.update({
                "PYTHONPATH": str(REPO / "scripts/resilience"),
                "TP4_RESILIENCE_RUNTIME": str(runtime),
                "TP4_RESILIENCE_BASE_CONNECTOR": str(base),
                "TP4_RESILIENCE_BASE_SHA256": hashlib.sha256(base.read_bytes()).hexdigest(),
            })
            completed = __import__("subprocess").run(
                [sys.executable, "-c", source], env=environment, text=True,
                stdout=__import__("subprocess").PIPE, stderr=__import__("subprocess").PIPE)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        value = json.loads(completed.stdout)
        self.assertEqual(len(value["finishes"]), 5)
        self.assertEqual(value["finishes"][0][1:4], [True, False, ["alias"]])
        by_digest = {event["digest"]: event for event in value["results"]}
        self.assertEqual((by_digest["a" * 64]["outcome"],
                          by_digest["a" * 64]["committed"]), ("ok", True))
        self.assertEqual(by_digest["b" * 64]["error"], "ValueError")
        self.assertTrue(by_digest["c" * 64]["evicted"])
        self.assertEqual(by_digest["d" * 64]["outcome"], "ok")
        self.assertEqual(by_digest["d" * 64]["instrumentation_error"],
                         "instrumentation:TimeoutError")
        self.assertNotIn("error", by_digest["d" * 64])
        self.assertEqual(by_digest["a" * 64]["case_id"], "publication-a")
        self.assertEqual(by_digest["f" * 64]["case_id"], "capture-f")
        self.assertEqual(by_digest["g" * 64]["case_id"], "restore-g")
        self.assertEqual(by_digest["g" * 64]["digest"], "g" * 64)
        self.assertNotIn("e" * 64, by_digest)

    def test_observed_budget_orders_updates_and_rolls_back_failed_admission_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.py"
            runtime = root / "runtime.py"
            base.write_text(textwrap.dedent('''
                import threading
                def _require_positive_int(name, value):
                    if not isinstance(value, int) or value <= 0: raise ValueError(name)
                class MemoryReservation:
                    def __init__(self, budget, peak):
                        self._budget, self._peak_bytes, self._released = budget, peak, False
                    def release(self): self._budget._release(self)
                class MemoryBudget:
                    def __init__(self, maximum, floor, available_reader=lambda: 1000000):
                        self._max_bytes, self._min_available_bytes = maximum, floor
                        self._available_reader, self._reserved_bytes = available_reader, 0
                        self._lock = threading.Lock()
                    @property
                    def reserved_bytes(self):
                        with self._lock: return self._reserved_bytes
                class SparkContextCacheConnector:
                    def shutdown(self): pass
            '''))
            runtime.write_text(textwrap.dedent('''
                records = []
                fail = False
                def install(): pass
                def write_reservation(owner, reserved, **fields):
                    if fail: raise OSError('telemetry write failed')
                    records.append(reserved)
                def begin_stage(*args, **kwargs): pass
                def emit_event(*args, **kwargs): pass
            '''))
            source = textwrap.dedent('''
                import json, threading
                import connector_wrapper as wrapper
                budget = wrapper._ObservedMemoryBudget(100, 1, lambda: 1000)
                wrapper._runtime.fail = True
                failed = False
                try: budget.try_reserve(10)
                except OSError: failed = True
                rolled_back = budget.reserved_bytes
                wrapper._runtime.fail = False
                reservation = budget.try_reserve(10)
                wrapper._runtime.fail = True
                release_failed = False
                try: reservation.release()
                except OSError: release_failed = True
                released_after_error = budget.reserved_bytes
                stale_after_error = wrapper._runtime.records[-1]
                wrapper._runtime.fail = False
                barrier = threading.Barrier(3)
                def use():
                    barrier.wait(); reservation = budget.try_reserve(10); reservation.release()
                threads = [threading.Thread(target=use) for _ in range(2)]
                [thread.start() for thread in threads]; barrier.wait()
                [thread.join() for thread in threads]
                print(json.dumps({'failed': failed, 'rolled_back': rolled_back,
                                  'release_failed': release_failed,
                                  'released_after_error': released_after_error,
                                  'stale_after_error': stale_after_error,
                                  'final': budget.reserved_bytes,
                                  'records': wrapper._runtime.records}))
            ''')
            environment = os.environ.copy()
            environment.update({
                "PYTHONPATH": str(REPO / "scripts/resilience"),
                "TP4_RESILIENCE_RUNTIME": str(runtime),
                "TP4_RESILIENCE_BASE_CONNECTOR": str(base),
                "TP4_RESILIENCE_BASE_SHA256": hashlib.sha256(base.read_bytes()).hexdigest(),
            })
            completed = __import__("subprocess").run(
                [sys.executable, "-c", source], env=environment, text=True,
                stdout=__import__("subprocess").PIPE, stderr=__import__("subprocess").PIPE)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        value = json.loads(completed.stdout)
        self.assertTrue(value["failed"])
        self.assertEqual(value["rolled_back"], 0)
        self.assertFalse(value["release_failed"])
        self.assertEqual(value["released_after_error"], 0)
        self.assertEqual(value["stale_after_error"], 10)
        self.assertEqual(value["final"], 0)
        self.assertEqual(value["records"][-1], 0)


if __name__ == "__main__":
    unittest.main()
