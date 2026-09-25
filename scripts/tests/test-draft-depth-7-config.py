#!/usr/bin/env python3
"""Offline contract for the E28 draft-depth-7 candidate overlay; no GPU, node or network.

The overlay must apply only on the default E27c recipe and change exactly the speculative
configuration (seven tokens, table [[1,1,7],[2,6,3]]), the adaptive high state and the CUDA
graph capture limit, identically on every rank.
"""
from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]
CANDIDATE = REPO / "scripts/node/experiments/e03/draft-depth-7"


def spec(tokens: int) -> str:
    return json.dumps({"method": "dflash", "model": "/draft", "num_speculative_tokens": tokens,
                       "num_speculative_tokens_per_batch_size": [[1, 1, tokens], [2, 6, 3]],
                       "kv_cache_dtype": "fp8_e4m3"}, separators=(",", ":"))


class Launcher(unittest.TestCase):
    def test_four_rank_parity_and_refusals(self):
        delta = (CANDIDATE / "delta.env").read_text()
        with tempfile.TemporaryDirectory(prefix="tp4-e28-") as temp:
            root = Path(temp)
            shutil.copyfile(REPO / "scripts/launcher/launch-glm53-tp4.sh", root / "launch.sh")
            (root / "cluster.env").write_text((REPO / "cluster.env.example").read_text() + '''
NODES="n0 n1 n2 n3"
MGMT_IPS="192.0.2.21 192.0.2.22 192.0.2.23 192.0.2.24"
MASTER_IP=192.0.2.21
RELAY_DEST=operator@192.0.2.23
''')
            e27 = (REPO / "scripts/node/reference/baseline-20260924-e27.env").read_text() + "\n"
            # The E28b default already carries seven draft tokens; E28 was measured on E27c.
            e27c = (REPO / "scripts/node/reference/baseline-20260925-e27c.env").read_text() + "\n"
            kv = (REPO / "scripts/node/experiments/e03/kv-16gib/delta.env").read_text()
            (root / "empty.env").write_text(e27c)
            (root / "candidate.env").write_text(e27c + delta)
            (root / "kv.env").write_text(e27c + delta + "\n" + kv)
            (root / "default.env").write_text("")
            env = dict(os.environ, TP4_DRY_RUN="1")
            env.pop("TP4_ENV", None)
            forbidden = root / "forbidden.log"
            bindir = root / "bin"
            bindir.mkdir()
            for name in ("sudo", "docker", "ssh", "systemctl", "curl", "ip", "sysctl"):
                p = bindir / name
                p.write_text('#!/bin/sh\nprintf "%s\\n" "$0" >> "$TP4_FORBIDDEN"\nexit 97\n')
                p.chmod(0o700)
            env.update(PATH=str(bindir) + os.pathsep + env["PATH"], TP4_FORBIDDEN=str(forbidden))

            def launch(overlay, rank=0):
                result = subprocess.run(["bash", str(root / "launch.sh"), str(rank)],
                                        env=dict(env, TP4_ENV=overlay), capture_output=True,
                                        text=True, timeout=20)
                self.assertFalse(forbidden.exists(), "Dry-run attempted an external action")
                return result

            def argv(result):
                self.assertEqual(result.returncode, 0, result.stderr)
                return [line[2:] for line in result.stdout.splitlines() if line.startswith("  ")]

            for rank in range(4):
                before, after = argv(launch("empty.env", rank)), argv(launch("candidate.env", rank))
                removed, added = Counter(before) - Counter(after), Counter(after) - Counter(before)
                self.assertEqual(removed, Counter([spec(5)]))
                self.assertEqual(added, Counter([spec(7), "-e", "VLLM_ADAPTIVE_K_HI=7",
                                                 '--compilation-config={"max_cudagraph_capture_size":72}']))
                self.assertEqual(after[after.index("--max-num-seqs") + 1], "6")
                # E28b: the 16 GiB KV overlay changes only the KV pool.
                kv16 = argv(launch("kv.env", rank))
                self.assertEqual(Counter(after) - Counter(kv16), Counter(["--kv-cache-memory-bytes=16106127360"]))
                self.assertEqual(Counter(kv16) - Counter(after), Counter(["--kv-cache-memory-bytes=17179869184"]))
                # The promoted E28b default is exactly that measured command.
                self.assertEqual(argv(launch("default.env", rank)), kv16)

            bad = {
                "on-e28b-default": delta,
                "kv-on-e27c": e27c + kv,
                "applied-twice": e27c + delta + "\n" + delta,
                "on-e27-rollback": e27 + delta,
                "max-seqs-5": e27c + "MAX_NUM_SEQS=5\n" + delta,
                "k-hi-present": e27c + "EXTRA_DOCKER_ENV+=' -e VLLM_ADAPTIVE_K_HI=5'\n" + delta,
                "compilation-present": e27c + "EXTRA_VLLM_ARGS+=' -O3'\n" + delta,
                "table-changed": e27c + """SPEC_EXTRA_JSON='"num_speculative_tokens_per_batch_size":[[1,2,5],[3,6,3]],"kv_cache_dtype":"fp8_e4m3"'\n""" + delta,
            }
            for name, text in bad.items():
                (root / "bad.env").write_text(text)
                self.assertNotEqual(launch("bad.env").returncode, 0, name)

    def test_readme_names_the_overlay(self):
        text = (CANDIDATE / "README.md").read_text()
        self.assertIn("TP4_ENV=scripts/node/experiments/e03/draft-depth-7/delta.env", text)
        self.assertIn("max_cudagraph_capture_size", text)


if __name__ == "__main__":
    unittest.main(verbosity=1)
