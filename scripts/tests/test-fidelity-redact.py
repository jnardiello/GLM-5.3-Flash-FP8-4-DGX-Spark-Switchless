#!/usr/bin/env python3
"""Offline tests for scripts/fidelity/corpus/redact.py; stdlib only, no network.

Every secret below is made up and assembled at run time from a seeded generator, so no
literal credential-shaped string is stored in the repository. The tests check that each
rule fires, that counts are exact, that no value is double counted, and that git SHAs,
UUIDs, digests, paths and ordinary code identifiers survive unchanged.
"""

from __future__ import annotations

from pathlib import Path
import random
import string
import sys
import unittest


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/fidelity/corpus"))
import redact  # noqa: E402

RNG = random.Random(20260927)
ALNUM = string.ascii_letters + string.digits


def fake(alphabet: str, n: int) -> str:
    """A made-up random string; mixes letters and digits when the alphabet has both."""
    mixed = any(c.isdigit() for c in alphabet) and any(c.isalpha() for c in alphabet)
    for _ in range(1000):
        text = "".join(RNG.choice(alphabet) for _ in range(n))
        if not mixed or (any(c.isdigit() for c in text) and any(c.isalpha() for c in text)):
            return text
    raise AssertionError("could not generate a mixed string")


def fake_b64url(n: int) -> str:
    return fake(ALNUM + "_-", n)


def jwt() -> str:
    return "eyJ" + fake_b64url(20) + ".eyJ" + fake_b64url(40) + "." + fake_b64url(43)


def pem() -> str:
    body = "\n".join(fake(ALNUM + "+/", 64) for _ in range(4))
    kind = "RSA " + "PRIVATE KEY"
    return f"-----BEGIN {kind}-----\n{body}\n-----END {kind}-----"


# (rule, text containing exactly one secret of that rule)
CASES = [
    ("pem_private_key", "key file:\n" + pem() + "\nend"),
    ("jwt", "cookie value " + jwt() + " expires"),
    ("anthropic_key", "export X=1; key " + "sk-" + "ant-" + "api03-" + fake_b64url(40)),
    ("openai_key", "using " + "sk-" + "proj-" + fake_b64url(40) + " for tests"),
    ("github_token", "remote token " + "gh" + "p_" + fake(ALNUM, 36)),
    ("github_token", "pat " + "github" + "_pat_" + fake(ALNUM + "_", 60)),
    ("slack_token", "hook " + "xo" + "xb-" + fake(string.digits, 12) + "-" + fake(ALNUM, 24)),
    ("aws_access_key", "id " + "AK" + "IA" + fake(string.ascii_uppercase + string.digits, 16) + " region"),
    ("google_api_key", "maps " + "AI" + "za" + fake(ALNUM + "_-", 35)),
    ("hf_token", "hub " + "h" + "f_" + fake(ALNUM, 34)),
    ("gitlab_token", "ci " + "gl" + "pat-" + fake(ALNUM + "_-", 20)),
    ("bearer", "curl -H 'Authorization: Bearer " + fake(ALNUM, 32) + ".x' url"),
    ("env_assignment", "cat .env\nDATABASE_PASSWORD=" + "Winter" + "Coat" + "Lamp\nDEBUG=1"),
    ("env_assignment", "docker run -e MY_SERVICE_TOKEN=" + "plain" + "words" + "value -it"),
    ("password_assignment", 'config = {"password": "' + "correct" + "horse" + '"}'),
    ("password_assignment", "mysql --passwd=" + "letmein" + "please now"),
    ("email", "contact jane.doe+test@example.org for access"),
    ("entropy", "session blob " + fake(ALNUM + "+/", 48) + " stored"),
]

# Strings that must survive redaction unchanged.
BENIGN = [
    "commit 3f786850e387550fdab836ed7e6dc881de23001b fixed it",
    "short sha 080fe09 and 1a2b3c4d5e6f",
    "image sha256:" + "".join(RNG.choice("0123456789abcdef") for _ in range(64)),
    "request id 123e4567-e89b-12d3-a456-426614174000 ok",
    "path /Users/example/workspace/project-name/data/fidelity/corpus",
    "run scripts/launcher/launch-glm53-tp4.sh with TP4_DRY_RUN=1",
    "class Glm5NextForConditionalGeneration and getElementByIdOrThrowAnError",
    "MAX_NUM_BATCHED_TOKENS=8192 NUM_SPECULATIVE_TOKENS=7 USE_AUTH=true",
    "export HF_TOKEN=$HF_TOKEN; API_KEY=${API_KEY}; SECRET_KEY=<your key>",
    "max_tokens=1024 and prompt_tokens=100 are request fields",
    "use a Bearer token in the Authorization header",
    "sparse_attn_indexer_kpool_sm121 fp8_e4m3_kv_cache_dtype_auto",
    "model zai-org/GLM-5.3-Flash/resolve/main/model-00001-of-00093.safetensors",
    "Qwen3-Coder-30B-A3B-Instruct-FP8 and nvidia/Llama-3_3-Nemotron-Super-49B-v1_5",
    "def bypass(x): return x  # no pass statement here",
    "output /tmp/build-x9k2q/reports/final-summary-document/index",
    "archive 20260926T231500Z-nightly-benchmark-results-folder",
    "",
]


class RuleTests(unittest.TestCase):
    def test_each_rule_fires_once(self):
        seen = set()
        for rule, text in CASES:
            with self.subTest(rule=rule):
                r = redact.Redactor()
                out = r.text(text)
                self.assertIn(f"[REDACTED:{rule}]", out)
                counts = r.snapshot()
                self.assertEqual(counts[rule], 1, counts)
                self.assertEqual(sum(counts.values()), 1, counts)
                seen.add(rule)
        rules = {name for name, _, _ in redact.RULES} | {"entropy"}
        self.assertEqual(seen, rules, "every rule needs a positive case")

    def test_secret_value_removed(self):
        for rule, text in CASES:
            with self.subTest(rule=rule):
                out = redact.Redactor().text(text)
                secret_tokens = [t for t in text.split() if len(t) >= 20 and t not in out]
                self.assertTrue(secret_tokens or rule in ("env_assignment", "password_assignment",
                                                          "email", "pem_private_key"))
                # No 12+ character run of the original secret survives.
                for token in secret_tokens:
                    core = token.strip("'\".,;")
                    for i in range(0, max(1, len(core) - 12)):
                        piece = core[i:i + 12]
                        if piece.isalnum() and any(c.isdigit() for c in piece):
                            self.assertNotIn(piece, out)

    def test_assignment_keeps_key(self):
        out = redact.Redactor().text("DATABASE_PASSWORD=" + "Winter" + "CoatLamp")
        self.assertEqual(out, "DATABASE_PASSWORD=[REDACTED:env_assignment]")
        out = redact.Redactor().text('"password": "' + "hunter" + 'two"')
        self.assertEqual(out, '"password": "[REDACTED:password_assignment]"')

    def test_benign_unchanged(self):
        for text in BENIGN:
            with self.subTest(text=text[:40]):
                r = redact.Redactor()
                self.assertEqual(r.text(text), text)
                self.assertEqual(sum(r.snapshot().values()), 0)

    def test_no_double_count(self):
        key = "sk-" + "proj-" + fake_b64url(40)
        r = redact.Redactor()
        out = r.text(f"OPENAI_API_KEY={key}\nAuthorization: Bearer {key}")
        self.assertEqual(out, "OPENAI_API_KEY=[REDACTED:openai_key]\nAuthorization: Bearer [REDACTED:openai_key]")
        counts = r.snapshot()
        self.assertEqual(counts["openai_key"], 2)
        self.assertEqual(sum(counts.values()), 2)

    def test_counts_accumulate(self):
        r = redact.Redactor()
        text = " ".join(t for _, t in CASES)
        r.text(text)
        r.text(text)
        expected = {}
        for rule, _ in CASES:
            expected[rule] = expected.get(rule, 0) + 2
        counts = {k: v for k, v in r.snapshot().items() if v}
        self.assertEqual(counts, expected)

    def test_value_recurses_and_keeps_keys(self):
        r = redact.Redactor()
        value = {"command": "curl -u jane.doe@example.com", "env": ["A=1", "hub " + "h" + "f_" + fake(ALNUM, 34)],
                 "n": 3, "password": None}
        out = r.value(value)
        self.assertEqual(set(out), set(value))
        self.assertEqual(out["n"], 3)
        self.assertIsNone(out["password"])
        self.assertIn("[REDACTED:email]", out["command"])
        self.assertIn("[REDACTED:hf_token]", out["env"][1])
        self.assertEqual(r.snapshot()["email"] + r.snapshot()["hf_token"], 2)

    def test_placeholder_not_rematched(self):
        r = redact.Redactor()
        once = r.text(" ".join(t for _, t in CASES))
        before = r.snapshot()
        self.assertEqual(r.text(once), once)
        self.assertEqual(r.snapshot(), before)


class EntropyTests(unittest.TestCase):
    def test_random_tokens_detected(self):
        hits = sum(redact.entropy_secret(fake(ALNUM + "+/", 40)) for _ in range(200))
        self.assertGreaterEqual(hits, 195)

    def test_hex_and_uuid_excluded(self):
        for _ in range(200):
            sha = "".join(RNG.choice("0123456789abcdef") for _ in range(40))
            self.assertFalse(redact.entropy_secret(sha))
            u = "-".join("".join(RNG.choice("0123456789abcdef") for _ in range(n)) for n in (8, 4, 4, 4, 12))
            self.assertFalse(redact.entropy_secret(u))
            # SHA-256 digests are not excluded, but hex cannot exceed 4.0 bits/char and
            # random 64-char digests stay below the threshold.
            digest = "".join(RNG.choice("0123456789abcdef") for _ in range(64))
            self.assertFalse(redact.entropy_secret(digest))

    def test_thresholds(self):
        self.assertEqual(redact.ENTROPY_MIN_LEN, 20)
        self.assertEqual(redact.ENTROPY_MIN_BITS, 4.0)
        self.assertFalse(redact.entropy_secret(fake(ALNUM, 19)))
        self.assertFalse(redact.entropy_secret("abcdefghijklmnopqrstuvwxyz"))  # one class
        self.assertAlmostEqual(redact.shannon_entropy("abcd"), 2.0)


if __name__ == "__main__":
    unittest.main(verbosity=1)
