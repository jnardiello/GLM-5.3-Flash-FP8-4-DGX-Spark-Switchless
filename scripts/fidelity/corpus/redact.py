"""Secret redaction for the fidelity corpus; stdlib only.

Every match is replaced with ``[REDACTED:<rule>]`` and counted per rule. Values are never
logged or returned: callers only see the redacted text and the per-rule counters.

Rules run in the order of ``RULES``; later rules never re-match an earlier placeholder.
The entropy rule is a heuristic for secrets that have no recognisable prefix:

* a candidate is a run of at least 20 characters from ``[A-Za-z0-9+/=_-]``;
* it is redacted when its Shannon entropy is at least 4.0 bits per character and it uses at
  least two character classes (lower case, upper case, digits, ``+/=_-``);
* it is kept unless its non-benign parts (between the separators ``+/=_-``) total at least
  20 characters. Benign parts are decimal numbers, pure hexadecimal of at most 40
  characters (git SHA-1 and short SHAs, the hex groups of a UUID) and word-like text (see
  ``_word_like``). This keeps git SHAs, UUIDs, file paths with an odd short component,
  compact timestamps and ordinary code identifiers (``snake_case``, ``camelCase``,
  ``Glm5NextForCausalLM``) while random tokens, which switch character class every one or
  two characters, are redacted.
  Pure hexadecimal longer than 40 characters (SHA-256 digests) is not excluded explicitly;
  a 16-symbol alphabet cannot exceed 4.0 bits per character, so it practically never meets
  the threshold.
"""

from __future__ import annotations

from collections import Counter
import math
import re


PLACEHOLDER = "[REDACTED:{}]"
_NOT_PLACEHOLDER = r"(?!\[REDACTED:)"

# Env-style keys (upper case, as in .env files and `export`/`-e` assignments).
_ENV_KEY = r"[A-Z][A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASS|PASSWORD|AUTH|CREDENTIAL|COOKIE|SESSION)[A-Z0-9_]*"
# Values that are configuration rather than credentials: numbers, booleans, references.
_ENV_BENIGN = re.compile(r"^(?:\d+(?:\.\d+)?|true|false|yes|no|on|off|none|null|\$.*|<.*|\{.*|%.*)$",
                         re.IGNORECASE)

_PEM = re.compile(
    r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----"
    r"(?:[A-Za-z0-9+/=:,.\\\r\n\t ]|-(?!----END))*"
    r"(?:-----END (?:[A-Z0-9]+ )*PRIVATE KEY-----)?")

RULES: list[tuple[str, re.Pattern, int]] = [
    # (name, pattern, group to replace; 0 = whole match)
    ("pem_private_key", _PEM, 0),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"), 0),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}"), 0),
    ("openai_key", re.compile(r"\bsk-(?=[A-Za-z0-9_-]*\d)[A-Za-z0-9_-]{20,}"), 0),
    ("github_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})"), 0),
    ("slack_token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"), 0),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"), 0),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}"), 0),
    ("hf_token", re.compile(r"\bhf_[A-Za-z0-9]{30,}"), 0),
    ("gitlab_token", re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}"), 0),
    ("bearer", re.compile(r"(?i:\bbearer)\s+" + _NOT_PLACEHOLDER
                          + r"((?=[A-Za-z0-9._~+/=-]*\d)[A-Za-z0-9._~+/=-]{16,})"), 1),
    ("env_assignment", re.compile(r"(?<![A-Za-z0-9_$])" + _ENV_KEY + r"\s*=\s*[\"']?" + _NOT_PLACEHOLDER
                                  + r"([^\s\"'#;&|]+)"), 1),
    ("password_assignment", re.compile(r"(?i)\b(?:password|passwd|pwd|pass)[\"']?\s*[:=]\s*[\"']?"
                                       + _NOT_PLACEHOLDER + r"([^\s\"',;]+)"), 1),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b"), 0),
]

_RUN = re.compile(r"[A-Za-z0-9+/=_-]{20,}")
_SEPARATORS = re.compile(r"[+/=_-]+")
_HEX = re.compile(r"^[0-9a-fA-F]+$")
_PIECES = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")
ENTROPY_MIN_LEN = 20
ENTROPY_MIN_BITS = 4.0


def shannon_entropy(text: str) -> float:
    counts = Counter(text)
    n = len(text)
    return -sum(c / n * math.log2(c / n) for c in counts.values())


def char_classes(text: str) -> int:
    return (any(c.islower() for c in text) + any(c.isupper() for c in text)
            + any(c.isdigit() for c in text) + any(c in "+/=_-" for c in text))


def _word_like(part: str) -> bool:
    """Identifier-like text, judged on its pieces (camelCase words, ACRONYMS, numbers).

    Word-like when the alphabetic pieces average at least 3 characters, at most a quarter
    of them have 1-2 characters, numbers do not alternate with words (at most one number
    per two words) and, from 8 letters on, at least 20% of the letters are vowels. Random
    strings change character class every one or two characters, so they fail these tests;
    words, paths and identifiers such as ``launch_glm53_tp4`` pass.
    """
    if len(part) <= 4:
        return True
    pieces = _PIECES.findall(part)
    if "".join(pieces) != part:
        return False
    alpha = [p for p in pieces if not p.isdigit()]
    digits = len(pieces) - len(alpha)
    if not alpha:
        return True
    letters = "".join(alpha)
    if sum(len(p) for p in alpha) / len(alpha) < 3.0:
        return False
    if sum(1 for p in alpha if len(p) <= 2) > max(1, len(alpha) // 4):
        return False
    if digits > max(1, len(alpha) // 2):
        return False
    if len(letters) >= 8 and sum(c in "aeiouyAEIOUY" for c in letters) < 0.2 * len(letters):
        return False
    return True


def _benign_part(part: str) -> bool:
    if not part or part.isdigit():
        return True
    if _HEX.match(part) and len(part) <= 40:
        return True
    return _word_like(part)


def entropy_secret(run: str) -> bool:
    """True when a candidate run should be redacted by the entropy rule."""
    if len(run) < ENTROPY_MIN_LEN or run.startswith("REDACTED"):
        return False
    suspicious = [p for p in _SEPARATORS.split(run) if not _benign_part(p)]
    if sum(len(p) for p in suspicious) < ENTROPY_MIN_LEN:
        return False
    return char_classes(run) >= 2 and shannon_entropy(run) >= ENTROPY_MIN_BITS


class Redactor:
    """Apply the rules and accumulate per-rule counts (numbers only)."""

    def __init__(self) -> None:
        self.counts: Counter = Counter({name: 0 for name, _, _ in RULES})
        self.counts["entropy"] = 0

    def _rule(self, name: str, pattern: re.Pattern, group: int, text: str) -> str:
        placeholder = PLACEHOLDER.format(name)

        def replace(match: re.Match) -> str:
            if name == "env_assignment" and _ENV_BENIGN.match(match.group(1)):
                return match.group(0)
            if name == "password_assignment" and match.group(1)[:1] in "$<{":
                return match.group(0)
            self.counts[name] += 1
            if group == 0:
                return placeholder
            start, end = match.span(group)
            base = match.start()
            whole = match.group(0)
            return whole[:start - base] + placeholder + whole[end - base:]

        return pattern.sub(replace, text)

    def _entropy(self, text: str) -> str:
        def replace(match: re.Match) -> str:
            if entropy_secret(match.group(0)):
                self.counts["entropy"] += 1
                return PLACEHOLDER.format("entropy")
            return match.group(0)

        return _RUN.sub(replace, text)

    def text(self, text: str) -> str:
        if not text:
            return text
        for name, pattern, group in RULES:
            text = self._rule(name, pattern, group, text)
        return self._entropy(text)

    def value(self, value):
        """Redact every string inside a JSON-like value (dict keys are kept)."""
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, list):
            return [self.value(v) for v in value]
        if isinstance(value, dict):
            return {k: self.value(v) for k, v in value.items()}
        return value

    def snapshot(self) -> dict:
        return dict(sorted(self.counts.items()))
