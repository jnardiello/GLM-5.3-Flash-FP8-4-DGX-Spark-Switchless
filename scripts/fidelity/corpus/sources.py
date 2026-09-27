"""Parse local agent sessions into GLM chat messages (unredacted, private, in memory only).

Sources:

* Claude Code: ``~/.claude/projects/<project>/<session>.jsonl`` and the subagent
  transcripts ``<project>/<session>/subagents/*.jsonl``; each file is one session.
* omp: every ``*.jsonl`` under ``~/.omp/agent/sessions/<project>/`` (top-level sessions and
  their nested subagent sessions); each file is one session.

A session is split into segments at context compactions (Claude Code compact summaries,
omp ``compaction`` entries); the summary becomes the first user message of the next
segment. Events are taken in file order (both formats are linear in practice; abandoned
branches after a rewind are kept in order). Hidden ``thinking`` blocks are dropped: they
are not GLM-native and may be encrypted. Harness attachments, notifications and metadata
events are skipped. Nothing here prints or logs message content.

Message form (the chat template's): ``{"role": "user", "content", "_human"}``,
``{"role": "assistant", "content", "tool_calls": [{"id", "type": "function",
"function": {"name", "arguments": dict}}]}``, ``{"role": "tool", "tool_call_id",
"content"}`` and ``{"role": "system", "content"}``. ``_human`` marks a user turn typed by a
person (not a tool result, meta injection or compaction summary).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path


CLAUDE_ROOT = Path(os.path.expanduser("~/.claude/projects"))
OMP_ROOT = Path(os.path.expanduser("~/.omp/agent/sessions"))


@dataclass
class Session:
    source: str            # claude_code | omp
    project: str           # project directory name
    key: str               # sha256 prefix of the relative path (no path is stored)
    segments: list = field(default_factory=list)   # list[list[message]]
    stats: Counter = field(default_factory=Counter)


def session_key(root: Path, path: Path) -> str:
    return hashlib.sha256(str(path.relative_to(root)).encode("utf-8")).hexdigest()[:16]


def _after_cutoff(event: dict, cutoff: str | None) -> bool:
    stamp = event.get("timestamp")
    return bool(cutoff and isinstance(stamp, str) and stamp > cutoff)


def _json_lines(path: Path):
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict):
                yield event


def _text_items(items) -> str:
    """Join the text of a content list; images are dropped, tool references keep the name."""
    if isinstance(items, str):
        return items
    parts = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "text" and isinstance(item.get("text"), str):
            parts.append(item["text"])
        elif kind == "tool_reference" and isinstance(item.get("tool_name"), str):
            parts.append(item["tool_name"])
    return "\n".join(parts)


class _Builder:
    """Accumulate messages into segments, merging split assistant events."""

    def __init__(self, session: Session):
        self.session = session
        self.current: list = []
        self.last_assistant_id = None

    def split(self):
        if self.current:
            self.session.segments.append(self.current)
        self.current = []
        self.last_assistant_id = None

    def user(self, text: str, human: bool):
        if not text:
            return
        self.current.append({"role": "user", "content": text, "_human": human})
        self.last_assistant_id = None
        self.session.stats["user_human" if human else "user_other"] += 1
        self.session.stats["chars"] += len(text)

    def system(self, text: str):
        if text:
            self.current.append({"role": "system", "content": text})
            self.last_assistant_id = None
            self.session.stats["system"] += 1
            self.session.stats["chars"] += len(text)

    def tool(self, call_id: str, text: str):
        self.current.append({"role": "tool", "tool_call_id": call_id or "", "content": text or ""})
        self.last_assistant_id = None
        self.session.stats["tool"] += 1
        self.session.stats["chars"] += len(text or "")

    def assistant(self, message_id, texts: list, calls: list):
        if not texts and not calls:
            return
        if (message_id is not None and message_id == self.last_assistant_id and self.current
                and self.current[-1]["role"] == "assistant"):
            target = self.current[-1]
        else:
            target = {"role": "assistant", "content": "", "tool_calls": []}
            self.current.append(target)
            self.session.stats["assistant"] += 1
        self.last_assistant_id = message_id
        text = "\n\n".join(t for t in texts if t)
        if text:
            target["content"] = (target["content"] + "\n\n" + text) if target["content"] else text
            self.session.stats["chars"] += len(text)
        for call_id, name, arguments in calls:
            args = arguments if isinstance(arguments, dict) else {"input": arguments}
            target["tool_calls"].append({"id": call_id or "", "type": "function",
                                         "function": {"name": name, "arguments": args}})
            self.session.stats["tool_calls"] += 1
            self.session.stats["chars"] += len(json.dumps(args, ensure_ascii=False))


def _finish(session: Session) -> Session:
    for segment in session.segments:
        for message in segment:
            if message["role"] == "assistant" and not message["tool_calls"]:
                del message["tool_calls"]
    return session


def parse_claude(path: Path, project: str, cutoff: str | None = None) -> Session:
    session = Session("claude_code", project, session_key(CLAUDE_ROOT, path))
    build = _Builder(session)
    for event in _json_lines(path):
        session.stats["events"] += 1
        kind = event.get("type")
        if kind not in ("user", "assistant") or _after_cutoff(event, cutoff):
            continue
        message = event.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if kind == "user":
            if event.get("isCompactSummary"):
                build.split()
                build.user(_text_items(content), human=False)
                continue
            human = not event.get("isMeta")
            if isinstance(content, str):
                build.user(content, human)
                continue
            texts = []
            for block in content or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_result":
                    build.tool(block.get("tool_use_id"), _text_items(block.get("content")))
                elif block.get("type") == "text":
                    texts.append(block.get("text") or "")
            build.user("\n".join(t for t in texts if t), human)
        else:
            if message.get("model") == "<synthetic>":
                continue
            texts, calls = [], []
            for block in content or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    texts.append(block.get("text") or "")
                elif block.get("type") == "tool_use":
                    calls.append((block.get("id"), block.get("name") or "unknown", block.get("input")))
            build.assistant(message.get("id"), texts, calls)
    build.split()
    return _finish(session)


def parse_omp(path: Path, project: str, cutoff: str | None = None) -> Session:
    session = Session("omp", project, session_key(OMP_ROOT, path))
    build = _Builder(session)
    for event in _json_lines(path):
        session.stats["events"] += 1
        kind = event.get("type")
        if _after_cutoff(event, cutoff):
            continue
        if kind == "compaction":
            build.split()
            build.user(event.get("summary") if isinstance(event.get("summary"), str) else "", human=False)
            continue
        if kind != "message" or not isinstance(event.get("message"), dict):
            continue
        message = event["message"]
        role = message.get("role")
        content = message.get("content")
        if role == "user":
            build.user(_text_items(content), human=True)
        elif role == "developer":
            build.system(_text_items(content))
        elif role == "toolResult":
            build.tool(message.get("toolCallId"), _text_items(content))
        elif role == "assistant":
            texts, calls = [], []
            for block in content or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    texts.append(block.get("text") or "")
                elif block.get("type") == "toolCall":
                    calls.append((block.get("id"), block.get("name") or "unknown", block.get("arguments")))
            build.assistant(event.get("id"), texts, calls)
    build.split()
    return _finish(session)


def claude_files(root: Path = CLAUDE_ROOT):
    """(project, path) for every Claude Code session and subagent transcript."""
    if not root.is_dir():
        return
    for project in sorted(p for p in root.iterdir() if p.is_dir()):
        for path in sorted(project.glob("*.jsonl")) + sorted(project.glob("*/subagents/*.jsonl")):
            yield project.name, path


def omp_files(root: Path = OMP_ROOT):
    if not root.is_dir():
        return
    for project in sorted(p for p in root.iterdir() if p.is_dir()):
        for path in sorted(project.rglob("*.jsonl")):
            yield project.name, path


def load_sessions(excluded: set[str], cutoff: str | None = None):
    """Yield parsed sessions from both sources; ``excluded`` holds project names.

    An exclusion entry is either a bare project directory name (both sources) or
    ``claude_code/<name>`` / ``omp/<name>``.
    """
    for source, files, parse in (("claude_code", claude_files(), parse_claude),
                                 ("omp", omp_files(), parse_omp)):
        for project, path in files:
            if project in excluded or f"{source}/{project}" in excluded:
                continue
            yield parse(path, project, cutoff)
