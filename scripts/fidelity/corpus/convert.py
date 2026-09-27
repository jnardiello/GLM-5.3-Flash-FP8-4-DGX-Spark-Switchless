"""Redact parsed messages and synthesise tool schemas from observed tool calls.

Redaction runs on every string before any rendering or tokenisation: message content,
tool-result text and every string inside tool-call arguments (argument keys are kept).
Tool schemas are synthesised per source from the tool names and the argument keys and
JSON types observed in that source's calls:
``{"type": "function", "function": {"name", "description": "", "parameters":
{"type": "object", "properties": {key: {"type": <most frequent JSON type>}}}}}``.
Each segment is rendered with the schemas of the tools it actually calls (sorted by name),
which keeps the repeated tools header short; segments without calls get no tools block.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json

from redact import Redactor


def json_type(value) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "null"


def redact_segment(messages: list, redactor: Redactor) -> list:
    out = []
    for message in messages:
        copy = dict(message)
        if isinstance(copy.get("content"), str):
            copy["content"] = redactor.text(copy["content"])
        if copy.get("tool_calls"):
            calls = []
            for call in copy["tool_calls"]:
                function = call["function"]
                calls.append({"id": call.get("id", ""), "type": "function",
                              "function": {"name": function["name"],
                                           "arguments": redactor.value(function["arguments"])}})
            copy["tool_calls"] = calls
        out.append(copy)
    return out


class ToolSchemas:
    """Observed argument keys/types per (source, tool name)."""

    def __init__(self):
        self.types: dict = defaultdict(lambda: defaultdict(Counter))

    def observe(self, source: str, messages: list) -> None:
        for message in messages:
            for call in message.get("tool_calls") or []:
                function = call["function"]
                table = self.types[(source, function["name"])]
                for key, value in function["arguments"].items():
                    table[key][json_type(value)] += 1

    def schema(self, source: str, name: str) -> dict:
        table = self.types.get((source, name), {})
        properties = {}
        for key in sorted(table, key=lambda k: (-sum(table[k].values()), k)):
            kind = sorted(table[key].items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            properties[key] = {"type": kind}
        return {"type": "function", "function": {"name": name, "description": "",
                                                 "parameters": {"type": "object", "properties": properties}}}

    def for_segment(self, source: str, messages: list):
        names = sorted({call["function"]["name"] for m in messages for call in m.get("tool_calls") or []})
        return [self.schema(source, name) for name in names] or None


CALL_OVERHEAD = 40      # approx. characters of <tool_call>/<arg_key>/<arg_value> markup
STRING_ARG_CAP = 256    # long string arguments (file bodies, scripts) count as free text


def structured_chars(unit: list) -> tuple[int, int]:
    """(structured, total) characters of a unit.

    Structured text is tool-call markup (names, keys, non-string values and the first
    ``STRING_ARG_CAP`` characters of each string value) and tool results that parse as
    JSON. The rest of long string arguments, such as a file written by a tool, is free text,
    so a window that only writes large files does not count as structured.
    """
    structured = total = 0
    for message in unit:
        content = message.get("content") or ""
        total += len(content)
        if message["role"] == "tool":
            stripped = content.strip()
            if stripped[:1] in "{[":
                try:
                    json.loads(stripped)
                    structured += len(content)
                except ValueError:
                    pass
        for call in message.get("tool_calls") or []:
            size = CALL_OVERHEAD + len(call["function"]["name"])
            full = size
            for key, value in call["function"]["arguments"].items():
                text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                size += len(key) + (min(len(text), STRING_ARG_CAP) if isinstance(value, str) else len(text))
                full += len(key) + len(text)
            structured += size
            total += full
    return structured, total
