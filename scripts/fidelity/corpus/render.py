"""Render chat messages with the runtime template and tokenize them as vLLM does.

vLLM's chat path renders the conversation with the tokenizer's ``apply_chat_template``
(``tokenize=False``) and then tokenizes the string with ``add_special_tokens=False``. This
module does exactly that with the pinned tokenizer and the runtime template
``tp4-chat-template.jinja`` (the file the server is started with), and records the
template kwargs. ``check_equivalence`` confirms on a synthetic conversation that this
matches ``apply_chat_template(tokenize=True)``.

Windows are assembled from per-unit token arrays. A unit is one user, system or
assistant message, the assistant unit including the tool results that follow it. Every
unit starts with a special token (``<|user|>``, ``<|system|>``, ``<|assistant|>``), and
the template renders each unit independently of the preceding ones (tool-result
ordering depends only on the unit's own assistant message; ``clear_thinking`` is off),
so header + concatenated unit tokens equals the tokenization of the full render. The
builder re-renders every selected window in full and asserts this.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TOKENIZER_DIR = REPO / "data/fidelity/tokenizer"
TEMPLATE_FILE = "tp4-chat-template.jinja"
TEMPLATE_KWARGS = {"reasoning_effort": "high"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _public(message: dict) -> dict:
    return {k: v for k, v in message.items() if not k.startswith("_")}


class Renderer:
    def __init__(self, tokenizer_dir: Path = TOKENIZER_DIR):
        from transformers import AutoTokenizer  # local files only; no network
        self.dir = Path(tokenizer_dir)
        self.tok = AutoTokenizer.from_pretrained(str(self.dir), local_files_only=True)
        self.template = (self.dir / TEMPLATE_FILE).read_text(encoding="utf-8")
        self.kwargs = dict(TEMPLATE_KWARGS)
        self.tokenizer_sha256 = sha256_file(self.dir / "tokenizer.json")
        self.template_sha256 = sha256_file(self.dir / TEMPLATE_FILE)
        self.vocab_len = len(self.tok)
        self._headers: dict = {}
        self._gen_prompt = None

    # ------------------------------------------------------------ primitives
    def render(self, messages: list, tools: list | None, add_generation_prompt: bool) -> str:
        return self.tok.apply_chat_template([_public(m) for m in messages], tools=tools or None,
                                            chat_template=self.template, tokenize=False,
                                            add_generation_prompt=add_generation_prompt, **self.kwargs)

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def encode_many(self, texts: list[str]) -> list[list[int]]:
        if not texts:
            return []
        return [list(e) for e in self.tok(texts, add_special_tokens=False)["input_ids"]]

    def tokens(self, messages: list, tools: list | None, add_generation_prompt: bool) -> list[int]:
        return self.encode(self.render(messages, tools, add_generation_prompt))

    # ------------------------------------------------------------ units
    def header_text(self, tools: list | None) -> str:
        # transformers refuses an empty conversation: render one empty user turn and
        # strip its role token.
        text = self.render([{"role": "user", "content": ""}], tools, False)
        if not text.endswith("<|user|>"):
            raise RuntimeError("unexpected header render")
        return text[:-len("<|user|>")]

    def header_ids(self, tools: list | None) -> list[int]:
        key = repr(tools)
        if key not in self._headers:
            self._headers[key] = self.encode(self.header_text(tools))
        return self._headers[key]

    def unit_texts(self, units: list[list[dict]]) -> list[str]:
        """Render each unit alone and strip the (tool-free) header prefix."""
        prefix = self.header_text(None)
        out = []
        for unit in units:
            text = self.render(unit, None, False)
            if not text.startswith(prefix):
                raise RuntimeError("unit render does not start with the header")
            out.append(text[len(prefix):])
        return out

    def generation_prompt_ids(self) -> list[int]:
        if self._gen_prompt is None:
            with_gen = self.render([{"role": "user", "content": "x"}], None, True)
            without = self.render([{"role": "user", "content": "x"}], None, False)
            if not with_gen.startswith(without):
                raise RuntimeError("generation prompt is not a suffix")
            self._gen_prompt = self.encode(with_gen[len(without):])
        return self._gen_prompt

    def check_equivalence(self) -> dict:
        """Render+encode vs apply_chat_template(tokenize=True) on a synthetic conversation."""
        messages = [
            {"role": "system", "content": "Synthetic system note."},
            {"role": "user", "content": "List the files, then count them."},
            {"role": "assistant", "content": "Listing.", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": {"command": "ls", "n": 2}}},
                {"id": "c2", "type": "function", "function": {"name": "bash", "arguments": {"command": "wc -l"}}}]},
            {"role": "tool", "tool_call_id": "c2", "content": "2"},
            {"role": "tool", "tool_call_id": "c1", "content": "a\nb"},
            {"role": "assistant", "content": "Two files: a and b."},
            {"role": "user", "content": "Grazie! {\"ok\": true}"},
        ]
        tools = [{"type": "function", "function": {"name": "bash", "description": "", "parameters": {
            "type": "object", "properties": {"command": {"type": "string"}, "n": {"type": "integer"}}}}}]
        result = {}
        for gen in (False, True):
            ours = self.tokens(messages, tools, gen)
            ref = self.tok.apply_chat_template(messages, tools=tools, chat_template=self.template,
                                               tokenize=True, add_generation_prompt=gen, **self.kwargs)
            if hasattr(ref, "keys"):
                ref = ref["input_ids"]
            result[f"add_generation_prompt={gen}"] = list(ref) == ours
        units = split_units(messages)
        assembled = list(self.header_ids(tools))
        for ids in self.encode_many(self.unit_texts(units)):
            assembled += ids
        result["units_concat"] = assembled == self.tokens(messages, tools, False)
        return result


def split_units(messages: list) -> list[list[dict]]:
    """Group messages into units; leading orphan tool results are dropped."""
    units: list = []
    for message in messages:
        if message["role"] == "tool":
            if units and units[-1][0]["role"] == "assistant":
                units[-1].append(message)
            continue
        units.append([message])
    return units
