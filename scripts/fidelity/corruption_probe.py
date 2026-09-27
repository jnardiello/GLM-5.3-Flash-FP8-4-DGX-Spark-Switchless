#!/usr/bin/env python3
"""Corruption probe for an arm (docs/fidelity/PLAN.md §6, amendment 13): Italian and tool-call
prompts at temperature 0, counting invalid UTF-8 byte-token runs, repetition locks and
tool-call parse failures. Motivated by vLLM #54150 (ModelOpt NVFP4 GLM-5.3-Flash checkpoints
emitting invalid UTF-8 byte tokens).

UTF-8 validity is checked on the concatenated bytes of the generated tokens (individual byte
tokens may legitimately hold partial characters). Repetition is only a flag (a 4-gram of
tokens repeated to cover >= 25% of the last quarter of the output) for later adjudication.

    data/fidelity/.venv/bin/python scripts/fidelity/corruption_probe.py \
        --base-url http://HOST:8000 --label N --out data/fidelity/probes/corruption-N.json
"""
import argparse
import json
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ITALIAN = [
    "Scrivi un paragrafo di circa 150 parole sulla città di Perugia, usando molte parole accentate (perché, città, più, già, università).",
    "Spiega a un bambino di otto anni perché il cielo è blu. Usa frasi brevi e parole semplici.",
    "Scrivi una ricetta dettagliata del risotto alla milanese, con dosi per quattro persone e tutti i passaggi.",
    "Descrivi una giornata d'autunno in montagna in Trentino, con almeno dieci aggettivi diversi.",
    "Scrivi una lettera formale al comune per segnalare una buca pericolosa in via Garibaldi.",
    "Riassumi in italiano la trama dei Promessi Sposi in 200 parole.",
    "Elenca dieci modi di dire italiani e spiegane il significato.",
    "Scrivi un dialogo tra una nonna e un nipote che preparano i cappelletti per Natale.",
    "Spiega la differenza tra 'né', 'ne', 'è', 'e' e 'é' con esempi.",
    "Scrivi una breve poesia in rima sul mare della Liguria.",
    "Descrivi i passaggi per iscrivere un figlio alla scuola primaria in Italia.",
    "Scrivi un messaggio di auguri per il compleanno di un collega, tono affettuoso ma professionale.",
    "Spiega cos'è la fotosintesi usando termini semplici ma corretti.",
    "Scrivi un annuncio per vendere una bicicletta usata, con descrizione e prezzo.",
    "Racconta una leggenda popolare del Sud Italia in circa 200 parole.",
    "Scrivi le istruzioni per preparare un caffè con la moka, passo per passo.",
    "Spiega la regola dell'accento su 'perché', 'poiché', 'affinché' e 'benché'.",
    "Scrivi una recensione di un ristorante immaginario a Bologna, con pro e contro.",
    "Descrivi il ciclo dell'acqua per una classe di quarta elementare.",
    "Scrivi un piccolo racconto giallo ambientato a Venezia, al massimo 250 parole.",
    "Traduci in italiano: 'The quick brown fox jumps over the lazy dog, and the city was already asleep.'",
    "Scrivi una lista della spesa per una settimana di cene vegetariane per quattro persone.",
    "Spiega come si calcola la percentuale di sconto su un prezzo, con tre esempi.",
    "Scrivi un'email per chiedere un appuntamento dal medico di base.",
    "Descrivi le principali regioni italiane e una specialità gastronomica per ciascuna.",
    "Scrivi un tema breve sul tema 'Il mio luogo del cuore'.",
    "Spiega perché l'olio d'oliva extravergine è considerato salutare.",
    "Scrivi un indovinello in rima e la sua soluzione.",
    "Descrivi come organizzare una festa di compleanno per bambini in giardino.",
    "Scrivi un breve testo con queste parole: caffè, città, perché, più, così, già, lunedì, virtù.",
    "Spiega le regole del gioco della briscola.",
    "Scrivi una favola con la morale finale, protagonisti una volpe e un corvo.",
    "Scrivi il testo di un cartello per una biblioteca di quartiere che riapre dopo i lavori.",
    "Spiega come si coniuga il verbo 'essere' al passato remoto, con esempi.",
    "Scrivi una descrizione di Roma al tramonto vista dal Gianicolo.",
    "Elenca i passaggi per richiedere la carta d'identità elettronica.",
    "Scrivi un breve discorso per la festa di fine anno scolastico.",
    "Spiega cosa significa 'sostenibilità ambientale' con esempi quotidiani.",
    "Scrivi una filastrocca per imparare i giorni della settimana.",
    "Descrivi il Palio di Siena a qualcuno che non l'ha mai visto.",
]
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
                   "required": ["city"]}}},
    {"type": "function", "function": {
    "name": "create_event", "description": "Create a calendar event",
    "parameters": {"type": "object", "properties": {"title": {"type": "string"}, "date": {"type": "string"},
                   "attendees": {"type": "array", "items": {"type": "string"}}}, "required": ["title", "date"]}}}]
TOOL_PROMPTS = [
    "What is the weather in Milan?", "Che tempo fa a Napoli oggi, in gradi Celsius?",
    "Crea un evento 'Riunione genitori' per il 12 ottobre con Maria e Luca.",
    "Schedule 'Dentist' on 2026-10-03.", "Com'è il meteo a Città di Castello?",
    "Crea un evento 'Cena di compleanno di Niccolò' il 20 novembre con Chiara, Tommaso e Gioia.",
    "Weather in São Paulo in fahrenheit please.", "Che tempo farà a Forlì?",
    "Metti in calendario 'Colloquio con la maestra' il 5 novembre.", "Is it raining in Zürich?",
]


def post(base, body, timeout=1800):
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read()), time.time() - t0


def token_bytes(tokens):
    """Concatenate the byte form of logprob tokens (vLLM returns 'bytes' per token)."""
    out = bytearray()
    for t in tokens:
        b = t.get("bytes")
        if b is None:
            out += t.get("token", "").encode("utf-8", "surrogatepass")
        else:
            out += bytes(b)
    return bytes(out)


def repetition_flag(ids):
    if len(ids) < 64:
        return False
    tail = ids[-len(ids) // 4:]
    grams = Counter(tuple(tail[i:i + 4]) for i in range(len(tail) - 3))
    return bool(grams) and grams.most_common(1)[0][1] * 4 >= 0.25 * len(tail)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=2048)
    a = ap.parse_args()
    with urllib.request.urlopen(a.base_url.rstrip("/") + "/v1/models", timeout=30) as r:
        model = json.loads(r.read())["data"][0]["id"]
    rows = []
    for i, p in enumerate(ITALIAN):
        body = {"model": model, "messages": [{"role": "user", "content": p}], "temperature": 0,
                "max_tokens": a.max_tokens, "logprobs": True, "top_logprobs": 0,
                "chat_template_kwargs": {"reasoning_effort": "low"}}
        r, dt = post(a.base_url, body)
        ch = r["choices"][0]
        toks = (ch.get("logprobs") or {}).get("content") or []
        raw = token_bytes(toks)
        try:
            raw.decode("utf-8"); valid = True
        except UnicodeDecodeError:
            valid = False
        text = ch["message"].get("content") or ""
        rows.append({"kind": "italian", "i": i, "utf8_valid": valid, "fffd": text.count("�"),
                     "tokens": len(toks), "finish": ch.get("finish_reason"),
                     "repetition_flag": repetition_flag([t.get("token") for t in toks]), "seconds": round(dt, 2)})
    for i, p in enumerate(TOOL_PROMPTS):
        body = {"model": model, "messages": [{"role": "user", "content": p}], "temperature": 0,
                "max_tokens": 512, "tools": TOOLS, "tool_choice": "auto"}
        r, dt = post(a.base_url, body)
        msg = r["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        ok = bool(calls)
        for c in calls:
            try:
                json.loads(c["function"]["arguments"])
            except (ValueError, TypeError, KeyError):
                ok = False
        rows.append({"kind": "tool", "i": i, "tool_call": bool(calls), "args_parse_ok": ok,
                     "tool_names": [c["function"]["name"] for c in calls], "fffd": (msg.get("content") or "").count("�"),
                     "seconds": round(dt, 2)})
    it = [r for r in rows if r["kind"] == "italian"]; tl = [r for r in rows if r["kind"] == "tool"]
    summary = {"label": a.label, "italian_prompts": len(it),
               "italian_invalid_utf8": sum(not r["utf8_valid"] for r in it),
               "italian_with_fffd": sum(r["fffd"] > 0 for r in it),
               "italian_fffd_total": sum(r["fffd"] for r in it),
               "italian_repetition_flags": sum(r["repetition_flag"] for r in it),
               "italian_truncated": sum(r["finish"] == "length" for r in it),
               "tool_prompts": len(tl), "tool_calls_made": sum(r["tool_call"] for r in tl),
               "tool_parse_failures": sum(not r["args_parse_ok"] for r in tl if r["tool_call"]),
               "tool_no_call": sum(not r["tool_call"] for r in tl)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1) + "\n")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
