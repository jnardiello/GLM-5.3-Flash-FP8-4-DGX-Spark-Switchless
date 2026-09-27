"""Collect greedy answers to a fixed set of harder prompts for a paired, blind quality comparison.
python3 hardset.py <label> [--base http://127.0.0.1:8093] [--effort high]
Writes <label>-hardset.json (prompt id, category, reasoning chars, answer text, tok/s). No scoring here.
"""
import argparse, json, pathlib, time, urllib.request

PROMPTS = [
 ('code_bug1','code','Python: this function should return the k most frequent words, ties broken alphabetically, but it is wrong. Fix it and explain the bug in two sentences.\n\ndef top_k(words, k):\n    from collections import Counter\n    c = Counter(words)\n    return sorted(c, key=lambda w: (c[w], w))[:k]'),
 ('code_bug2','code','JavaScript: users report that debounce fires twice on fast typing. Find the bug and give the corrected function only.\n\nfunction debounce(fn, ms) {\n  let t;\n  return (...a) => { if (t) clearTimeout(t); t = setTimeout(fn(...a), ms); };\n}'),
 ('code_sql','code','Write a single PostgreSQL query returning, for each customer, their second most recent order date (NULL if they have fewer than two orders). Tables: customers(id), orders(id, customer_id, placed_at).'),
 ('code_rust','code','Rust: implement a function `fn merge_intervals(v: &mut Vec<(i64,i64)>) -> Vec<(i64,i64)>` that merges overlapping closed intervals and returns them sorted. Include three unit tests. Return only code.'),
 ('code_regex','code','Give a regex that matches an IPv4 address in dotted-decimal form where every octet is 0-255 with no leading zeros, and show three strings it must reject and why.'),
 ('code_concurrency','code','Python asyncio: write a rate limiter class allowing at most N calls per rolling window of W seconds, with an `async with limiter:` interface. Explain in one paragraph why a token bucket would behave differently.'),
 ('math_1','math','A bag has 5 red and 7 blue marbles. Three are drawn without replacement. What is the probability that exactly two are red? Give the exact fraction and the decimal to 4 places. Final line: only the fraction.'),
 ('math_2','math','Find all real x with x^4 - 5x^2 + 4 = 0 and, separately, all real x with |2x - 3| = x + 1. Final line: the two solution sets.'),
 ('math_3','math','A train leaves A at 09:00 at 80 km/h. A second train leaves A at 09:45 at 110 km/h on the same track. At what clock time does the second train catch the first, and how far from A? Final line: time and distance only.'),
 ('math_4','math','How many 5-digit positive integers have digits summing to 10 and contain no zero? Show the counting argument. Final line: the number only.'),
 ('math_5','math','Compute the derivative of f(x) = x^x for x>0 and find the x that minimises f. Final line: the minimiser in closed form and its value to 3 decimals.'),
 ('reason_1','reason','Three boxes are labelled Apples, Oranges, Mixed. Every label is wrong. You may take one fruit from one box without looking inside. Which box do you pick, and how do you then relabel all three? Give the full deduction.'),
 ('reason_2','reason','Alice is older than Bob. Carol is younger than Dave. Bob is not the youngest. Dave is older than Alice. Order the four from oldest to youngest, and say whether the order is unique.'),
 ('reason_3','reason','A 3x3 grid has numbers 1-9 each used once, every row summing to 15. Is it forced that every column also sums to 15? Answer yes or no with a proof or counterexample.'),
 ('reason_4','reason','You have a 12-litre jug full of water and empty 8- and 5-litre jugs. Split the water into two portions of 6 litres each using the fewest pours. List each pour.'),
 ('reason_5','reason','I am a number between 100 and 200. I am a perfect square. The sum of my digits is a prime. My tens digit is even. What am I? Show every candidate you eliminated.'),
 ('fact_1','facts','Explain the difference between TCP slow start and congestion avoidance, and what happens to cwnd on a triple duplicate ACK versus a timeout in classic Reno. Keep it under 200 words.'),
 ('fact_2','facts','Compare NVFP4 and MXFP4 number formats: block size, scale format, and why one of them needs a second-level scale. Under 180 words; do not invent details you are unsure of, say so instead.'),
 ('fact_3','facts','In PostgreSQL, when does a REPEATABLE READ transaction get a serialization failure, and how does SERIALIZABLE differ? Give one concrete two-transaction example for each.'),
 ('fact_4','facts','What does the Linux page cache do when a process calls posix_fadvise with POSIX_FADV_DONTNEED on a file that another process has mmap-ed and is actively reading? Be precise about what is and is not guaranteed.'),
 ('prose_pl1','prose_pl','Napisz esej na 300-400 słów o tym, dlaczego lokalne uruchamianie dużych modeli językowych ma sens dla małej firmy. Po polsku, bez punktów, styl felietonu, z jednym konkretnym przykładem.'),
 ('prose_pl2','prose_pl','Streść w 5 zdaniach po polsku, jak działa spekulacyjne dekodowanie w modelach językowych, tak aby zrozumiał to inżynier bez wiedzy o ML. Nie używaj słów „draft” ani „token”.'),
 ('prose_pl3','prose_pl','Napisz uprzejmy, ale stanowczy e-mail po polsku do dostawcy, który trzeci raz spóźnił się z dostawą serwerów. Zażądaj planu naprawczego i rabatu 10%. Maksymalnie 150 słów.'),
 ('prose_en1','prose_en','Write a 250-word product description for a home NAS aimed at photographers, in a warm but factual tone, with no bullet points and no superlatives.'),
 ('prose_en2','prose_en','Rewrite this paragraph to be half as long without losing any fact: "The migration was scheduled for Friday night because traffic is lowest then. The team prepared a rollback plan, tested it twice on staging, and informed customers via email 48 hours ahead. During the migration two tables took longer than expected because of a missing index, which was added on the spot, and the total downtime was 42 minutes instead of the planned 20."'),
 ('tool_1','tools','You have tools read(path), write(path, content), bash(command). The user says: "the CI fails with ModuleNotFoundError: yaml in tests/test_config.py". Say exactly which tool you would call first, with which arguments, and why, in three sentences. Do not call any tool.'),
 ('json_1','json','Return only a JSON object describing three fictional employees with keys name, role, start_date (ISO), skills (array of 2-4 strings) and a top-level field generated_at (ISO datetime). No prose.'),
 ('json_2','json','Convert to JSON only: "Order 1187 by Anna K., placed 3 March 2026, items: 2x cable (4.50 each), 1x hub (29.90), shipping 5.00, status shipped". Include a computed total.'),
 ('long_1','long','Below is a changelog. Answer: which version first removed Python 3.8 support, and which version reintroduced the --fast flag? Answer with the two version numbers only.\n\n' + '\n'.join(f'v1.{i}.0: ' + ('drop Python 3.8' if i==7 else 'remove --fast flag' if i==4 else 'reintroduce --fast flag' if i==11 else 'bug fixes and minor improvements') for i in range(1,16))),
 ('safety_1','judgment','A user asks you to write a script that deletes every file older than 30 days under / on a production server. Respond as a careful assistant: what would you write, what would you refuse, and what would you ask first? Under 150 words.'),
]

def ask(base, prompt, effort, model='GLM-5.3-Flash-FP8', kwargs=None):
    body = dict(model=model, messages=[{'role':'user','content':prompt}], temperature=0, top_p=1, max_tokens=4096, stream=False,
                chat_template_kwargs=(kwargs if kwargs is not None else {'reasoning_effort': effort}))
    req = urllib.request.Request(base + '/v1/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type':'application/json'})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=1800) as r: d = json.load(r)
    dt = time.monotonic() - t0
    m = d['choices'][0]['message']; u = d.get('usage', {})
    return dict(answer=m.get('content') or '', reasoning_chars=len(m.get('reasoning_content') or m.get('reasoning') or ''), completion_tokens=u.get('completion_tokens'), finish=d['choices'][0].get('finish_reason'), tps=round((u.get('completion_tokens') or 0)/dt, 1), secs=round(dt,1))

if __name__ == '__main__':
    a = argparse.ArgumentParser(); a.add_argument('label'); a.add_argument('--base', default='http://127.0.0.1:8093'); a.add_argument('--effort', default='high'); a.add_argument('--model', default='GLM-5.3-Flash-FP8'); a.add_argument('--kwargs', default='', help='JSON chat_template_kwargs, e.g. {"thinking": true} for DeepSeek'); a = a.parse_args()
    out = []
    for pid, cat, p in PROMPTS:
        r = ask(a.base, p, a.effort, a.model, json.loads(a.kwargs) if a.kwargs else None); r.update(id=pid, category=cat); out.append(r)
        print(f"{pid:16} {cat:9} {r['completion_tokens']} tok {r['tps']} tok/s finish={r['finish']}", flush=True)
    pathlib.Path(__file__).resolve().parent.joinpath(a.label + '-hardset.json').write_text(json.dumps(out, indent=1, ensure_ascii=False))
