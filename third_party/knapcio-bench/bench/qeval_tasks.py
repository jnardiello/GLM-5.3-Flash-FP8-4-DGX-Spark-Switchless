"""Task set for the DS-V4.1-Flash quality gate. Auto-scored, no LLM judge.

Every task is (id, category, thinking, max_tokens, prompt, checker). A checker takes the
assistant's content and returns (ok: bool, reason: str). Checkers must be deterministic and
must not depend on wording -- only on what the prompt actually demands.
"""
import json, re, subprocess, sys, tempfile, os

# ---------------------------------------------------------------- extractors

def extract_code(text):
    m = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    if m:
        return max(m, key=len)
    return text


def extract_json(text):
    m = re.findall(r"```(?:json)?\s*\n(.*?)```", text, re.S)
    for blob in (m or []) + [text]:
        blob = blob.strip()
        i, j = blob.find("{"), blob.rfind("}")
        if i >= 0 and j > i:
            try:
                return json.loads(blob[i:j + 1])
            except Exception:
                continue
    return None


_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_final_number(text):
    """The last standalone number, preferring one after an 'answer'-ish marker."""
    tail = text
    for marker in ("final answer", "answer:", "answer is", "**", "="):
        idx = text.lower().rfind(marker)
        if idx >= 0:
            tail = text[idx:]
            break
    nums = _NUM.findall(tail) or _NUM.findall(text)
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", ""))
    except ValueError:
        return None


def run_python(code, harness, timeout=20):
    """Run model code plus an assert harness in an isolated interpreter. Returns (ok, reason).

    The tasks are benign algorithmic prompts written here, but this does execute generated
    code: isolated mode (-I, no user site / PYTHONPATH), a scratch cwd, a hard timeout, no shell.
    """
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cand.py")
        with open(path, "w") as f:
            f.write(code + "\n\n" + harness + "\nprint('HARNESS_OK')\n")
        try:
            p = subprocess.run([sys.executable, "-I", "-S", path], cwd=d, timeout=timeout,
                               capture_output=True, text=True)
        except subprocess.TimeoutExpired:
            return False, "timeout"
    if "HARNESS_OK" in p.stdout:
        return True, ""
    err = (p.stderr or p.stdout or "").strip().splitlines()
    return False, (err[-1][:120] if err else "no output")


def code_task(harness):
    return lambda text: run_python(extract_code(text), harness)


def json_task(check):
    def inner(text):
        obj = extract_json(text)
        if obj is None:
            return False, "no parseable JSON object"
        try:
            return check(obj)
        except Exception as exc:
            return False, f"checker raised {exc!r}"
    return inner


def number_task(expected, tol=1e-6):
    def inner(text):
        got = extract_final_number(text)
        if got is None:
            return False, "no number found"
        return (abs(got - expected) <= tol), f"got {got}, want {expected}"
    return inner


def contains_task(*needles, forbid=()):
    def inner(text):
        low = text.lower()
        for n in needles:
            if n.lower() not in low:
                return False, f"missing {n!r}"
        for n in forbid:
            if n.lower() in low:
                return False, f"contains forbidden {n!r}"
        return True, ""
    return inner


def degeneration_check(min_words, max_words, forbid_headings=True):
    """Prose is not judged for quality -- only for the failure modes that are objective."""
    def inner(text):
        words = text.split()
        if not (min_words <= len(words) <= max_words):
            return False, f"{len(words)} words, want {min_words}-{max_words}"
        if forbid_headings and re.search(r"^\s*#{1,6}\s", text, re.M):
            return False, "markdown heading present"
        non_ascii = sum(1 for c in text if ord(c) > 0x2100)
        if non_ascii > len(text) * 0.02:
            return False, f"{non_ascii} exotic glyphs"
        low = [w.lower().strip(".,;:!?") for w in words]
        for n in (4, 6):                                   # verbatim n-gram loops
            grams = [tuple(low[i:i + n]) for i in range(len(low) - n)]
            if grams and (len(grams) - len(set(grams))) > max(2, len(grams) * 0.05):
                return False, f"repeated {n}-grams"
        return True, ""
    return inner


# ---------------------------------------------------------------- the tasks
T = []
def add(tid, cat, thinking, max_tokens, prompt, checker):
    T.append(dict(id=tid, category=cat, thinking=thinking, max_tokens=max_tokens,
                  prompt=prompt, checker=checker))


_CODE = [
    ("merge_intervals", "merge_intervals(intervals)", "merges overlapping closed integer intervals given as a list of [start, end] pairs and returns them sorted by start",
     "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]\n"
     "assert merge_intervals([]) == []\nassert merge_intervals([[1,4],[4,5]]) == [[1,5]]"),
    ("longest_run", "longest_run(xs)", "returns the length of the longest strictly increasing contiguous run in a list of ints",
     "assert longest_run([1,2,3,1,2]) == 3\nassert longest_run([]) == 0\nassert longest_run([5,4,3]) == 1"),
    ("roman_to_int", "roman_to_int(s)", "converts a Roman numeral string to an int",
     "assert roman_to_int('MCMXCIV') == 1994\nassert roman_to_int('IX') == 9\nassert roman_to_int('LVIII') == 58"),
    ("balanced", "balanced(s)", "returns True iff the brackets ()[]{} in the string are balanced and correctly nested",
     "assert balanced('([]{})') is True\nassert balanced('([)]') is False\nassert balanced('') is True"),
    ("flatten", "flatten(xs)", "flattens an arbitrarily nested list of ints into a flat list, preserving order",
     "assert flatten([1,[2,[3,[4]]],5]) == [1,2,3,4,5]\nassert flatten([]) == []\nassert flatten([[],[1]]) == [1]"),
    ("top_words", "top_words(text, n)", "returns the n most frequent lowercase alphabetic words as a list of (word, count), ties broken alphabetically",
     "assert top_words('a b a c b a', 2) == [('a',3),('b',2)]\n"
     "assert top_words('x y', 2) == [('x',1),('y',1)]"),
    ("bisect_left", "bisect_left(xs, v)", "returns the leftmost index at which v can be inserted into the sorted list xs keeping it sorted",
     "assert bisect_left([1,2,2,3], 2) == 1\nassert bisect_left([], 5) == 0\nassert bisect_left([1,3], 4) == 2"),
    ("transpose", "transpose(m)", "transposes a rectangular matrix given as a list of equal-length lists",
     "assert transpose([[1,2,3],[4,5,6]]) == [[1,4],[2,5],[3,6]]\nassert transpose([]) == []"),
    ("rle", "rle(s)", "run-length encodes a string as a list of (char, count) pairs",
     "assert rle('aaabbc') == [('a',3),('b',2),('c',1)]\nassert rle('') == []"),
    ("is_palindrome", "is_palindrome(s)", "returns True iff the string is a palindrome ignoring case and non-alphanumeric characters",
     "assert is_palindrome('A man, a plan, a canal: Panama') is True\nassert is_palindrome('ab') is False\nassert is_palindrome('') is True"),
    ("chunk", "chunk(xs, k)", "splits a list into consecutive chunks of size k, the last one possibly shorter",
     "assert chunk([1,2,3,4,5], 2) == [[1,2],[3,4],[5]]\nassert chunk([], 3) == []"),
    ("dedupe", "dedupe(xs)", "removes duplicates from a list while preserving first-occurrence order",
     "assert dedupe([3,1,3,2,1]) == [3,1,2]\nassert dedupe([]) == []"),
    ("camel_to_snake", "camel_to_snake(s)", "converts a CamelCase or camelCase identifier to snake_case",
     "assert camel_to_snake('parseHTTPResponse') == 'parse_http_response'\n"
     "assert camel_to_snake('Simple') == 'simple'\nassert camel_to_snake('a') == 'a'"),
    ("divisor_sum", "divisor_sum(n)", "returns the sum of all positive divisors of n including n itself",
     "assert divisor_sum(28) == 56\nassert divisor_sum(1) == 1\nassert divisor_sum(12) == 28"),
    ("levenshtein", "levenshtein(a, b)", "returns the Levenshtein edit distance between two strings",
     "assert levenshtein('kitten','sitting') == 3\nassert levenshtein('','abc') == 3\nassert levenshtein('x','x') == 0"),
]
_CODE += [
    ("group_anagrams", "group_anagrams(words)", "groups words that are anagrams of each other; returns a list of groups, each group sorted alphabetically, and the groups sorted by their first element",
     "assert group_anagrams(['eat','tea','tan','ate','nat','bat']) == [['ate','eat','tea'],['bat'],['nat','tan']]\n"
     "assert group_anagrams([]) == []"),
    ("two_sum", "two_sum(xs, target)", "returns the pair of indices (i, j) with i < j such that xs[i] + xs[j] == target, or None if there is none; if several exist return the one with the smallest i then smallest j",
     "assert two_sum([2,7,11,15], 9) == (0,1)\nassert two_sum([3,2,4], 6) == (1,2)\nassert two_sum([1,2], 99) is None"),
    ("rotate90", "rotate90(m)", "rotates a square matrix 90 degrees clockwise and returns a new matrix",
     "assert rotate90([[1,2],[3,4]]) == [[3,1],[4,2]]\nassert rotate90([]) == []\n"
     "assert rotate90([[1,2,3],[4,5,6],[7,8,9]]) == [[7,4,1],[8,5,2],[9,6,3]]"),
    ("valid_ipv4", "valid_ipv4(s)", "returns True iff the string is a valid dotted-quad IPv4 address with no leading zeros and each octet in 0-255",
     "assert valid_ipv4('192.168.0.1') is True\nassert valid_ipv4('256.1.1.1') is False\n"
     "assert valid_ipv4('01.2.3.4') is False\nassert valid_ipv4('1.2.3') is False"),
    ("parse_query", "parse_query(qs)", "parses a URL query string into a dict; repeated keys collect into a list in order; a key with no '=' maps to the empty string",
     "assert parse_query('a=1&b=2') == {'a':'1','b':'2'}\n"
     "assert parse_query('a=1&a=2') == {'a':['1','2']}\nassert parse_query('') == {}\nassert parse_query('flag') == {'flag':''}"),
    ("kth_largest", "kth_largest(xs, k)", "returns the k-th largest element (k=1 means the maximum), counting duplicates as distinct positions; returns None if k is out of range",
     "assert kth_largest([3,1,4,1,5], 2) == 4\nassert kth_largest([3,3,3], 3) == 3\nassert kth_largest([1], 2) is None"),
    ("spiral", "spiral(m)", "returns the elements of a rectangular matrix in clockwise spiral order starting from the top-left",
     "assert spiral([[1,2,3],[4,5,6],[7,8,9]]) == [1,2,3,6,9,8,7,4,5]\n"
     "assert spiral([]) == []\nassert spiral([[1,2]]) == [1,2]"),
    ("word_wrap", "word_wrap(text, width)", "greedily wraps a space-separated string into lines of at most `width` characters without splitting words; returns a list of lines",
     "assert word_wrap('the quick brown fox', 10) == ['the quick','brown fox']\n"
     "assert word_wrap('', 5) == []\nassert word_wrap('abc', 2) == ['abc']"),
    ("interval_intersect", "interval_intersect(a, b)", "returns the intersection of two lists of disjoint sorted closed integer intervals",
     "assert interval_intersect([[0,2],[5,10]],[[1,5],[8,12]]) == [[1,2],[5,5],[8,10]]\n"
     "assert interval_intersect([],[[1,2]]) == []"),
    ("base_to_int", "base_to_int(s, base)", "converts a string in the given base (2-36, digits 0-9 then a-z, case-insensitive) to an int; raises ValueError on an invalid digit",
     "assert base_to_int('ff', 16) == 255\nassert base_to_int('1010', 2) == 10\nassert base_to_int('z', 36) == 35\n"
     "try:\n    base_to_int('2', 2); raise AssertionError('should have raised')\nexcept ValueError: pass"),
]

for tid, sig, desc, harness in _CODE:
    add(f"code_{tid}", "code", False, 420,
        f"Write a Python function `{sig}` that {desc}. Handle empty input sensibly. "
        f"Return ONLY the function in a single ```python code block, no explanation, no tests, "
        f"no imports beyond the standard library.",
        code_task(harness))

add("json_manifest", "json", False, 320,
    'Return ONLY a JSON object (no prose, no code fence text outside it) with exactly these keys: '
    '"name" (string), "version" (string of the form X.Y.Z), "deps" (array of exactly 3 strings), '
    '"stable" (boolean), "max_workers" (integer between 2 and 8). Invent plausible values for a build tool.',
    json_task(lambda o: (
        (set(o) == {"name", "version", "deps", "stable", "max_workers"}
         and isinstance(o["name"], str) and re.fullmatch(r"\d+\.\d+\.\d+", str(o["version"])) is not None
         and isinstance(o["deps"], list) and len(o["deps"]) == 3 and all(isinstance(x, str) for x in o["deps"])
         and isinstance(o["stable"], bool)
         and isinstance(o["max_workers"], int) and 2 <= o["max_workers"] <= 8),
        f"got keys {sorted(o)}")))
add("json_nested", "json", False, 360,
    'Return ONLY a JSON object describing a server: key "host" (string), "port" (integer 1024-65535), '
    '"tls" (object with keys "enabled" (bool) and "min_version" (string)), '
    '"routes" (array of exactly 2 objects, each with "path" (string starting with /) and "methods" (array of strings)).',
    json_task(lambda o: (
        (isinstance(o.get("host"), str) and isinstance(o.get("port"), int) and 1024 <= o["port"] <= 65535
         and isinstance(o.get("tls"), dict) and isinstance(o["tls"].get("enabled"), bool)
         and isinstance(o["tls"].get("min_version"), str)
         and isinstance(o.get("routes"), list) and len(o["routes"]) == 2
         and all(isinstance(r, dict) and str(r.get("path", "")).startswith("/")
                 and isinstance(r.get("methods"), list) for r in o["routes"])),
        "schema mismatch")))
add("json_toolcall", "json", False, 280,
    'The user says: "book a table for four at 7pm tomorrow at Kappa". Return ONLY a JSON object '
    '{"name": "<tool name>", "arguments": {...}} for a restaurant booking tool. The arguments must '
    'include a party size as an integer 4 and a time as a string.',
    json_task(lambda o: (
        (isinstance(o.get("name"), str) and isinstance(o.get("arguments"), dict)
         and 4 in [v for v in o["arguments"].values() if isinstance(v, int)]
         and any(isinstance(v, str) and re.search(r"\d", v) for v in o["arguments"].values())),
        "no int 4 and/or no time-ish string in arguments")))
add("json_escape", "json", False, 280,
    'Return ONLY a JSON object with key "text" whose value is exactly this sentence including the '
    'quotes and backslash: She said "hi\\there" — twice. And a key "len" whose integer value is the '
    'number of characters in that string.',
    json_task(lambda o: (("text" in o and isinstance(o.get("len"), int)
                          and o["len"] == len(o["text"])), "len does not match text length")))
add("json_sort", "json", False, 320,
    'Return ONLY a JSON array of the following objects sorted by "score" descending, then "name" '
    'ascending: [{"name":"bo","score":3},{"name":"al","score":7},{"name":"cy","score":3},{"name":"di","score":9}]',
    lambda text: ((lambda o: (o == [{"name": "di", "score": 9}, {"name": "al", "score": 7},
                                    {"name": "bo", "score": 3}, {"name": "cy", "score": 3}], "wrong order"))
                  (json.loads(re.search(r"\[.*\]", text, re.S).group(0))) if re.search(r"\[.*\]", text, re.S)
                  else (False, "no JSON array")))
add("json_types", "json", False, 260,
    'Return ONLY a JSON object with keys "a" (the integer 0), "b" (the boolean false), '
    '"c" (null), "d" (the empty array), "e" (the empty string). Nothing else.',
    json_task(lambda o: ((o == {"a": 0, "b": False, "c": None, "d": [], "e": ""}), f"got {o}")))
add("json_count", "json", False, 300,
    'Count the vowels (a,e,i,o,u, case-insensitive) in the string "Encyclopaedia Britannica" and '
    'return ONLY {"vowels": <integer>}.',
    json_task(lambda o: ((o.get("vowels") == 10), f"got {o.get('vowels')}, want 10")))
add("json_strict_empty", "json", False, 260,
    'Return ONLY a JSON object mapping each of the words "alpha", "beta", "gamma" to its length '
    'as an integer. No other keys.',
    json_task(lambda o: ((o == {"alpha": 5, "beta": 4, "gamma": 5}), f"got {o}")))

_MATH = [
    ("m1", "Compute 17 * 243 + 5**4 - 1000. Show your steps, then give the final number on its own line.", 17 * 243 + 625 - 1000),
    ("m2", "A train travels 315 km in 3.5 hours, then 210 km in 2 hours. What is its average speed in km/h over the whole journey? Give the final number on its own line.", (315 + 210) / 5.5),
    ("m3", "What is the sum of all integers from 1 to 200 that are divisible by 7? Final number on its own line.", sum(i for i in range(1, 201) if i % 7 == 0)),
    ("m4", "A rectangle has perimeter 46 and area 120. What is the length of its longer side? Final number on its own line.", 15),
    ("m5", "Compute the greatest common divisor of 1071 and 462. Final number on its own line.", 21),
    ("m6", "If 3 painters paint 5 rooms in 8 hours, how many hours do 5 painters need for 12 rooms at the same rate? Final number on its own line.", 11.52),
    ("m7", "What is 2**20 - 3**10? Final number on its own line.", 2 ** 20 - 3 ** 10),
    ("m8", "A shirt costs 80 after a 20% discount and then a further 10% off at the till. What was the original price? Final number on its own line.", 80 / (0.8 * 0.9)),
    ("m9", "How many distinct 4-letter strings can be made from the letters of BANANA (using each letter at most as often as it appears)? Final number on its own line.", 38),
    ("m10", "The sequence is 2, 6, 12, 20, 30, ... What is the 12th term? Final number on its own line.", 12 * 13),
]
_MATH += [
    ("m11", "What is 7**100 mod 13? Final number on its own line.", 9),
    ("m12", "In how many distinct ways can 8 people be seated around a round table, treating rotations as identical? Final number on its own line.", 5040),
    ("m13", "A cone has radius 3 and height 4. What is its total surface area, using pi = 3.14159, to two decimal places? Final number on its own line.", 75.40),
    ("m14", "Two pipes fill a tank in 12 and 18 minutes respectively. A drain empties it in 36 minutes. With all three open, how many minutes to fill the tank? Final number on its own line.", 9),
]

for tid, prompt, want in _MATH:
    add(f"math_{tid}", "math", False, 640, prompt, number_task(float(want), tol=0.01))

_REASON = [
    ("r1", "A farmer has 17 sheep. All but 9 run away. He then buys twice as many as remain, and sells 4. How many sheep does he have? Give the final number on its own line.", 23),
    ("r2", "Three boxes are labelled APPLES, ORANGES and MIXED. Every label is wrong. You may draw one fruit from one box. From which labelled box should you draw to deduce all three contents? Answer with the single label word in capitals on its own line.", "MIXED"),
    ("r3", "It takes 5 machines 5 minutes to make 5 widgets. How many minutes would 100 machines take to make 100 widgets? Final number on its own line.", 5),
    ("r4", "A bat and a ball cost 1.10 together. The bat costs 1.00 more than the ball. What does the ball cost? Final number on its own line.", 0.05),
    ("r5", "In a race you overtake the runner in second place. What place are you in now? Final number on its own line.", 2),
    ("r6", "A lily pad patch doubles in size every day and covers the lake on day 48. On which day did it cover half the lake? Final number on its own line.", 47),
    ("r7", "You have 8 identical-looking balls, one heavier. Using a balance scale, what is the minimum number of weighings guaranteed to find it? Final number on its own line.", 2),
    ("r8", "Anna is twice as old as Ben was when Anna was as old as Ben is now. Anna is 24. How old is Ben? Final number on its own line.", 18),
    ("r9", "A clock shows 3:15. What is the angle in degrees between the hour and minute hands? Final number on its own line.", 7.5),
    ("r10", "Five houses in a row. The green house is immediately right of the ivory one. The Norwegian lives in the first house. Milk is drunk in the middle house. If the green house is the fourth, which position is the ivory house in? Final number on its own line.", 3),
]
_REASON += [
    ("r11", "A man must ferry a wolf, a goat and a cabbage across a river. The boat holds him and one item. The wolf eats the goat if left alone with it; the goat eats the cabbage if left alone with it. What is the minimum number of river crossings (counting each one-way trip) to get all three across? Final number on its own line.", 7),
    ("r12", "You have 12 identical-looking balls, exactly one of which has a different weight (you do not know whether heavier or lighter). Using a balance scale, what is the minimum number of weighings that is guaranteed to identify it? Final number on its own line.", 3),
    ("r13", "On an island every inhabitant is either a knight who always tells the truth or a knave who always lies. A says 'B is a knave'. B says 'A and I are of the same type'. How many knights are there among A and B? Final number on its own line.", 1),
    ("r14", "A bag holds 3 red and 5 blue marbles. Two are drawn without replacement. What is the probability that both are red? Give the answer as a decimal to three places on its own line.", 0.107),
    ("r15", "Today is Wednesday. What day of the week will it be in 1000 days? Answer with the single day name in capitals on its own line.", "TUESDAY"),
    ("r16", "Four people must cross a bridge at night with one torch; the bridge holds two at a time and whoever crosses must carry the torch. Their individual crossing times are 1, 2, 7 and 10 minutes; a pair moves at the slower person's pace. What is the minimum total time in minutes? Final number on its own line.", 17),
]

for tid, prompt, want in _REASON:
    add(f"reason_{tid}", "reason", True, 1400, prompt,
        number_task(float(want), tol=0.01) if isinstance(want, (int, float))
        else contains_task(want))

add("fmt_table", "format", False, 420,
    "Produce a markdown table with exactly the columns | step | command | why | and exactly three "
    "data rows, describing how to open an SSH tunnel to a remote API on port 8888. Output the table "
    "and nothing else.",
    lambda t: ((len([l for l in t.strip().splitlines() if l.strip().startswith("|")]) == 5,
                f"{len([l for l in t.strip().splitlines() if l.strip().startswith('|')])} pipe lines, want 5")))
add("fmt_bullets", "format", False, 360,
    "List exactly five differences between TCP and UDP as markdown bullets starting with '- '. "
    "No heading, no intro, no closing sentence.",
    lambda t: ((len([l for l in t.strip().splitlines() if l.strip().startswith("- ")]) == 5,
                "wrong bullet count")))
add("fmt_nocode", "format", False, 300,
    "Explain what a mutex is in exactly two sentences. Do not use any code, any markdown formatting, "
    "or the word 'lock'.",
    lambda t: ((len([s for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s.strip()]) == 2
                and "lock" not in t.lower() and "`" not in t and "*" not in t, "constraint violated")))
add("fmt_uppercase", "format", False, 260,
    "Reply with the five largest planets of the Solar System, largest first, as a single "
    "comma-separated line in ALL CAPS. Nothing else.",
    lambda t: ((t.strip().upper() == t.strip() and t.strip().count(",") == 4
                and "JUPITER" in t.upper() and t.strip().upper().startswith("JUPITER"), "format/content wrong")))
add("fmt_exactword", "format", False, 300,
    "Answer with exactly one word: what is the capital of Australia?",
    lambda t: ((len(t.strip().split()) == 1 and t.strip().strip(".").lower() == "canberra",
                f"got {t.strip()!r}")))
add("fmt_json_only", "format", False, 280,
    "Output ONLY the JSON array [1, 2, 3]. No fence, no prose.",
    lambda t: ((t.strip() == "[1, 2, 3]" or t.strip() == "[1,2,3]", f"got {t.strip()[:40]!r}")))
add("fmt_linecount", "format", False, 360,
    "Write exactly four lines, each being a single English word that rhymes with 'light'. "
    "No numbering, no punctuation, nothing else.",
    lambda t: ((len([l for l in t.strip().splitlines() if l.strip()]) == 4
                and all(len(l.split()) == 1 for l in t.strip().splitlines() if l.strip()), "not four bare words")))

_PROSE = [
    ("p1", "Write one paragraph of about 120 words describing a harbour at dawn. Prose only, no headings.", 80, 190),
    ("p2", "Tell a 150-word story about a clockmaker who refuses to sell her last clock. Prose only.", 100, 230),
    ("p3", "In about 100 words, explain to a non-programmer what a cache is, using one everyday analogy. Prose only.", 70, 165),
    ("p4", "Write about 130 words of continuous prose arguing that maps are always political. No headings, no bullets.", 90, 200),
    ("p5", "Describe, in roughly 120 words of plain prose, the experience of reading a letter written forty years ago.", 80, 190),
]
for tid, prompt, lo, hi in _PROSE:
    add(f"prose_{tid}", "prose", False, 460, prompt, degeneration_check(lo, hi))

TASKS = T
