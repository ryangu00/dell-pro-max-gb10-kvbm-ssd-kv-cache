#!/usr/bin/env python3
"""KVBM tier probe: send the same long document twice with N distinct documents in between, and judge the
second answer three ways — TTFT, per-tier KVBM counters, and answer correctness.

  kvbm_probe.py <base_url> <model> [--tokens 30000] [--evict 4] [--thinking off] [--metrics URL] [--out FILE]

Exit code: 0 = all requests completed and A2 answered correctly; 2 = A2 answered wrongly; 3 = a request failed.
TTFT = monotonic wall time to the first streamed content token (temperature 0). Prompt size is calibrated with one
request (tokens/word for this vocabulary) instead of guessed.
"""
import json, sys, time, argparse, random, urllib.request, urllib.error, re

ap = argparse.ArgumentParser()
ap.add_argument("base"); ap.add_argument("model")
ap.add_argument("--tokens", type=int, default=30000, help="approx prompt tokens (must fit the engine's max-model-len)")
ap.add_argument("--evict", type=int, default=4, help="distinct documents sent between A1 and A2")
ap.add_argument("--thinking", default="off", choices=["on", "off"])
ap.add_argument("--thinking-kwarg", default="enable_thinking", help="chat_template_kwargs key: enable_thinking (Qwen) or thinking (DeepSeek)")
ap.add_argument("--metrics", default="", help="KVBM metrics URL (default: <base without /v1>/metrics)")
ap.add_argument("--out", default="kvbm_probe.json"); ap.add_argument("--label", default="")
a = ap.parse_args()
if a.tokens <= 0 or a.evict < 0: sys.exit("--tokens must be > 0 and --evict >= 0")

WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa "
         "quebec romeo sierra tango uniform victor whiskey xray yankee zulu").split()
TOK_PER_WORD = 1.33  # replaced by calibrate()

def doc(seed, ntok):
    """Deterministic document: '[doc SEED] word1234 word5678 ...' + the question. The first code word is the oracle."""
    r = random.Random(seed); n = max(1, int(ntok / TOK_PER_WORD))
    words = [r.choice(WORDS) + str(r.randint(0, 9999)) for _ in range(n)]
    body = " ".join(words)
    return (f"[doc {seed}] " + body + "\n\nQuestion: what is the first code word in this document? Answer with the word only."), words[0]

def post(body, timeout):
    req = urllib.request.Request(f"{a.base}/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)

def call(prompt):
    body = {"model": a.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 24, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": {a.thinking_kwarg: a.thinking == "on"}}
    t0 = time.monotonic(); ttft = None; text = ""; usage = None; finish = None; done = False; err = None
    try:
        with post(body, 3600) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:"): continue
                d = line[5:].strip()
                if d == "[DONE]": done = True; break
                j = json.loads(d)
                if j.get("usage"): usage = j["usage"]
                for c in j.get("choices", []):
                    if c.get("finish_reason"): finish = c["finish_reason"]
                    delta = (c.get("delta") or {}).get("content") or ""
                    if delta:
                        if ttft is None: ttft = time.monotonic() - t0
                        text += delta
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
        err = f"{type(e).__name__}: {e}"
    return {"ttft": round(ttft, 3) if ttft is not None else None, "wall": round(time.monotonic() - t0, 3), "text": text,
            "usage": usage, "finish_reason": finish, "stream_complete": done, "error": err}

def metrics():
    """kvbm_* counters. Labeled series with the same name are summed. Returns {} (not zeros) on failure."""
    url = a.metrics or a.base.replace("/v1", "") + "/metrics"
    out = {}
    try:
        t = urllib.request.urlopen(url, timeout=30).read().decode()
    except Exception as e:
        return {"_error": str(e)}
    for m in re.finditer(r"^(kvbm_[a-z0-9_]+)(?:\{[^}]*\})?\s+([0-9.eE+-]+)$", t, re.M):
        out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(2))
    return out

def delta(after, before, key):
    if key not in after or key not in before: return None
    return after[key] - before[key]

def judge(text, expected_word):
    """Correct iff the first alphabetic token of the answer equals the expected word stem (the model may or may not
    include the numeric suffix). 'not yankee; delta' -> 'not' -> wrong."""
    m = re.match(r"\s*([A-Za-z]+)", text or "")
    return bool(m) and m.group(1).lower() == expected_word.lower()

def calibrate():
    global TOK_PER_WORD
    words = 2000; r = random.Random(7)
    body = " ".join(r.choice(WORDS) + str(r.randint(0, 9999)) for _ in range(words))
    with post({"model": a.model, "messages": [{"role": "user", "content": body}], "max_tokens": 1}, 300) as r_:
        u = json.load(r_)["usage"]["prompt_tokens"]
    TOK_PER_WORD = u / words
    print(f"calibrated: {u} tokens for {words} words -> {TOK_PER_WORD:.2f} tok/word", flush=True)

calibrate()
res = {"label": a.label, "model": a.model, "tokens": a.tokens, "evict": a.evict, "thinking": a.thinking, "steps": []}
prev = metrics()
KEYS = ["kvbm_onboard_blocks_d2d", "kvbm_onboard_blocks_h2d", "kvbm_offload_blocks_d2h", "kvbm_offload_blocks_h2d", "kvbm_matched_tokens"]

def step(name, seed):
    global prev
    prompt, first = doc(seed, a.tokens)
    r = call(prompt); r["step"] = name; r["seed"] = seed
    stem = re.match(r"[a-z]+", first).group(0)
    r["expected_first_code"] = first; r["expected_word"] = stem; r["correct"] = judge(r["text"], stem)
    after = metrics(); r["metrics_after"] = after; r["metrics_delta_this_step"] = {k: delta(after, prev, k) for k in KEYS}; prev = after
    res["steps"].append(r)
    pt = (r["usage"] or {}).get("prompt_tokens")
    print(f"{name}: ttft={r['ttft']} wall={r['wall']} prompt_tokens={pt} finish={r['finish_reason']} correct={r['correct']} "
          f"text={r['text'][:32]!r} delta={r['metrics_delta_this_step']}" + (f" ERROR={r['error']}" if r["error"] else ""), flush=True)
    return r

a1 = step("A1", 1000)
for i in range(a.evict): step(f"E{i+1}", 2000 + i)
a2 = step("A2", 1000)
failed = any(s["error"] or not s["stream_complete"] for s in res["steps"])
res["summary"] = {
    "ttft_A1": a1["ttft"], "ttft_A2": a2["ttft"],
    "ratio_A2_over_A1": round(a2["ttft"] / a1["ttft"], 3) if a1["ttft"] and a2["ttft"] else None,
    "identical": a1["text"] == a2["text"], "correct_A1": a1["correct"], "correct_A2": a2["correct"],
    "expected_word": a1["expected_word"], "text_A1": a1["text"][:60], "text_A2": a2["text"][:60],
    "A2_step_delta": a2["metrics_delta_this_step"], "any_request_failed": failed,
    "verdict": "request_failed" if failed else ("hit_correct" if a2["correct"] and a1["correct"] else "wrong_answer"),
}
json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
print("SUMMARY", json.dumps(res["summary"]))
sys.exit(3 if failed else (0 if res["summary"]["verdict"] == "hit_correct" else 2))
