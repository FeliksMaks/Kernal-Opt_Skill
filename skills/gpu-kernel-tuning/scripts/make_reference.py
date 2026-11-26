"""Greedy reference from a running llama-server: per prompt its ids, the greedy ids and the top-5 logprobs a step.

usage: python make_reference.py --server http://127.0.0.1:8080 --prompts prompts.json --n 48 --out ref.json
prompts.json: ["What is the capital of France? Answer in one sentence.", "Write a Python function ...", ...]
Each prompt is wrapped in the server's chat template (/apply-template), tokenised by the server (/tokenize), and
completed from the ids with temperature 0 and n_probs 5. Start llama-server yourself (any -ngl / -ncmoe), stop it after.
Output entries: {"q", "prompt_ids", "gen", "top": [[[id, logprob] x5] per step], "text"}  (score.py reads this)
"""
import argparse, json, math, urllib.request


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8080")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = []
    for q in json.load(open(a.prompts, encoding="utf-8")):
        tpl = post(a.server + "/apply-template", {"messages": [{"role": "user", "content": q}]})["prompt"]
        ids = post(a.server + "/tokenize", {"content": tpl, "add_special": True, "parse_special": True})["tokens"]
        r = post(a.server + "/completion", {"prompt": ids, "n_predict": a.n, "temperature": 0, "top_k": 1,
                                            "n_probs": 5, "cache_prompt": False, "return_tokens": True,
                                            "post_sampling_probs": False})
        gen, top = [], []
        for p in r.get("completion_probabilities", []):
            gen.append(p["id"])
            top.append([[c["id"], c.get("logprob", math.log(max(c.get("prob", 1e-30), 1e-30)))]
                        for c in p["top_logprobs" if "top_logprobs" in p else "top_probs"]])
        out.append({"q": q, "prompt_ids": ids, "gen": gen, "top": top, "text": r.get("content", "")})
        print(f"{q[:50]!r}: {len(ids)} prompt ids, {len(gen)} generated")
    json.dump(out, open(a.out, "w", encoding="utf-8"))


if __name__ == "__main__":
    main()
