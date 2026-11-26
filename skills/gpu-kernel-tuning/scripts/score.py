"""Teacher-forced correctness gate: does the engine pick llama.cpp's greedy token at every step?

usage: python score.py --ref ref.json --cmd "ENGINE --pack X --max-context 512 --score {P} --decode {ids}" [--log-dir D]
{P}: prompt length, {ids}: prompt + greedy ids comma-separated. The engine prints, for every scored position,
  S <pos> <next_id> <logprob_of_next> <rank_of_next> <id>:<lp> <id>:<lp> ...   (top-5)
Pass: >= 98% agreement, misses near ties (gap top1 - next < ~0.5). Exit code 0 always; read the TOTAL line.
Env vars of this process pass through to the engine (use them for A/B switches).
"""
import argparse, json, os, shlex, subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cmd", required=True)
    ap.add_argument("--log-dir", default=".")
    ap.add_argument("--tag", default="score")
    a = ap.parse_args()
    ref = json.load(open(a.ref, encoding="utf-8"))
    tot = agree_all = 0
    gaps = []
    for qi, r in enumerate(ref):
        ids = r["prompt_ids"] + r["gen"]
        P = len(r["prompt_ids"])
        cmd = a.cmd.replace("{P}", str(P)).replace("{ids}", ",".join(map(str, ids)))
        log = os.path.join(a.log_dir, f"{a.tag}-{qi}.log")
        with open(log, "w", encoding="utf-8", errors="replace") as f:
            subprocess.run(cmd if os.name == "nt" else shlex.split(cmd), stdout=f, stderr=subprocess.STDOUT)
        lines = [l for l in open(log, errors="replace") if l.startswith("S ")]
        agree = 0
        for l in lines:
            f = l.split()
            pos, nxt, lp, rank = int(f[1]), int(f[2]), float(f[3]), int(f[4])
            top = [(int(x.split(":")[0]), float(x.split(":")[1])) for x in f[5:]]
            step = pos + 1 - P
            agree += rank == 0
            if rank != 0:
                gaps.append(top[0][1] - lp)
                llp = r["top"][step][0][1] if step < len(r["top"]) else float("nan")
                print(f"    step {step:2d} want {nxt} (llama lp {llp:.3f}, engine lp {lp:.3f}, rank {rank}); "
                      f"engine top1 {top[0][0]}:{top[0][1]:.3f}")
        tot += len(lines)
        agree_all += agree
        print(f"Q{qi} {r['q'][:50]!r}: {agree}/{len(lines)}" + ("  (NO S LINES - see " + log + ")" if not lines else ""))
    g = sorted(gaps)
    print(f"TOTAL agree {agree_all}/{tot}; median gap on misses {g[len(g) // 2] if g else 0:.3f}")


if __name__ == "__main__":
    main()
