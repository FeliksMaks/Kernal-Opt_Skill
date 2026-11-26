#!/usr/bin/env python3
"""Alternating A/B of engine commands, one fresh process per run.

  python ab.py --cmd "engine --model M --tokens @p.txt --gen 200" --rounds 3 ^
      --speed "generation: .*?\\(([\\d.]+) tok/s\\)" --prompt "prompt: .*?\\(([\\d.]+) tok/s\\)" ^
      --ids "generated: (.*)" base new:MY_SWITCH=1 old:MY_SWITCH=0,OTHER=2

Each config is NAME or NAME:VAR=value,VAR=value (added to the environment). An unmeasured run of each config comes first (file
cache, first compile). The order is reversed every second round. Prints every value, the median, the spread, and whether the
ids equal those of the first config, in every round. --exe NAME=PATH replaces the program of config NAME.
Exit code 1 if a run failed."""
import argparse, os, re, shlex, statistics, subprocess, sys


def split(cmd):
    parts = shlex.split(cmd, posix=os.name != "nt")
    return [p[1:-1] if len(p) > 1 and p[0] == p[-1] and p[0] in "\"'" else p for p in parts]


def run(cmd, env, timeout):
    try:
        p = subprocess.run(cmd, env={**os.environ, **env}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    except OSError as e:
        return None, str(e)
    out = p.stdout.decode("utf-8", "replace")
    return (out, "") if p.returncode == 0 else (None, f"exit {p.returncode}: {out[-400:]}")


def find(rx, out):
    m = re.search(rx, out, re.M) if rx else None
    return m.group(1) if m else None


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--cmd", required=True)
    a.add_argument("--speed", required=True, help="regex, group 1 = decode tok/s")
    a.add_argument("--prompt", help="regex, group 1 = prompt tok/s")
    a.add_argument("--ids", help="regex, group 1 = generated ids")
    a.add_argument("--rounds", type=int, default=3)
    a.add_argument("--timeout", type=float, default=900)
    a.add_argument("--exe", action="append", default=[], help="NAME=PATH")
    a.add_argument("configs", nargs="+")
    o = a.parse_args()
    exes = dict(e.split("=", 1) for e in o.exe)
    cfg = []
    for c in o.configs:
        name, _, rest = c.partition(":")
        cmd = split(o.cmd)
        if name in exes:
            cmd[0] = exes[name]
        cfg.append((name, cmd, dict(kv.split("=", 1) for kv in rest.split(",") if kv)))
    if len({c[0] for c in cfg}) != len(cfg):
        sys.exit("config names must differ")
    if set(exes) - {c[0] for c in cfg}:
        sys.exit(f"--exe names no config: {sorted(set(exes) - {c[0] for c in cfg})}")
    failed, got = False, {c[0]: [] for c in cfg}
    for r in range(o.rounds + 1):
        for name, cmd, env in (cfg if r % 2 == 0 else cfg[::-1]):
            out, err = run(cmd, env, o.timeout)
            v = (find(o.speed, out), find(o.prompt, out), find(o.ids, out)) if out is not None else None
            if v is None or v[0] is None or (o.prompt and v[1] is None) or (o.ids and v[2] is None):
                print(f"FAILED {name} round {r}: {err or 'a pattern did not match: ' + out[-300:]}", flush=True)
                failed = True
            elif r:
                got[name].append(v)
    ref = next((x[2] for x in got[cfg[0][0]]), None)
    for name, _, _ in cfg:
        g = got[name]
        if not g:
            continue
        line = name.ljust(12)
        for label, k in (("prompt", 1), ("decode", 0)):
            v = [float(x[k]) for x in g if x[k] is not None]
            if v:
                m = statistics.median(v)
                line += f" {label} {v} median {m:g} spread {100 * (max(v) - min(v)) / m:.1f}%"
        if ref is not None:
            d = []
            for x in g:   # (every round: a build that differs from itself is a bug)
                p, q = ref.split(), x[2].split()
                if p != q:
                    d.append(next((i for i, (u, w) in enumerate(zip(p, q)) if u != w), min(len(p), len(q))))
            line += " ids " + ("same" if not d else f"differ at {d}")
        print(line, flush=True)
    sys.exit(1 if failed else 0)


main()
