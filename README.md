# gpu-kernel-tuning

An agent skill for making the GPU kernels of a working LLM inference engine faster without changing its output: decode
speed, prompt (prefill) speed and first-prompt time. It covers how to find the kernels that limit speed, which changes paid
and in what order, how to prove each step with alternating A/B runs and a token-level correctness check, and when to stop.

It was written from a complete optimization of an MoE engine on AMD GPUs (Inkling 2 Mini, IQ4_XS GGUF, RX 7900 XT and
Radeon 8060S, HIP): decode 94 -> 109 tok/s and 4K-token prompts 650 -> 3,800 tok/s on the RX 7900 XT, every step checked
against llama.cpp. The method is not tied to that model or vendor; `references/any-model.md` maps it to other
architectures, weight formats, GPUs and engines, and marks what was tested.
Since then it was applied to a second quant of the same model (ISTA-DASLab's GSQ-RCO file: decode 65 -> 103 tok/s on
the RX 7900 XT) and to a second model in another engine (Ling-3.2-air on a Strix Halo iGPU plus an RX 7900 XT over
USB4: 4K prompts 460 -> 810 tok/s, decode 51.7 -> 64.2 tok/s from one stream join inside the token graph);
`references/case-study-2.md` has both.

## Contents

```
skills/gpu-kernel-tuning/
  SKILL.md                       the procedure: steps, decisions, mistakes to avoid, checks before finishing
  references/any-model.md        worksheet for a new model and machine; where to look by architecture, format, GPU, engine
  references/patterns.md         the code patterns that paid (batched loads, matrix-core tiles, fewer copies, warm-up, fusion)
  references/correctness-gate.md the teacher-forced token check against llama.cpp
  references/case-study.md       the case it was written from: numbers, dropped attempts, mistakes, how the skill was tested
  references/case-study-2.md     a second quant of the same model and a second model (hybrid MoE, two GPUs): what paid,
                                 what did not, why a smaller quant was not faster, and mistakes to avoid
  scripts/ab.py                  alternating A/B of engine commands: medians, spread, identical-output check
  scripts/make_reference.py      greedy reference with top-5 log-probabilities from a running llama-server
  scripts/score.py               teacher-forced agreement of an engine with that reference
  scripts/kregs.py               registers, spills and shared memory per kernel of a HIP build (AMD)
  scripts/matrix_peak.hip        measured matrix-core rate (RDNA 3 WMMA: FP16, FP16 accumulate, INT8)
  scripts/tile_layers.hip        how much of that rate a tiled matrix kernel keeps as barriers, decode and loads are added
```

The scripts need Python 3.10 or newer and only its standard library. The `.hip` files need a ROCm compiler.

## Install

The skill is a folder with a `SKILL.md`; each agent finds it in its own skills directory. Clone this repository and link
or copy `skills/gpu-kernel-tuning` there.

| Agent | Skills directory |
|---|---|
| Claude Code | `~/.claude/skills/` |
| Codex CLI | `~/.agents/skills/` |
| Cline | `~/.cline/skills/` (or `.cline/skills/` in a project) |
| Hermes Agent | `~/.hermes/skills/` on Linux and macOS, `%LOCALAPPDATA%\hermes\skills\` on Windows |
| DeepSeek Harness (dsh) | `~/.dsh/skills/` |
| oh-my-pi (omp) | `~/.omp/agent/skills/` |
| pi | `~/.pi/agent/skills/` |

`install.ps1` (Windows) and `install.sh` (Linux, macOS) link the skill into every directory above whose agent is installed:

```powershell
.\install.ps1            # directory junctions; -Copy copies instead
```

```bash
./install.sh             # symbolic links; --copy copies instead
```

Then ask the agent, for example: "Profile this engine's decode path and make it faster; keep the output identical."

## How it was tested

Following the method of [skill-distill](https://github.com/sybil-solutions/skill-distill), a smaller open model was given
two short kernel tasks with and without the skill (three trials each), and its answers were compiled and checked. With the
skill it solved 6 of 6; without it, 0 of 5. Details are at the end of `references/case-study.md`.

## Credits

- The testing approach follows [sybil-solutions/skill-distill](https://github.com/sybil-solutions/skill-distill) (MIT):
  a stronger model writes a skill from a finished task, and a weaker model is run with it to find where it fails. Its
  student-skill template shaped the layout of `SKILL.md`. No code from that repository is included.
- The case study is a port of the `maya-ik2` engine from
  [alesha-pro/maya-inkling-mini](https://github.com/alesha-pro/maya-inkling-mini) (MIT, Project Maya and Strata
  contributors). Some prompt-kernel designs it mentions follow CIRU AI's
  [Inkling-2-Mini-CIRU-Halo-Agent-gfx1151](https://github.com/ciru-ai/Inkling-2-Mini-CIRU-Halo-Agent-gfx1151)
  (Apache-2.0). The code snippets in `references/patterns.md` are short illustrations written for this skill.
- The correctness gate uses [llama.cpp](https://github.com/ggml-org/llama.cpp)'s server as the reference.

## License

MIT, see [LICENSE](LICENSE).
