# Case study 2: a second quant of the same model, and a second model

Two follow-ups to the Inkling port (`case-study.md`): ISTA-DASLab's GSQ-RCO file of Inkling 2 Mini (IQ3_S, 11.4 GB,
against 15.7 GB for the UD-IQ4_XS file), and Ling-3.2-air (a 42-layer hybrid MoE: linear-attention layers, a few MLA
layers, 512 experts with 8 active) in another engine on two GPUs. What paid, what did not, and what went wrong.

## A new quant of a model the engine already runs

| Step | RX 7900 XT decode | Output |
|---|---|---|
| Loads and runs (general path for the new formats) | 65 tok/s | gated |
| Dense weights requantized to Q8_0 at load (pattern k) | 82 | changed, gated, opt-out switch |
| Compiled kernels for the new expert formats (pattern j) | 103 | same ids as the step before |
| Reference: the larger UD file on the same engine | 109 | |

Where the time was: the profile of one token per group, both files side by side. Only two groups differed: the attention
projections (a mix of Q4_K, Q5_K, Q6_K and BF16 that the fast fused path did not take) and the experts (IQ3_XXS and
IQ4_NL, formats that the compiled per-type switch did not list, so they fell to the general kernel). Everything else was
level. A side-by-side profile against a file that is already fast finds this in minutes.

What did not pay:
- int8 dot products in the expert kernel (input quantized to int8 per 32 values, grid bytes signed in place, dp4a):
  +0.5%. The kernel waits for its loads; fewer arithmetic instructions do not matter there.
- Block shapes for the expert kernels: within noise.
- Speculative decoding with the model's own draft head: slower for both files (103 -> 51, 109 -> 79). Its verification
  path was not tuned. Measure it before you assume it helps.

### The smaller file is not faster

At 100 tok/s the token reads about 1.1 GB: an eighth of the card's bandwidth. The model runs its middle layers three
times, so a token is 48 layer passes of small kernels, and decode is bound by per-step cost. Fewer bytes shorten only
the part that was never the bottleneck. The smaller file pays where it decides whether the model fits: on a 16 GB
RX 7800 XT, GSQ 83 tok/s against 7 tok/s for the UD file, whose experts had to be read from system memory.

Work this out before promising a speed-up from a smaller quant: bytes per token / bandwidth against the measured token
time. Under ~25% means per-step cost dominates.

### Quality, measured rather than eyeballed

Perplexity through the engine on 20K tokens of mixed code and prose (+1.1%, with a paired bootstrap interval), top-1
agreement between the two files (96.9%), and a small scored task set (maths with exact answers, functions with unit
tests). One sample prompt read by eye says nothing about a quant.

## Mistakes to avoid

- **A fast number from a model that did not fully run.** The new file named its loop keys differently; the engine found
  none and ran each layer once. It reported 123 tok/s and fluent text. The token gate scored 70% with 27 hard misses,
  which found it. Never report a speed before the gate.
- **An A/B harness that ran something else.** A model override in the benchmark script did not apply (a failed text
  substitution), so the "new file" runs were the old file twice: identical ids, identical speed. Check one log line that
  names the model, and expect ids to differ where a change is not exact.
- **The reference tool cannot read the file.** The newest llama.cpp read the GSQ file but not the older UD file; an
  older build read the UD file but not the newer layout. Keep the build each reference was made with. Where no reference
  build can load the model at all, use a relative gate: the engine's unchanged path as the reference, both paths
  teacher-forced over the same tokens, top-1 agreement and the log-probability gap (Ling below).
- **A GPU without its runtime libraries.** The build had code for the RX 7800 XT (`gfx1101`), but the installed BLAS
  library had no kernels for it and every run failed at start. Test each architecture you ship for on the card, not only
  in the build.
- **Windows file locks.** A memory-mapped dump stays open until every array view is gone; deleting it fails. Read the
  part you need into memory instead.

## A second model: Ling-3.2-air on two GPUs

The engine already worked (CUDA source built for HIP). Prompt speed at 4,096 tokens went 460 -> 810 tok/s:

| Step | Prompt 4K | Output |
|---|---|---|
| Baseline (BLAS for the dense products after an FP16 decode) | 460 | |
| Dense products on the matrix cores with the Inkling engine's tile kernel | 810 (+76%) | relative gate 121/128, baseline 122/128 |
| Routed experts' down product on matrix-core tiles | 780-810 | relative gate 121/128, baseline 119/128 |

- The profile per part (a sync and a timestamp per part, timing builds only) pointed at the dense products: the BLAS path
  ran at ~2.6 TFLOPS on the iGPU. Reusing a tile kernel that was already proven in another engine was the cheapest win.
- Expert tiles: the down product got faster (0.68 -> 0.30 s), the gate/up product slower (1.24 -> 1.40 s). With 512
  experts an expert gets ~64 rows of a 4K prompt, so a 128-row tile is half empty and each thread's weight decode is
  paid for few rows. Measure each matrix of a layer separately; keep only the one that wins.
- The reference problem: mainline llama.cpp could not load this file (a tensor shape it does not expect), so every
  change was gated relative to the engine's own unchanged path (above).
- Layer split across two GPUs by free memory instead of evenly took decode from ~10 to 31 tok/s, because the even split
  left experts on the SSD. Check where every expert lives before tuning kernels.

What to watch for with hybrid models: the linear-attention recurrence and the MLA layers have their own kernels and
their own shapes. A portable kernel can refuse a shape on one GPU (here an MLA decode kernel on the discrete card) and
fall back silently to a slow path; log which kernel each layer takes.

## Ling-3.2-air decode over USB4: an iGPU and an eGPU in one pipeline

Setup: a Radeon 8060S iGPU (Strix Halo, 64 GB of system memory, 48 GB of it for the GPU) holding layers 0-30 and an
RX 7900 XT 20 GB on USB4 holding layers 31-41; each token's activations cross between them through host memory (no
peer access), 0.57 ms a token. Decode with each GPU's layers as one graph a token.

| Step | Decode, drafting off | Decode, MTP drafting on (default) | Output |
|---|---|---|---|
| Baseline | 34.5 tok/s | 51.7 tok/s | |
| Shared expert on the main stream, no event join (pattern l) | **44.7** | **64.2** | same ids |

How it was found, and what the profile said on the way:
- Per GPU and per part first (a sync at the end of each GPU's segment, graphs kept): iGPU 21.0 ms, eGPU 6.5 ms, hand-off
  0.57 ms. The iGPU is the bottleneck.
- Per kernel next (graphs off, a sync after each kernel), with each GPU's **sync floor** measured from the tiniest
  kernels and subtracted: ~0.04 ms on the iGPU, ~0.1 ms on the USB4 card. Without subtracting it, every small kernel
  looks slow on the eGPU.
- Net of the floor, the iGPU's kernels ran near its bandwidth (dense projections ~100%, expert matvecs 58-74%). The sum
  of kernel times was well below the graph's real time: the missing time was between kernels.
- The layer's shared expert ran on a second stream, joined back with an event: inside a HIP graph on Windows that join
  cost ~6 ms a token. Running it inline removed it. Fusing one more kernel a layer (the expert sum into the next norm,
  exact) gave nothing measurable: a launch inside a graph was not the cost; the cross-stream wait was.
- The verify path of speculative decoding had its own copy of the same join; fixing only the single-token path moved
  drafting decode by 2%, fixing both by 24%.

What did not help on this pair:
- Moving layers from the iGPU to the eGPU: a layer cost ~0.6 ms on either GPU, although the eGPU has ~3.5x the
  bandwidth - it is latency-bound there (higher per-kernel cost over USB4), so the split saves little, and one layer too
  many pushed the eGPU out of VRAM (decode 34 -> 26 tok/s).
- Draft depth 2 or 3 instead of the auto rule: within noise. A 3-token verify step cost ~1.5x a single token, because
  each drafted token pulls in its own experts.

## Porting CUDA code to HIP: two silent failures

- **Capability checks.** Ling's vision tower chose BF16 or FP16 from `compute capability major >= 8`. On HIP that
  attribute is the gfx major (11 on RDNA 3), so it kept the weights in BF16 while the HIP branch told the GEMMs FP16:
  BF16 bits read as FP16. Nothing crashed and the embeddings had plausible magnitudes; the model simply could not see
  the image (a half-red, half-blue picture became "black text on white"). Grep a CUDA codebase for capability checks
  and data-type choices before trusting its HIP path, and test vision with a trivial image (two colours, a large
  number) before a real one.
- **Kernels enabled by architecture name.** The MLA decode kernel was on for gfx1151 only; on gfx1100 the model refused
  its shape unless an environment variable forced it. Log which kernel each layer takes, and check every architecture
  the build ships for.
