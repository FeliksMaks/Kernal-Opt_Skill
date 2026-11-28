---
name: gpu-kernel-tuning
description: Speed up the GPU kernels of any local LLM engine that already works, for any model (MoE, dense, looped, hybrid), quant and GPU (proven with HIP/ROCm on AMD RDNA 3; the method carries to CUDA, SYCL and Metal) - decode tok/s, prompt (prefill) tok/s and first-prompt time - without changing its output. Finds kernels that wait on memory latency, batches their loads bit-for-bit, moves prompt work onto the matrix cores, removes launches, copies and first-launch costs, and proves each step with alternating A/B runs and a token check. Use when asked to optimise, rewrite or port GPU kernels, raise decode or prefill speed towards the bandwidth ceiling, or work out why an engine is far below it.
whenToUse: "The user has a working GPU inference engine with source code and wants its kernels faster: decode, prompt speed or short prompts, on AMD, NVIDIA, Intel or Apple GPUs, for any model."
---

# Tuning the GPU kernels of a working LLM engine

Taken from a port of an MoE model (Inkling-2-Mini, IQ4_XS GGUF) to an RX 7900 XT and a Radeon 8060S. Its numbers are in
`references/case-study.md`. They show what kind of step pays, not what your model will get.
For every other model, quant, GPU or engine start at `references/any-model.md`: a worksheet and tables of where to look first.

No engine source, only serving flags? This skill does not apply. No correctness reference yet? Make one first
(`references/correctness-gate.md`).

## Done means

- [ ] A ledger (`PROGRESS.md`) with the ceiling, the baseline, and a line per attempt: change, numbers, kept or dropped.
- [ ] Every kept step: the median of >= 3 alternating rounds is better, and the generated ids are the same as before
      (or, if they differ, the token gate did not get worse - step 5).
- [ ] The final build passes the gate on every GPU it ships for.
- [ ] The report says what did not work and what was not tested.

## Harness notes

- A build or benchmark of more than ~2 minutes goes in the background with its output in a file. A tool call that timed out
  may still be running: look before you start it again. Two runs on one GPU ruin both numbers.
- Write scripts to a file with your file tool, then run the file. Inline commands lose backslashes and quotes.
- Read only the part of a source or log that you need. Engine sources are often over 5,000 lines.
- Stop only processes you started, by PID. Do not kill by image name, delete files, commit or install without being asked.
- Keep the sources and the binary of every kept step, so that you can go back.

## Steps

0. **Worksheet.** Fill in `references/any-model.md` for this model, file and machine, from the header and measurements.
   You should see: a weight type for every tensor group and layer, and no empty ceiling row. A new file of a model the
   engine already runs: list the formats it adds and check that every kernel switch knows them.

1. **Baseline.** Pick the GPU by name (HIP's device order can change between boots) and check that nothing else uses it.
   Run prompts of ~30, ~500, ~4,000 and ~32,000 tokens (as far as the context allows), >= 200 generated tokens, greedy, with
   the settings the user runs. `scripts/ab.py` does one length. Write down the medians and the ids.
   You should see: decode spread within ~2%. If not, something else runs, or the GPU is hot or power-limited. Fix that first.

2. **Ceiling.** Decode: bytes read per token (dense weights + the experts in use + KV) / measured read bandwidth. 60-70% of
   it is good, 80-85% excellent. Prompt: the operations of the largest kernel group / the GPU's matrix rate,
   **measured** (`scripts/matrix_peak.hip`). A rate worked out on paper was 30% too low here and hid the headroom.
   Use both to say early if a target cannot be reached.

3. **Find where the time goes.** In this order; stop at the first that answers the question:
   - *Duplicate, do not skip.* Launch one kernel group twice (it writes the same output again) and compare the token time.
     The difference / launches is the cost. It is a floor, because the second run reads from cache.
     **Never skip a group to time it**: stale indices downstream can read outside buffers and hang the GPU.
   - *Cost against bytes.* Per group: time per launch against bytes / measured bandwidth + launch cost. Over about three
     times that, it waits for single loads (here: 13 us for 59 KB). Those are the cheapest wins (4a).
   - *Queueing against waiting.* Time the host part of a token. If it is a large part, fewer launches are the lever (4f).
   - *First prompt against second.* Run the same prompt twice in one process. A big gap is start-up cost (4e).
   - *Runtime settings before kernels.* Prompt chunk size, batch and buffer sizes: a chunk of 4,096 tokens instead of 2,048 gave
     5-8% and no code. Check that it still fits in VRAM at the longest context.
   - *Per GPU, then per kernel, minus the sync floor.* With several GPUs, time each GPU's segment first. Per kernel,
     measure the floor (the smallest kernel with a sync) per GPU and subtract it: on an eGPU it was 0.1 ms, which makes
     every small kernel look slow. If the kernels' sum is well under the graph's time, the cost is between them.
   - *Why a matrix kernel is under its peak*: `scripts/tile_layers.hip` adds barriers, decode and loads one at a time.
   - A profiler that synchronises after each kernel overstates small ones. Confirm with the A/B.

4. **Change one thing, behind an environment variable that turns it off.** Read the matching section of
   `references/patterns.md` before you write code.
   a. Batch the loads of small kernels (router, gates, norms, combine): same sums, same order, same bits.
   b. Prompt: one launch per layer on the matrix cores, not one per expert.
   c. Put the odd weight formats into the same tiles.
   d. Read inputs where they lie and write outputs to their place: no gather, unpack or copy passes.
   e. Warm up at load: full-size buffers and each kernel's first launch, then reset the state it wrote.
   f. Fuse launches that always follow each other.
   g. Prompt attention: one lane per query, the bias computed in the kernel.
   h. Narrow formats (FP8, 4-bit) into the matrix cores by moving bits, not converting values; scales absorb the factor.
   i. Check every path a setting switches: an option such as an FP8 cache can still run an old kernel.
   j. A new quant of the same model: give each of its weight formats a compiled kernel; the old list leaves them on the
      general path.
   k. Dense weights in a mix of formats: requantize to the fast path's format at load, gated, with a switch.
   l. Graph time far above the sum of its kernels: look for event joins between streams; run small side work inline.

5. **Check the output before you believe the speed.**
   - Same ids as the old build at every length: go on.
   - Ids differ: run the token gate (teacher-forced top-1 against llama.cpp; `references/correctness-gate.md`). Accept if
     agreement is not lower than the old build's and each new miss is a near tie (reference top-2 gap < 0.5).
   - Near the pass line, or fewer than 500 positions: make a longer reference (>= 3,000 positions, prompts of 1-3K tokens)
     and compare the KL of both builds to an independent path, such as the unfused kernel. Report the number of positions.
   - A kernel that differs is not necessarily the wrong one. The old one can be further from the reference.

6. **A/B**: `python scripts/ab.py ...`, >= 3 rounds, one binary with the switch on and off when possible. Keep a gain only if
   the median improves by more than the baseline's spread. Run every prompt length and every GPU: a gain on one GPU is often
   nothing on the other.

7. **Write it down**, kept and dropped, save sources and binary, go to step 3. No pattern fits: `references/any-model.md`,
   order of work, point 4.
   Three honest attempts in an area without a kept gain: the area is exhausted. Say so and take the next one.

## Decisions (and why)

- Exact changes first: they need no quality argument, and gave most of the decode gain.
- A switch per change: one binary, no build noise, and a regression is found in minutes.
- Only exact or gated changes become defaults; the rest is opt-in and the user decides.
- Measure with the launcher's environment variables: without them the numbers are not the user's.

## Don't

- Don't skip launches to profile. It wedged a card, and only a restart brought it back.
- Don't reorder a reduction (float4 loads, other lane counts, other tile widths) and call it exact. Compare the ids.
- Don't duplicate a matrix-core loop for a fast path without `scripts/kregs.py`: one copy more spilled registers, 5 times slower.
- Don't unroll everything. Loads in flight help kernels that wait; in kernels that stream they only cost registers.
- Don't trust one short-prompt number: at 128-511 tokens the spread can be 20%, and the first launch hides in it.
- Don't keep the logit dumps of a gate run. Long ones filled a disk with 42 GB. Delete each after scoring.
- Don't compare with another engine's published number. Run it on this machine with the same file and lengths.
- Don't report a speed before the gate. A new file whose loop keys the engine did not find ran each layer once and was
  "faster"; only the gate (70%) showed it.
- Don't trust an A/B whose configs give identical ids and speed when the change is not exact: check that each run loaded
  the model and settings you meant.
- Don't promise a speed-up from a smaller quant before you compare bytes per token with the token time
  (`references/case-study-2.md`). Where decode is bound by per-step cost, the smaller file only helps when it fits where the
  larger one does not.
- Don't move layers to the faster card without measuring a layer's cost there: over USB4 a layer cost the same on the
  eGPU as on the iGPU (latency, not bandwidth), and one layer too many pushed it out of VRAM.
- Don't trust a CUDA codebase's capability checks on HIP (`major >= 8` is true on RDNA 3 and changes data-type paths),
  and test any vision path with a trivial image first.
- Don't let an override fail quietly: a layer-split variable whose total was off by one was ignored without a word.
- Don't assume matrix-core tiles win for every product: with many experts and few rows each, a tile can be slower than
  the token-style kernel. Measure each matrix separately.

## Verify before you say you are done

1. `python scripts/ab.py --help` runs; the last A/B of the final binary against the baseline is in the ledger.
2. The gate of the final binary, per GPU, is in the ledger as matches / positions.
3. The final binary comes from the saved sources: build once more and compare ids.

Then reply with a table (length x GPU: before, after), matches / positions, what was dropped, and what was not tested.

## Files

- `references/any-model.md` - the worksheet, and where to look by architecture, weight type, GPU and engine.
- `references/correctness-gate.md` - the teacher-forced token check against llama.cpp.
- `references/patterns.md` - code for step 4, and how to recognise each case.
- `references/case-study.md` - the Inkling port: numbers, dead ends, mistakes; how this skill was tested.
- `references/case-study-2.md` - a second quant of the same model, a second model (hybrid MoE on two GPUs), and the
  mistakes made on the way.
- `scripts/kregs.py` - registers, spills and shared memory per kernel of a built exe (AMD).
- `scripts/matrix_peak.hip`, `scripts/tile_layers.hip` - the matrix cores' real rate, and what a tile loses to each part (RDNA 3).
- `scripts/make_reference.py`, `scripts/score.py` - a llama.cpp reference and the token gate.
- `scripts/ab.py` - alternating A/B of engine commands: medians, spread, ids.
