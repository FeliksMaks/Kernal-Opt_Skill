# Case study: Inkling-2-Mini (IQ4_XS) on RX 7900 XT (XT) and Radeon 8060S

24 layers, 12 of them run three times: 48 visits a token, 8 of 128 experts each. 3.88 GB a token: ceiling 200 tok/s on the XT
(775 GB/s), 62 on the 8060S (241 GB/s). Windows, HIP / ROCm 7.14.1. tok/s, medians of alternating runs.

| Step | XT | 8060S | Ids |
|---|---|---|---|
| Decode: router product, 24 loads in flight | 93.5 -> 100.4 | with the next row | same |
| Decode: gates, sinks, combine, attention merge, same way | 99.7 -> 108.6 | 43.4 -> 47.4 | same |
| Decode: gate/up and down experts in one dispatch | 94.7 -> 95.6 | 44.1 -> 44.2 | same |
| Decode: attention on the matrix cores | 91.3 -> 93.2 | 42.6 -> 43.6 | some differ late |
| Prompt 4K: one launch a layer on the matrix cores | 650 -> ~1,300 | 318 -> ~590 | gated |
| Prompt 4K: dense products on the matrix cores | -> ~2,400 | -> ~940 | gated |
| Prompt 4K: odd formats in the same tiles | 2,510 -> 3,102 | 1,305 -> 1,359 | gated |
| First prompt, 511: warm-up at load | 1,557 -> 2,248 | 910 -> 1,058 | same |
| Prompt 4K: no gather, no unpack | 3,457 -> 3,490 (noise) | 1,355 -> 1,382 | same |
| Prompt 4K: attention, a lane per query (opt-in) | no gain | 1,382 -> 1,562 | differ; gate equal |

Decode ended at 54% of the XT's ceiling and 76% of the 8060S's. The last attempts (norms, head, epilogues, expert loads)
were neutral or worse: a token is several hundred launches that depend on each other.

Dropped: both expert rows' loads in flight (103 against 109), down-block prefetch (106-108 against 109), 128-column expert
tiles, weights two steps ahead, flushing the queue mid-token, batched conv / gate / combine in the prompt (about 1%, and some
tokens changed).

Prompt tiles, later: the 8060S's matrix cores measured 21 TMAC/s, where paper said 14.8. A tile loop alone reached 20, with loads
13.6-17, the real one 12.5. Tiles of 256 rows, a kernel for full tiles only, a register limit (8 waves a SIMD) and 256-row dense tiles
did not help (0 to -5%): what is left is traffic between kernels, not the product. The prompt chunk of 4,096 tokens did:

| Prompt | XT | 8060S |
|---|---|---|
| 4K, chunk 2,048 -> 4,096 | 3,539 -> 3,802 | 1,420 -> 1,485; with the lane-per-query attention 1,605 -> ~1,710 |
| 32K | 3,279 -> 3,493 | 1,214 -> 1,272 |

Behind the server at a 64K context the same 4,096 filled the XT's 20 GB and prompts fell from 3,460 to 2,480: 3,072 was right there. The
short benchmark had not shown it.

Mistakes: skipping launches as a probe wedged the XT. A short gate (96 positions) made a better attention look worse, and 3,072
positions settled it (2,996 matches against 2,988). Benchmarks without the launcher's variables did not compare. Several
short-prompt gains were the 55 ms first launch. Saved logits filled the disk.

## How this skill was tested

Short tasks, not a whole tuning job. Student: Inkling-2-Mini (IQ4_XS) over a chat API without tools, temperature 0.7, with
SKILL.md and patterns.md in the prompt, against the same model without them. 16,000 tokens of budget, thinking included.

| Task | With the skill | Without |
|---|---|---|
| Batch the loads of a dot product (version 1 of patterns.md, 6,000-token budget) | 2 of 3 right, 1 ran out | 0 of 3, all ran out |
| The same, after patterns.md showed the tail | 3 of 3 | 0 of 3: 2 unrolled but still loaded one term at a time, 1 ran out |
| Mix of indexed rows (not in the skill) | 3 of 3 (by reading) | 0 of 2, both ran out; the third was not run to the end |

"Right" on the first task: 90,816 cases with the same bits, no read past the end, 32-48 loads before the first use. That
answer is close to the code in patterns.md, so the third row is the better evidence.
