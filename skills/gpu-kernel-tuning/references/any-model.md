# Another model, quant, GPU or engine

Fill in the worksheet from the model's config or GGUF header and from the machine. Do not fill it from memory.
Rows marked (tested) were done with this method. The rest is where to look first, not a result.

## Worksheet (put it at the top of PROGRESS.md)

| Field | Value |
|---|---|
| Layers, and how often each runs per token (loops, shared layers) | |
| Hidden size, FFN size, heads, KV heads, head size | |
| Experts: total, active, shared; or dense | |
| Attention: full, sliding, latent (MLA), linear / state space, mixed; its state per token | |
| Weight type per tensor group **and per layer** (one file often mixes 3-6 types) | |
| Head: type, tied or not | |
| Extra parts: draft heads, vision / audio encoder, memory tables | |
| Bytes read per token: dense, experts, head, KV at 4K and at 32K | |
| GPU: name, arch, lane width (32 or 64), matrix instructions and their types | |
| Measured: read GB/s, launch us, host-GPU round trip us | |
| Engine, language, how kernels are launched, where the switches go | |
| Workload that counts: one chat or many, prompt length, answer length | |
| Ceiling: decode at 4K and 32K, prompt | |

Waiting or streaming, per kernel: `stream time = bytes / measured bandwidth`. If a launch takes more than about three times
`stream time + launch cost`, it waits (pattern a). If it is close, only fewer bytes or fewer operations help.

## By architecture

| Model | Look first at |
|---|---|
| Mixture of experts (tested: 2 models) | router and gates (a), experts per launch (b, f), layers in odd formats (c), gather / combine (d) |
| Looped or shared layers (tested) | everything small costs visits x its time; per-visit state must be reset in the warm-up (e) |
| Dense | matrix-vector products stream: decode is near the ceiling or a product's decode is slow; then norms, sampling, launches (f) |
| Latent attention (MLA) | the extra down / up projections are small products (a); fused with the norm (f) |
| Sliding-window or mixed attention | KV bytes differ per layer; ceiling per layer type; masks computed in the kernel (g) |
| State space / linear attention | recurrences: one load in flight per step is common (a); the prompt needs a scan, so check that it is not run token by token |
| Draft heads, speculative decoding | measure acceptance x batch cost. The kept default must give the same greedy ids as without |
| Vision / audio encoder | its own baseline and gate (compare embeddings, not ids); usually prompt-side only |

## By weight type

| Type | Kernel |
|---|---|
| F16 / BF16 | exact conversions; hardware half conversion, not shifts. Check whether F32 tensors are exact BF16 (halves their bytes) |
| Q8_0 and similar | int8 dot products, or straight onto the matrix cores for the prompt (tested) |
| Q4 / Q5 / Q6 K | decode to int8 in shared memory, dot products; one template per type |
| IQ types, codebooks (tested) | tables in shared memory; whole words loaded before decoding |
| NVFP4, FP8 (tested: NVFP4) | keep their block scales as stored; matrix cores for the prompt |

## By GPU

| GPU | Differs |
|---|---|
| AMD RDNA 3 / 3.5 (tested: gfx1100, gfx1151) | wave32, WMMA FP16 / BF16 / int8, sudot4. The iGPU streams at once and is bandwidth-bound; the discrete card waits |
| AMD RDNA 4, CDNA | other matrix instructions, CDNA is wave64: lane counts and kB change |
| NVIDIA | tensor cores through `nvcuda::wmma`, `__dp4a`, profilers that can be trusted more; launches are cheaper |
| Intel (SYCL / Level Zero), Apple (Metal) | sub-group size varies, joint-matrix / simdgroup matrices; the steps hold, the code does not |
| Two GPUs | tune each alone first; then split by layers (one hand-off a token) rather than by experts (a round trip a layer) |

## By engine

| Engine | How |
|---|---|
| Own engine, Maya, Strata | edit the kernel, environment variable as the switch |
| llama.cpp | edit the backend's op (ggml-cuda / -hip / -vulkan / -sycl / -metal); switch by environment variable; gate against an unpatched build |
| vLLM, SGLang, TensorRT | kernels are in libraries: try flags and versions first |

## Order of work for a new model

1. Worksheet, gate and baseline. 2. Rank kernel groups by (time - ideal time). 3. Take the largest and its pattern from the tables.
4. If no pattern fits, micro-benchmark that kernel alone, then a version that only loads: near peak means that decoding is the
limit, far from it means that loading is. 5. Add a row here for what you learned, with its numbers.
