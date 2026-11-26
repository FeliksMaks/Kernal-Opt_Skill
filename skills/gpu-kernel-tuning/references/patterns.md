# Patterns

## a. Batched loads in a kernel that waits

Recognise: one launch takes over about three times bytes / bandwidth + launch cost; the loop loads, uses, loads, uses. The same
holds for a product, a norm, a mix of several rows, a gather: load first, then use in the old order.

```c
// before: one load in flight per lane and step (32 lanes; K = 2304, so 72 steps)
for (int i = lane; i < K; i += 32) a += h2f(w[i]) * x[i];
```
```c
// after: kB loads in flight, summed in the same order - the same bits
constexpr int kB = 24;                       // 72 steps = 3 batches
int i = lane;
for (; i + 32 * (kB - 1) < K; i += 32 * kB) {   // full batches only
    uint16_t wv[kB]; float xv[kB];
#pragma unroll
    for (int b = 0; b < kB; ++b) wv[b] = w[i + 32 * b], xv[b] = x[i + 32 * b];
#pragma unroll
    for (int b = 0; b < kB; ++b) a += h2f(wv[b]) * xv[b];
}
for (; i < K; i += 32) a += h2f(w[i]) * x[i];   // the rest, as before
```

- Exact only if every lane keeps its own accumulator and its order of terms, and the last reduction is unchanged.
- Two separate, fully unrolled loops. In one loop the compiler goes back to a load per step.
- Tail: never load past the end. Never pad with `0 * x`: it is NaN for inf or NaN.
- Try kB = 8, 16, 24 and keep the fastest. More registers lower occupancy.
- Neutral or slower in kernels that stream (experts, head).

## b. One prompt launch per layer

Recognise: the prompt launches a kernel per expert and layer, and the GPU idles between them.
Make a list of (expert, rows) tiles and launch one kernel over all of them. Each block decodes its weight tile to FP16 in
LDS and multiplies it with its rows on the matrix cores. Tiles of 128 rows by 256 weight rows did best on RDNA 3. A tile
needs several rows to pay for its decode, and at short prompts an expert has few.

## c. Odd formats in the same tiles

Recognise: profile by layer, not only by kernel. Four visits of 48 were 40% of the expert time at 511 tokens, because their
formats went expert by expert. Make the tile's decode a template on the weight type and leave the product alone.

## d. No copies between kernels

Recognise: gather, pack, unpack and concat kernels around the products.
Pass the index to the consumer (`row = idx[r]`), and an output pointer and stride for each part. Exact by construction.

## e. Warm-up at load

Recognise: the first prompt of a process is much slower than the second. One big kernel's first launch took 55 ms.
At load: allocate at full size, run a dummy prompt and two single tokens through the normal code, then zero every recurrent
state and counter they touched. Check that the first prompt's ids are unchanged.

## f. Fusing launches

Recognise: host queueing is a large part of the token, and pairs of kernels always follow each other.
Inside one dispatch a consumer can wait for its producers on a counter in memory only if blocks start in order on this GPU.
Bound the wait and write NaN when it runs out, so that a wrong assumption shows at once and cannot hang.

## g. Prompt attention

One lane a query; keys as a matrix tile; the positional bias computed in the kernel, not read from a table. Not exact: it
needs the gate and a KL comparison (SKILL.md, step 5). It gained 13% on an iGPU and nothing on a discrete card.

## CUDA

The same patterns map as follows: WMMA intrinsics -> `nvcuda::wmma`; LDS -> shared memory; wave32 -> warp of 32; `sudot4` -> `__dp4a`. Measure launch cost
and latency again: they decide which patterns pay.
