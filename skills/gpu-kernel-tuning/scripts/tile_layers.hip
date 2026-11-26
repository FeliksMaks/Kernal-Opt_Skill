// How much of the matrix cores' rate a 128 x 256 x K tile loop keeps as parts are added. Variants: 0 LDS reads and WMMA only;
// 1 + barriers and LDS writes; 2 + an NVFP4-like weight decode; 3 + global loads (each block its own rows); 4 as 3 with X shared by
// 9 blocks; 5, 6 weights spread over 8 GB; 7 as 4 with loads two steps ahead.  Build: clang++ -O3 -x hip --offload-arch=gfxXXXX.
// Run: tile.exe [repeats]; HIP_VISIBLE_DEVICES picks the GPU.
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
using h16 = _Float16 __attribute__((ext_vector_type(16)));
using f8 = float __attribute__((ext_vector_type(8)));
__device__ __forceinline__ h16 frag(const uint16_t* p) { h16 r; uint4 a[2] = {((const uint4*) p)[0], ((const uint4*) p)[1]}; __builtin_memcpy(&r, a, 32); return r; }
__device__ __forceinline__ uint32_t dec(uint32_t w, uint32_t sc) {   // two 4-bit values -> two FP16, a product each
    using hh = _Float16 __attribute__((ext_vector_type(2)));
    uint32_t t = ((w & 0x0F0Fu) | ((w & 0x0F0F00u) << 8)) * 0x200u + 0x3C003C00u;
    hh a = __builtin_bit_cast(hh, t), s = __builtin_bit_cast(hh, sc);
    return __builtin_bit_cast(uint32_t, a * s);
}
template <int V>
__global__ void __launch_bounds__(256) tile(const uint8_t* W, const uint16_t* X, float* out, int K, int C, int rb) {
    constexpr int LD = 40;
    __shared__ __align__(16) uint16_t Xs[128 * LD];
    __shared__ __align__(16) uint16_t Ws[256 * LD];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp >> 2, wn = warp & 3;
    const int xr = tid >> 1, xc = (tid & 1) * 16;
    const size_t far = (size_t) (((blockIdx.x / 9) * 2654435761u) % 1800u) * 4300000u / 256 * 256;   // (an expert's place)
    const bool F5 = V == 5 || V == 6;
    const uint32_t* wrow = (const uint32_t*) (W + (F5 ? far + (size_t) ((blockIdx.x % 9) * 128 + (tid & 127)) * rb + (tid < 128 ? 0 : 1492992)
                                                      : (size_t) (blockIdx.x * 256 + tid) * rb));
    const int xrow_s = (int) (((V == 4 || V == 6 || V == 7 ? blockIdx.x / 9 : blockIdx.x) * 128 + xr) * 2654435761u >> 20);   // 0 .. 4095
    const uint4* xsrc = (const uint4*) (X + (size_t) (F5 ? xrow_s : (V == 4 || V == 7 ? blockIdx.x / 9 : blockIdx.x) * 128 + xr) * K + xc);
    uint32_t w[5] = {1u, 2u + lane, 3u, 4u, 5u};
    uint4 xa = make_uint4(tid, 1, 2, 3), xb = xa;
    auto load = [&](int ks) {
        const uint32_t* p = wrow + (ks >> 1) * 9;
        w[0] = p[0];
#pragma unroll
        for (int i = 0; i < 4; ++i) w[1 + i] = p[1 + 4 * (ks & 1) + i];
        xa = xsrc[ks * 4], xb = xsrc[ks * 4 + 1];
    };
    uint32_t w2[5]; uint4 xa2, xb2;
    if constexpr (V >= 3) load(0);
    auto load2 = [&](int ks) {
        const uint32_t* p = wrow + (ks >> 1) * 9;
        w2[0] = p[0];
#pragma unroll
        for (int i = 0; i < 4; ++i) w2[1 + i] = p[1 + 4 * (ks & 1) + i];
        xa2 = xsrc[ks * 4], xb2 = xsrc[ks * 4 + 1];
    };
    if constexpr (V == 7) load2(1);
    f8 acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = f8{};
    const int nk = K / 32;
    for (int ks = 0; ks < nk; ++ks) {
        if constexpr (V >= 1) {
            __syncthreads();
            *(uint4*) (Xs + xr * LD + xc) = xa;
            *(uint4*) (Xs + xr * LD + xc + 8) = xb;
            uint32_t h[16];
            if constexpr (V >= 2) {
#pragma unroll
                for (int i = 0; i < 16; ++i) h[i] = dec(w[1 + (i >> 2)] >> (4 * (i & 3)), w[0] + (i >> 3));
            } else {
#pragma unroll
                for (int i = 0; i < 16; ++i) h[i] = w[i & 3] + ks;
            }
            uint4* d4 = (uint4*) (Ws + tid * LD);
#pragma unroll
            for (int i = 0; i < 4; ++i) d4[i] = make_uint4(h[4 * i], h[4 * i + 1], h[4 * i + 2], h[4 * i + 3]);
            __syncthreads();
            if constexpr (V == 7) {
#pragma unroll
                for (int i = 0; i < 5; ++i) w[i] = w2[i];
                xa = xa2, xb = xb2;
                if (ks + 2 < nk) load2(ks + 2);
            } else if constexpr (V >= 3) if (ks + 1 < nk) load(ks + 1);
        }
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            h16 a[4];
#pragma unroll
            for (int im = 0; im < 4; ++im) a[im] = frag(Xs + (64 * wm + 16 * im + (lane & 15)) * LD + kk * 16);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const h16 b = frag(Ws + (64 * wn + 16 * j + (lane & 15)) * LD + kk * 16);
#pragma unroll
                for (int im = 0; im < 4; ++im) acc[im][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[im], b, acc[im][j]);
            }
        }
        if constexpr (V == 0) if (ks == 1000000) __syncthreads();
    }
    float s = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            for (int q = 0; q < 8; ++q) s += acc[i][j][q];
    out[tid + 256 * blockIdx.x] = s;
}
int main(int argc, char** argv) {
    const int Q = argc > 1 ? atoi(argv[1]) : 10;
    hipDeviceProp_t p;
    hipGetDeviceProperties(&p, 0);
    printf("%s\n", p.name);
    const int K = 2304, blocks = 576, rb = K / 64 * 36;
    uint8_t* W; uint16_t* X; float* out;
    hipMalloc(&W, (size_t) 8 << 30), hipMalloc(&X, (size_t) blocks * 128 * K * 2), hipMalloc(&out, blocks * 256 * 4);
    hipMemset(W, 0x35, (size_t) blocks * 256 * rb), hipMemset(X, 0x11, (size_t) blocks * 128 * K * 2);
    hipDeviceSynchronize();
    const char* n[] = {"LDS reads + WMMA", "+ barriers, LDS writes", "+ decode", "+ loads (all own)", "loads, X shared by 9", "far weights, own rows", "far weights, shared rows", "shared, two steps ahead"};
    for (int v = 0; v < 8; ++v) {
        double best = 1e9;
        for (int r = 0; r < (Q > 100 ? 1 : 6); ++r) {
            auto t0 = std::chrono::steady_clock::now();
            for (int q = 0; q < Q; ++q) {
                if (v == 0) tile<0><<<blocks, 256>>>(W, X, out, K, K, rb);
                if (v == 1) tile<1><<<blocks, 256>>>(W, X, out, K, K, rb);
                if (v == 2) tile<2><<<blocks, 256>>>(W, X, out, K, K, rb);
                if (v == 3) tile<3><<<blocks, 256>>>(W, X, out, K, K, rb);
                if (v == 5) tile<5><<<blocks, 256>>>(W, X, out, K, K, rb);
                if (v == 6) tile<6><<<blocks, 256>>>(W, X, out, K, K, rb);
                if (v == 7) tile<7><<<blocks, 256>>>(W, X, out, K, K, rb);
                if (v == 4) tile<4><<<blocks, 256>>>(W, X, out, K, K, rb);
            }
            hipDeviceSynchronize();
            double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (t < best) best = t;
        }
        printf("%-26s %.2f TMAC/s\n", n[v], (double) Q * blocks * 128 * 256 * (double) K / best / 1e12);
    }
}
