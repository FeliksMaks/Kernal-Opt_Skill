// The matrix cores' real rate: 16 x 16 x 16 products from registers only, no loads.  HIP_VISIBLE_DEVICES picks the GPU; clang++ -O3 -x hip --offload-arch=gfxXXXX
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
using h16 = _Float16 __attribute__((ext_vector_type(16)));
using f8 = float __attribute__((ext_vector_type(8)));
using hh8 = _Float16 __attribute__((ext_vector_type(8)));
using i8v = int __attribute__((ext_vector_type(8)));
using i4v = int __attribute__((ext_vector_type(4)));
// M: 0 f16->f32, 1 f16->f16, 2 iu8->i32
template <int M, int NA>
__global__ void __launch_bounds__(256) peak(float* out, int iters, float seed) {
    if constexpr (M == 0) {
        h16 a, b[4];
        f8 acc[NA];
        for (int i = 0; i < 16; ++i) a[i] = (_Float16) (seed * (threadIdx.x + i));
#pragma unroll
        for (int j = 0; j < 4; ++j)
            for (int i = 0; i < 16; ++i) b[j][i] = (_Float16) (seed + j + i);
#pragma unroll
        for (int j = 0; j < NA; ++j) acc[j] = f8{};
        for (int k = 0; k < iters; ++k)
#pragma unroll
            for (int j = 0; j < NA; ++j) acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b[j & 3], acc[j]);
        float s = 0;
#pragma unroll
        for (int j = 0; j < NA; ++j)
            for (int i = 0; i < 8; ++i) s += acc[j][i];
        out[threadIdx.x + 256 * blockIdx.x] = s;
    } else if constexpr (M == 1) {
        h16 a, b[4];
        h16 acc[NA];
        for (int i = 0; i < 16; ++i) a[i] = (_Float16) (seed * (threadIdx.x + i));
#pragma unroll
        for (int j = 0; j < 4; ++j)
            for (int i = 0; i < 16; ++i) b[j][i] = (_Float16) (seed + j + i);
#pragma unroll
        for (int j = 0; j < NA; ++j) acc[j] = h16{};
        for (int k = 0; k < iters; ++k)
#pragma unroll
            for (int j = 0; j < NA; ++j) acc[j] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a, b[j & 3], acc[j], false);
        float s = 0;
#pragma unroll
        for (int j = 0; j < NA; ++j)
            for (int i = 0; i < 8; ++i) s += (float) acc[j][i];
        out[threadIdx.x + 256 * blockIdx.x] = s;
    } else {
        i4v a, b[4];
        i8v acc[NA];
        for (int i = 0; i < 4; ++i) a[i] = (int) (seed * 1000) + threadIdx.x * 77 + i;
#pragma unroll
        for (int j = 0; j < 4; ++j)
            for (int i = 0; i < 4; ++i) b[j][i] = (int) (seed * 500) + j * 31 + i;
#pragma unroll
        for (int j = 0; j < NA; ++j) acc[j] = i8v{};
        for (int k = 0; k < iters; ++k)
#pragma unroll
            for (int j = 0; j < NA; ++j) acc[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b[j & 3], acc[j], false);
        int s = 0;
#pragma unroll
        for (int j = 0; j < NA; ++j)
            for (int i = 0; i < 8; ++i) s += acc[j][i];
        out[threadIdx.x + 256 * blockIdx.x] = (float) s;
    }
}
template <int M, int NA>
void go(const char* n, float* out, int cus) {
    const int blocks = cus * 8, iters = 20000;
    for (int r = 0; r < 3; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        peak<M, NA><<<blocks, 256>>>(out, r ? iters : 100, 0.001f);
        hipDeviceSynchronize();
        double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        // a wmma: 16 x 16 x 16 = 4096 MAC a wave; 8 waves a block
        if (r) printf("%s x%d  %.2f TMAC/s (%.1f ms)\n", n, NA, 4096.0 * 8 * blocks * NA * iters / t / 1e12, t * 1e3);
    }
}
int main() {
    hipDeviceProp_t p;
    hipGetDeviceProperties(&p, 0);
    printf("%s  CUs %d  clock %d MHz\n", p.name, p.multiProcessorCount, p.clockRate / 1000);
    float* out;
    hipMalloc(&out, 4 << 20);
    const int cu = p.multiProcessorCount;
    go<0, 4>("f16->f32", out, cu);
    go<0, 8>("f16->f32", out, cu);
    go<1, 8>("f16->f16", out, cu);
    go<2, 8>("iu8->i32", out, cu);
}
