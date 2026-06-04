// matmul_skeleton.cu  --  YOUR KERNEL (start here)
// =============================================================================
// Assignment: optimize this kernel so it outperforms matmul_base.
//
// This file starts as an exact copy of the naive baseline (matmul_base).
// Your goal is to improve it by applying what you learned from studying
// matmul_compute, matmul_memory, and matmul_latency.
//
// Workflow:
//   1. Run kernel 5 (matmul_base) with ncu and record the baseline metrics.
//   2. Run kernels 2-4 (compute/memory/latency) and compare -- what improved?
//   3. Modify this file and run kernel 6 (matmul_skeleton) with ncu.
//   4. Compare your report to the baseline and explain what changed and why.
//
// After every edit, delete the build cache and rerun to recompile:
//   rmdir /s /q "%LOCALAPPDATA%\torch_extensions\torch_extensions\Cache\py312_cu128\matmul_skeleton"
//   python run.py 6 --tokens 250
//
// Tunable parameters:
//   TILE  -- 0: no tiling  |  N>0: use NxN shared-memory tiles
//   BLOCK -- block side length used when TILE == 0
// =============================================================================
#include <torch/extension.h>

#define TILE  0    // try enabling tiling and adjusting the size
#define BLOCK 4    // try increasing this and observe occupancy changes

#if TILE > 0
  #define BDIM TILE
#else
  #define BDIM BLOCK
#endif

__global__ void matmul_skeleton(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N)
{
    int b   = blockIdx.z;
    int row = blockIdx.y * BDIM + threadIdx.y;
    int col = blockIdx.x * BDIM + threadIdx.x;

    const float* a  = A + b * M * K;
    const float* bm = B + b * K * N;
    float*       c  = C + b * M * N;

#if TILE > 0
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];
    float sum = 0.0f;
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;
        sA[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? a[row * K + aCol] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? bm[bRow * N + col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; k++)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) c[row * N + col] = sum;
#else
    // Baseline: every multiply reads directly from global memory.
    // Each element of A and B is loaded K or M times across the block.
    // Think about: can any of these loads be shared between threads?
    if (row >= M || col >= N) return;
    float sum = 0.0f;
    for (int k = 0; k < K; k++)
        sum += a[row * K + k] * bm[k * N + col];
    c[row * N + col] = sum;
#endif
}

torch::Tensor batched_matmul(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "Only float32 supported");

    auto A3 = A.contiguous().reshape({-1, A.size(-2), A.size(-1)});
    auto B3 = B.contiguous().reshape({-1, B.size(-2), B.size(-1)});

    int batch = A3.size(0), M = A3.size(1), K = A3.size(2), N = B3.size(2);
    auto C3 = torch::empty({batch, M, N}, A.options());

    dim3 block(BDIM, BDIM);
    dim3 grid((N + BDIM - 1) / BDIM, (M + BDIM - 1) / BDIM, batch);
    matmul_skeleton<<<grid, block>>>(
        A3.data_ptr<float>(), B3.data_ptr<float>(), C3.data_ptr<float>(), M, K, N);

    auto out = A.sizes().vec();
    out[out.size() - 1] = N;
    return C3.reshape(out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_matmul", &batched_matmul, "Student matmul kernel");
}
