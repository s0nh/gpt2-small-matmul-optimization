// matmul_memory.cu  --  MEMORY OPTIMIZED
// =============================================================================
// Optimization strategy: maximize memory bandwidth utilization.
//
// The base kernel uses only 16 threads per block (half a warp), so each SM
// can issue very few memory requests at a time. Memory bandwidth goes unused.
//
// This kernel uses the largest practical block size (32x32 = 1024 threads =
// 32 warps). More warps per block means more in-flight memory transactions,
// which drives up L1/L2/DRAM throughput and better saturates the memory bus.
//
// Effect on NCU report (compare to matmul_base):
//   - Memory Throughput (L1/L2)  : increases  -- more requests in flight
//   - Achieved Occupancy         : increases  -- more threads resident per SM
//   - Duration                   : decreases  -- bandwidth is no longer wasted
//
// Tunable: adjust BLOCK and observe how memory throughput and occupancy change.
// =============================================================================
#include <torch/extension.h>

#define TILE  0    // no tiling -- focus is on bandwidth, not data reuse
#define BLOCK 32   // 32x32 = 1024 threads -- maximizes in-flight memory requests

#if TILE > 0
  #define BDIM TILE
#else
  #define BDIM BLOCK
#endif

__global__ void matmul_memory(
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

    if (row >= M || col >= N) return;
    float sum = 0.0f;
    for (int k = 0; k < K; k++)
        sum += a[row * K + k] * bm[k * N + col];
    c[row * N + col] = sum;
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
    matmul_memory<<<grid, block>>>(
        A3.data_ptr<float>(), B3.data_ptr<float>(), C3.data_ptr<float>(), M, K, N);

    auto out = A.sizes().vec();
    out[out.size() - 1] = N;
    return C3.reshape(out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_matmul", &batched_matmul, "Memory-optimized matmul (large blocks)");
}
