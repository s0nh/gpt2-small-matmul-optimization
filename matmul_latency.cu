// matmul_latency.cu  --  LATENCY OPTIMIZED
// =============================================================================
// Optimization strategy: hide memory latency with sufficient warp occupancy.
//
// When a warp stalls waiting for a global memory response, the SM scheduler
// switches to another eligible warp to keep the pipeline busy. This only works
// if there are enough resident warps. The base kernel's 4x4 block gives just
// half a warp per block, so the SM has almost nothing to switch to -- it stalls.
//
// This kernel uses a 16x16 block (256 threads = 8 warps). Combined with
// no artificial block limits, multiple blocks can reside per SM, giving the
// scheduler enough warps to overlap memory latency with useful computation.
//
// Effect on NCU report (compare to matmul_base):
//   - Achieved Occupancy        : increases  -- more warps resident per SM
//   - Compute (SM) Throughput   : increases  -- SM stalls less
//   - Duration                  : decreases  -- latency is hidden, not waited on
//
// Tunable: adjust BLOCK and observe how occupancy and stall time change.
// =============================================================================
#include <torch/extension.h>

#define TILE  0    // no tiling -- focus is on occupancy, not data reuse
#define BLOCK 16   // 16x16 = 256 threads = 8 warps per block

#if TILE > 0
  #define BDIM TILE
#else
  #define BDIM BLOCK
#endif

__global__ void matmul_latency(
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
    matmul_latency<<<grid, block>>>(
        A3.data_ptr<float>(), B3.data_ptr<float>(), C3.data_ptr<float>(), M, K, N);

    auto out = A.sizes().vec();
    out[out.size() - 1] = N;
    return C3.reshape(out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_matmul", &batched_matmul, "Latency-optimized matmul (occupancy)");
}
