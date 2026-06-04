// matmul_base.cu  --  NAIVE BASELINE (intentionally unoptimized)
// =============================================================================
// This is the starting reference kernel. It is intentionally written in the
// most naive way possible: no data reuse and a 1-thread block, so every warp
// runs with only 1 of its 32 lanes active (~3% warp execution efficiency).
//
// Students compare the optimized kernels against this baseline to observe what
// each optimization strategy improves. The honest headline metric here is
// Duration (this kernel is ~30-40x slower than an optimized one), driven by
// catastrophic warp under-utilization -- NOT occupancy (the SM still packs
// many tiny blocks, so achieved occupancy looks deceptively moderate).
//
// Tunable parameters (change TILE or BLOCK, delete cache, then rerun):
//   #define TILE  -- 0: no tiling  |  N>0: shared-memory tile of size NxN
//   #define BLOCK -- block side length when TILE==0
// =============================================================================
#include <torch/extension.h>

#define TILE  0   // no tiling -- every MAC goes to global memory
#define BLOCK 1   // 1x1 = 1 thread per block -- only 1 of 32 warp lanes active

#if TILE > 0
  #define BDIM TILE
#else
  #define BDIM BLOCK
#endif

__global__ void matmul_base(
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
    matmul_base<<<grid, block>>>(
        A3.data_ptr<float>(), B3.data_ptr<float>(), C3.data_ptr<float>(), M, K, N);

    auto out = A.sizes().vec();
    out[out.size() - 1] = N;
    return C3.reshape(out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_matmul", &batched_matmul, "Naive baseline matmul");
}
