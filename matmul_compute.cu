// matmul_compute.cu  --  COMPUTE OPTIMIZED
// =============================================================================
// Optimization strategy: shared-memory tiling.
//
// The base kernel reads A and B from global memory on every multiply.
// Across a TILE x TILE block, the same row of A is needed by every thread in
// that row, and the same column of B is needed by every thread in that column.
// By loading one TILE x TILE tile of A and B into shared memory first, each
// element is fetched from global memory only once and reused TILE times.
//
// Effect on NCU report (compare to matmul_base):
//   - Compute (SM) Throughput   : increases  -- more FLOPs per global load
//   - Memory Throughput         : decreases  -- fewer global memory transactions
//   - Duration                  : decreases  -- more work done per cycle
//
// Tunable: adjust TILE and observe how the roofline balance shifts.
// =============================================================================
#include <torch/extension.h>

#define TILE  16   // shared-memory tile side length
#define BLOCK 16   // unused when TILE > 0

#if TILE > 0
  #define BDIM TILE
#else
  #define BDIM BLOCK
#endif

__global__ void matmul_compute(
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
    matmul_compute<<<grid, block>>>(
        A3.data_ptr<float>(), B3.data_ptr<float>(), C3.data_ptr<float>(), M, K, N);

    auto out = A.sizes().vec();
    out[out.size() - 1] = N;
    return C3.reshape(out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_matmul", &batched_matmul, "Compute-optimized matmul (tiling)");
}
