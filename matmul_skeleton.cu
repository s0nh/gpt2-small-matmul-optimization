// matmul_skeleton.cu
// =============================================================================
// Tensor Core / Shared Memory configurable matmul skeleton
//
// Modes:
//   USE_TENSOR_CORE = 1:
//      - input  : torch.float16
//      - compute: Tensor Core WMMA, FP16 input + FP32 accumulation
//      - output : torch.float16
//
//   USE_TENSOR_CORE = 0:
//      - input/output: torch.float32
//      - if TILE > 0: FP32 shared-memory tiled matmul
//      - if TILE == 0: FP32 naive matmul
//
// Recommended first try on RTX 5080:
//   #define USE_TENSOR_CORE 1
//   #define BLOCK 128
//   #define TILE 16
//
// Notes:
//   - BLOCK is the number of threads per CUDA block in Tensor Core mode.
//   - BLOCK must be a multiple of 32 in Tensor Core mode.
//   - Each warp computes one 16x16 output tile.
// =============================================================================

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// -----------------------------------------------------------------------------
// Tunable parameters
// -----------------------------------------------------------------------------
#define USE_TENSOR_CORE 1

// Tensor Core WMMA uses 16x16x16 tiles.
// Keep TILE = 16 for WMMA.
#define TILE 16

// Tensor Core mode:
//   BLOCK = total threads per CUDA block.
//   128 means 4 warps per block.
//   256 means 8 warps per block.
// FP32 fallback mode:
//   BLOCK is used as block side length only when TILE == 0.
#define BLOCK 96

#if USE_TENSOR_CORE
static_assert(TILE == 16, "WMMA Tensor Core path requires TILE == 16.");
static_assert(BLOCK % 32 == 0, "Tensor Core path requires BLOCK to be a multiple of 32.");
#define WARPS_PER_BLOCK (BLOCK / 32)
#endif

#if !USE_TENSOR_CORE
#if TILE > 0
#define BDIM TILE
#else
#define BDIM BLOCK
#endif
#endif

// -----------------------------------------------------------------------------
// Tensor Core WMMA kernel
//
// A: [batch, M, K], half
// B: [batch, K, N], half
// C: [batch, M, N], half
//
// Each warp computes one C tile of shape 16x16.
// Shared memory is used to pad boundary tiles safely.
// -----------------------------------------------------------------------------
#if USE_TENSOR_CORE
__global__ void matmul_skeleton_tc(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int K, int N)
{
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;

    const int batch_id = blockIdx.y;

    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    const int num_tiles_m = (M + WMMA_M - 1) / WMMA_M;
    const int num_tiles_n = (N + WMMA_N - 1) / WMMA_N;
    const int total_tiles = num_tiles_m * num_tiles_n;

    const int tile_linear = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (tile_linear >= total_tiles) return;

    const int tile_m = tile_linear / num_tiles_n;
    const int tile_n = tile_linear % num_tiles_n;

    const int row_base = tile_m * WMMA_M;
    const int col_base = tile_n * WMMA_N;

    const half* a = A + batch_id * M * K;
    const half* b = B + batch_id * K * N;
    half* c = C + batch_id * M * N;

    // Per-warp shared-memory tiles.
    // A tile: WARPS_PER_BLOCK * 16 * 16 half
    // B tile: WARPS_PER_BLOCK * 16 * 16 half
    // C tile: WARPS_PER_BLOCK * 16 * 16 float
    __shared__ half shmem_a[WARPS_PER_BLOCK][WMMA_M * WMMA_K];
    __shared__ half shmem_b[WARPS_PER_BLOCK][WMMA_K * WMMA_N];
    __shared__ float shmem_c[WARPS_PER_BLOCK][WMMA_M * WMMA_N];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    for (int k0 = 0; k0 < K; k0 += WMMA_K) {
        // Load A and B tiles into shared memory with zero padding.
        // One warp cooperatively copies its own A/B tile.
        for (int idx = lane_id; idx < WMMA_M * WMMA_K; idx += 32) {
            int local_r = idx / WMMA_K;
            int local_c = idx % WMMA_K;

            int global_r = row_base + local_r;
            int global_c = k0 + local_c;

            if (global_r < M && global_c < K) {
                shmem_a[warp_id][idx] = a[global_r * K + global_c];
            } else {
                shmem_a[warp_id][idx] = __float2half(0.0f);
            }
        }

        for (int idx = lane_id; idx < WMMA_K * WMMA_N; idx += 32) {
            int local_r = idx / WMMA_N;
            int local_c = idx % WMMA_N;

            int global_r = k0 + local_r;
            int global_c = col_base + local_c;

            if (global_r < K && global_c < N) {
                shmem_b[warp_id][idx] = b[global_r * N + global_c];
            } else {
                shmem_b[warp_id][idx] = __float2half(0.0f);
            }
        }

        __syncwarp();

        wmma::load_matrix_sync(a_frag, shmem_a[warp_id], WMMA_K);
        wmma::load_matrix_sync(b_frag, shmem_b[warp_id], WMMA_N);

        // Tensor Core MMA:
        // acc_frag = a_frag * b_frag + acc_frag
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

        __syncwarp();
    }

    // Store accumulator to shared memory first, then copy only valid elements.
    wmma::store_matrix_sync(shmem_c[warp_id], acc_frag, WMMA_N, wmma::mem_row_major);

    __syncwarp();

    for (int idx = lane_id; idx < WMMA_M * WMMA_N; idx += 32) {
        int local_r = idx / WMMA_N;
        int local_c = idx % WMMA_N;

        int global_r = row_base + local_r;
        int global_c = col_base + local_c;

        if (global_r < M && global_c < N) {
            c[global_r * N + global_c] = __float2half(shmem_c[warp_id][idx]);
        }
    }
}
#endif

// -----------------------------------------------------------------------------
// FP32 fallback kernel
// USE_TENSOR_CORE = 0 only
// -----------------------------------------------------------------------------
#if !USE_TENSOR_CORE
__global__ void matmul_skeleton_fp32(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N)
{
    int bidx = blockIdx.z;
    int row = blockIdx.y * BDIM + threadIdx.y;
    int col = blockIdx.x * BDIM + threadIdx.x;

    const float* a = A + bidx * M * K;
    const float* bm = B + bidx * K * N;
    float* c = C + bidx * M * N;

#if TILE > 0
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        sA[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? a[row * K + a_col] : 0.0f;

        sB[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? bm[b_row * N + col] : 0.0f;

        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE; k++) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        c[row * N + col] = sum;
    }

#else
    if (row >= M || col >= N) return;

    float sum = 0.0f;

    for (int k = 0; k < K; k++) {
        sum += a[row * K + k] * bm[k * N + col];
    }

    c[row * N + col] = sum;
#endif
}
#endif

// -----------------------------------------------------------------------------
// PyTorch binding
// -----------------------------------------------------------------------------
torch::Tensor batched_matmul(torch::Tensor A, torch::Tensor B)
{
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dim() >= 2 && B.dim() >= 2, "Inputs must be at least 2D tensors");
    TORCH_CHECK(A.size(-1) == B.size(-2), "Shape mismatch: A[..., M, K] x B[..., K, N]");

    auto A3 = A.contiguous().reshape({-1, A.size(-2), A.size(-1)});
    auto B3 = B.contiguous().reshape({-1, B.size(-2), B.size(-1)});

    int batch = A3.size(0);
    int M = A3.size(1);
    int K = A3.size(2);
    int N = B3.size(2);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

#if USE_TENSOR_CORE
    TORCH_CHECK(A3.scalar_type() == torch::kHalf, "Tensor Core mode requires A to be torch.float16");
    TORCH_CHECK(B3.scalar_type() == torch::kHalf, "Tensor Core mode requires B to be torch.float16");

    auto C3 = torch::empty({batch, M, N}, A.options());

    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;

    int num_tiles_m = (M + WMMA_M - 1) / WMMA_M;
    int num_tiles_n = (N + WMMA_N - 1) / WMMA_N;
    int total_tiles = num_tiles_m * num_tiles_n;

    dim3 block(BLOCK);
    dim3 grid((total_tiles + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, batch);

    matmul_skeleton_tc<<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(A3.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(B3.data_ptr<at::Half>()),
        reinterpret_cast<half*>(C3.data_ptr<at::Half>()),
        M, K, N
    );

#else
    TORCH_CHECK(A3.scalar_type() == torch::kFloat32, "FP32 mode requires A to be torch.float32");
    TORCH_CHECK(B3.scalar_type() == torch::kFloat32, "FP32 mode requires B to be torch.float32");

    auto C3 = torch::empty({batch, M, N}, A.options());

    dim3 block(BDIM, BDIM);
    dim3 grid((N + BDIM - 1) / BDIM, (M + BDIM - 1) / BDIM, batch);

    matmul_skeleton_fp32<<<grid, block, 0, stream>>>(
        A3.data_ptr<float>(),
        B3.data_ptr<float>(),
        C3.data_ptr<float>(),
        M, K, N
    );
#endif

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto out = A.sizes().vec();
    out[out.size() - 1] = N;
    return C3.reshape(out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("batched_matmul", &batched_matmul, "Student matmul kernel");
}
