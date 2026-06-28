#include <torch/extension.h>
#include <cuda_fp16.h>

#define TILE  0
#define BLOCK 1

#if TILE > 0
  #define BDIM TILE
#else
  #define BDIM BLOCK
#endif

__global__ void matmul_base_fp32(
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
    for (int k = 0; k < K; k++) {
        sum += a[row * K + k] * bm[k * N + col];
    }

    c[row * N + col] = sum;
}

__global__ void matmul_base_fp16(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int K, int N)
{
    int b   = blockIdx.z;
    int row = blockIdx.y * BDIM + threadIdx.y;
    int col = blockIdx.x * BDIM + threadIdx.x;

    const half* a  = A + b * M * K;
    const half* bm = B + b * K * N;
    half*       c  = C + b * M * N;

    if (row >= M || col >= N) return;

    // FP16 input, FP32 accumulate
    float sum = 0.0f;

    for (int k = 0; k < K; k++) {
        float av = __half2float(a[row * K + k]);
        float bv = __half2float(bm[k * N + col]);
        sum += av * bv;
    }

    // output은 FP16으로 저장
    c[row * N + col] = __float2half(sum);
}

torch::Tensor batched_matmul(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.scalar_type() == B.scalar_type(), "A and B must have same dtype");

    auto A3 = A.contiguous().reshape({-1, A.size(-2), A.size(-1)});
    auto B3 = B.contiguous().reshape({-1, B.size(-2), B.size(-1)});

    int batch = A3.size(0);
    int M = A3.size(1);
    int K = A3.size(2);
    int N = B3.size(2);

    auto C3 = torch::empty({batch, M, N}, A.options());

    dim3 block(BDIM, BDIM);
    dim3 grid((N + BDIM - 1) / BDIM, (M + BDIM - 1) / BDIM, batch);

    if (A3.scalar_type() == torch::kFloat32) {
        matmul_base_fp32<<<grid, block>>>(
            A3.data_ptr<float>(),
            B3.data_ptr<float>(),
            C3.data_ptr<float>(),
            M, K, N
        );
    }
    else if (A3.scalar_type() == torch::kHalf) {
        matmul_base_fp16<<<grid, block>>>(
            reinterpret_cast<const half*>(A3.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(B3.data_ptr<at::Half>()),
            reinterpret_cast<half*>(C3.data_ptr<at::Half>()),
            M, K, N
        );
    }
    else {
        TORCH_CHECK(false, "Only float32 and float16 are supported");
    }

    auto out = A.sizes().vec();
    out[out.size() - 1] = N;
    return C3.reshape(out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_matmul", &batched_matmul, "Naive baseline matmul");
}
