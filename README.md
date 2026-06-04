# GPU Kernel Profiling Lab
## Profiling and Optimizing the GPT-2 Attention Matrix Multiply with NVIDIA Nsight Compute

This lab explains how to profile and optimize the matrix-multiplication kernel that performs GPT-2 self-attention (QK᷊ and P·V). It covers why each kernel behaves the way it does and exactly which Nsight Compute (ncu) metrics respond to each change.

---

## Contents

1. [What This Lab Measures](#1-what-this-lab-measures)
2. [Prerequisites and Setup](#2-prerequisites-and-setup)
3. [Running a Kernel](#3-running-a-kernel)
4. [Profiling with Nsight Compute](#4-profiling-with-nsight-compute)
5. [How to Read the Report](#5-how-to-read-the-report)
6. [The Baseline, Dissected](#6-the-baseline-dissected)
7. [The Three Optimization Strategies](#7-the-three-optimization-strategies)
8. [Assignment — Optimize matmul_skeleton.cu](#8-assignment--optimize-matmul_skeletoncu)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. What This Lab Measures

All attention in this lab is computed in eager mode. The driver script (`run.py`) patches Hugging Face's GPT-2 eager attention so that the two attention matrix multiplies are routed through a kernel you select at runtime. Both multiplies are wrapped in an NVTX range named `matmul_op`, which lets ncu capture exactly those launches and nothing else.

Concretely, for each attention head the kernel computes a batched product `C[b] = A[b] · B[b]`, where A is M×K, B is K×N, and b indexes the batch (batch = heads × sequence groups). The sequence length is controlled by `--tokens`; longer sequences enlarge M, K, and N, which makes every bottleneck more visible in the report.

Six kernels share one identical numerical body. They differ only in launch geometry (block size), whether shared-memory tiling is enabled, and the kernel name. Because the math is identical, any difference you see in the ncu report is caused purely by how the work is mapped onto the GPU — which is the entire point of the lab.

| # | Kernel | What it isolates |
|---|--------|-----------------|
| 1 | eager_matmul | PyTorch `torch.matmul` (cuBLAS) — the fast reference |
| 2 | matmul_memory | Memory bandwidth: very large blocks (32×32) |
| 3 | matmul_latency | Latency hiding / occupancy: enough warps per SM (16×16) |
| 4 | matmul_compute | Block size **plus** shared-memory tiling (16×16, data reuse) |
| 5 | matmul_base | Naive baseline (1×1) — deliberately bad |
| 6 | matmul_skeleton | Your kernel — starts as a copy of the baseline |

### The cumulative optimization story

There are really only **two independent levers** in these kernels:

1. **Block size** — how many warps run per SM. Raising it fixes low occupancy, hides memory latency, and increases in-flight memory bandwidth, all at once.
2. **Shared-memory tiling** — data reuse. Independent of block size; it reduces the *amount* of data moved rather than how fast it moves.

`matmul_memory` and `matmul_latency` both pull the **same** lever (block size), just to different sizes, so they cannot be stacked — they are two points on one axis. The lab therefore tells the story with two cumulative steps, each pulling a different lever:

```
kernel 5 (base)         1×1 block, no tiling          ~47 ms   <- baseline
   │  raise block size (lever 1)
   ▼
kernel 3 (latency)      16×16 block, no tiling        ~1.4 ms  <- occupancy fixed
   │  add tiling on top (lever 2)
   ▼
kernel 4 (compute)      16×16 block + tiling          ~1.0 ms  <- reuse added
```

`matmul_compute` already uses a 16×16 block (because `TILE=16` sets the block dimension), so it carries lever 1 forward *and* adds lever 2 — it is the baseline with **both** optimizations applied.

`matmul_memory` (kernel 2, 32×32) is shown for its code only: it is the same lever as `matmul_latency` and actually regresses here (a 1024-thread block packs fewer blocks per SM and suffers tail effects), which is itself a useful lesson that "bigger block" is not monotonically better.

> **Magnitude note:** step 1 (base → latency) is dramatic (~33×). Step 2 (latency → compute) is modest (~1.4×): with GPT-2's small contraction dimension (K = 64) there is little data to reuse, so tiling's benefit is limited. Use **Duration** as the honest headline metric throughout.

---

## 2. Prerequisites and Setup

- An NVIDIA GPU with CUDA support
- Python 3.x with a CUDA-enabled PyTorch build
- NVIDIA Nsight Compute, with the `ncu` command on your PATH
- A host C++ compiler reachable by NVCC (MSVC/Visual Studio on Windows, GCC on Linux)

Verify the toolchain before profiling:
```bash
python -c "import torch; print(torch.cuda.is_available())"   # must print True
ncu --version
```

On Windows, kernels 2–6 are JIT-compiled by PyTorch on first use and require the MSVC environment (run from "x64 Native Tools Command Prompt for VS", or call `vcvars64.bat` first). First compilation takes roughly 30–60 seconds; later runs load from the build cache instantly.

---

## 3. Running a Kernel

```bash
python run.py <kernel_number> [--tokens N]
```

```bash
python run.py 5 --tokens 250   # naive baseline
python run.py 6 --tokens 250   # your kernel
```

The default `--tokens 250` repeats the seed phrase to produce roughly 1000 tokens (truncated to GPT-2's 1024-token context). A plain run does not profile anything — it confirms the kernel compiles and runs, then prints the exact ncu command to capture it.

---

## 4. Profiling with Nsight Compute

> **Windows:** run ncu from an **Administrator** terminal — GPU performance counters require elevated privileges.

```bash
ncu --nvtx --nvtx-include "matmul_op/" --set basic \
    --metrics smsp__thread_inst_executed_per_inst_executed.ratio -f -c 4 \
    -o <report_name> python run.py <number> --tokens 250
```

```sweep from 2 ~ 5
for %k in (2 3 4 5) do ncu --nvtx --nvtx-include "matmul_op/" --set basic --metrics smsp__thread_inst_executed_per_inst_executed.ratio -f -c 4 -o kernel%k_report python run.py %k --tokens 250
```
**Flag by flag:**

| Flag | Purpose |
|------|---------|
| `--nvtx --nvtx-include "matmul_op/"` | Capture only kernels launched inside the `matmul_op` NVTX range — the attention matmul and nothing else (no softmax, no copies) |
| `--set basic` | Collect the standard metric set: Speed-of-Light throughputs, occupancy, and launch statistics |
| `--metrics smsp__thread_inst_executed_per_inst_executed.ratio` | Add **warp execution efficiency** — active threads per warp instruction (out of 32). The honest "is the kernel wasting the GPU?" metric; the baseline reads ~1 (3%), an optimized kernel ~32 |
| `-c 4` | Capture only the first four matmul launches — enough to characterize the kernel, keeps profiling fast |
| `-f -o <name>` | Overwrite and write `<name>.ncu-rep`, opened in the Nsight Compute GUI |

Example — capture the baseline, then your kernel:
```bash
ncu --nvtx --nvtx-include "matmul_op/" --set basic --metrics smsp__thread_inst_executed_per_inst_executed.ratio -f -c 4 -o base_report     python run.py 5 --tokens 250
ncu --nvtx --nvtx-include "matmul_op/" --set basic --metrics smsp__thread_inst_executed_per_inst_executed.ratio -f -c 4 -o skeleton_report python run.py 6 --tokens 250
```

Profile all six kernels at once (cmd):
```cmd
for %k in (1 2 3 4 5 6) do ncu --nvtx --nvtx-include "matmul_op/" --set basic --metrics smsp__thread_inst_executed_per_inst_executed.ratio -f -c 4 -o kernel%k_report python run.py %k --tokens 250
```

---

## 5. How to Read the Report

Open the `.ncu-rep` file and read these three sections in order.

### 5.1 GPU Speed of Light (SOL)

The Speed-of-Light section reports what fraction of the hardware's peak each subsystem reached. It is your first diagnostic: a well-optimized kernel pushes at least one of these close to 100%, whereas the naive baseline leaves all of them low.

| Metric | Meaning |
|--------|---------|
| **Compute (SM) Throughput** | How busy the arithmetic units are, as a percent of peak |
| **Memory Throughput** | How busy the on-chip memory pipeline (L1/L2) is |
| **DRAM Throughput** | How much of raw device-memory bandwidth is consumed |
| **Duration** | Wall-clock time for the kernel launch |

Reading the pair matters more than either number alone. High memory throughput with low compute throughput means the kernel is memory-bound; the reverse means it is compute-bound. The baseline is neither — it is latency-bound, with everything low because the SM spends most of its cycles stalled and idle.

### 5.2 Occupancy

Occupancy is the ratio of active warps on an SM to the hardware maximum. Theoretical occupancy is the ceiling set by your launch configuration and per-thread resource use; achieved occupancy is what actually ran. A large gap between them, or a low theoretical ceiling, tells you why the SM sat idle.

- **Low because of block size:** too few threads per block (the baseline's 16) cannot fill the SM.
- **Low because of registers:** high registers-per-thread limits how many blocks fit at once.
- **Low because of shared memory:** a large tile can cap the number of resident blocks per SM.

Occupancy is not a goal in itself — it is a means to hide latency. Enough warps let the scheduler swap a stalled warp for a ready one, keeping the pipelines fed.

### 5.3 Launch Statistics

- **Block Size** — small values explain poor warp utilization. The baseline's 1×1 block is the root cause of its slowness: each warp runs with only 1 of 32 lanes active.
- **Registers Per Thread** — high values reduce how many blocks fit per SM, capping theoretical occupancy.
- **Waves Per SM** — how many rounds of blocks the GPU processed. A fractional last wave means the grid does not evenly fill the device (the "tail").

---

## 6. The Baseline, Dissected

Every optimized kernel is a one-line change away from `matmul_base.cu`. Understanding the baseline precisely is therefore the key to the whole lab. The inner loop is:

```cuda
// BDIM = BLOCK = 1: one thread per block
int row = blockIdx.y * BDIM + threadIdx.y;
int col = blockIdx.x * BDIM + threadIdx.x;
if (row >= M || col >= N) return;
float sum = 0.0f;
for (int k = 0; k < K; k++)
    sum += a[row * K + k] * bm[k * N + col];   // two global loads per iteration
c[row * N + col] = sum;
```

The baseline is slow for two distinct reasons, each addressed by one of the two optimization levers:

**Problem 1 — Catastrophic warp under-utilization (fixed by block size).** `BLOCK = 1` means 1 thread per block. The hardware always executes full 32-thread warps, so each warp runs with only **1 of its 32 lanes active** — about 3% warp execution efficiency. The other 31 lanes are masked off every cycle. Equivalently, almost no independent work is available per warp, so when a thread stalls on a ~hundreds-of-cycles global load, the scheduler has little else to run. This single defect dominates the baseline's ~47 ms runtime.

**Problem 2 — No data reuse (fixed by tiling).** Each output element streams an entire row of A and an entire column of B from global memory, and neighboring outputs re-fetch the same rows and columns. The same bytes cross the memory bus many times. Tiling is what removes this; block size cannot.

> **Read the right metric.** It is tempting to point at *occupancy* to explain why the baseline is bad, but achieved occupancy here is a deceptively moderate ~49 % — the SM packs many 1-thread blocks, so it looks half-full of warps even though each warp is 31/32 idle. The honest headline is **Duration** (≈47 ms vs ≈1 ms optimized), and the supporting metric is **warp execution efficiency**, not occupancy.

---

## 7. The Optimization Strategy (Two Cumulative Steps)

The lab demonstrates optimization as two cumulative steps, each pulling one of the two independent levers. Profile them in this order against the baseline.

### Step 1 — Fix warp utilization with block size → `matmul_latency` (kernel 3)

**Change from baseline:** `BLOCK = 1` → `BLOCK = 16`. A 16×16 block is 256 threads = 8 full warps.

**What it fixes:** Problem 1. Now every warp runs with all 32 lanes active, and several blocks reside per SM, giving the scheduler a deep pool of eligible warps. While one warp waits on a global load, another runs — stall cycles are overlapped with useful work instead of wasted.

**Measured on an RTX 5060 (seq ≈ 1001, `--tokens 250`):**

| Metric | base | latency | change |
|--------|---:|---:|:--|
| Warp execution efficiency | 1.00 / 32 | 31.86 / 32 | 3% → **99.6%** |
| Duration | 47.18 ms | 1.39 ms | **~34× faster** |
| Achieved Occupancy | 48.8% | 94.2% | +45 pp |
| DRAM Throughput | 0.8% | 8.4% | memory finally used |

**Key metric to watch:** `warp execution efficiency` (the smoking gun) and `Duration`.

---

### Step 2 — Add data reuse on top with tiling → `matmul_compute` (kernel 4)

**Change from step 1:** keep the 16×16 block, **add** shared-memory tiling (`TILE = 16` sets both the tile and the block dimension). This kernel is the baseline with **both** levers applied.

**What it fixes:** Problem 2, on top of step 1. The K dimension is walked in 16-wide tiles. Each iteration cooperatively loads one 16×16 tile of A and one of B into shared memory, the block synchronizes, then every thread reuses those cached values for 16 multiply-accumulates before loading the next tile:

```cuda
for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
    sA[ty][tx] = (row < M && aCol < K) ? a[row*K + aCol] : 0.0f;
    sB[ty][tx] = (bRow < K && col < N) ? bm[bRow*N + col] : 0.0f;
    __syncthreads();                     // wait until the tile is fully loaded
    for (int k = 0; k < TILE; k++)       // 16 MACs from fast shared memory
        sum += sA[ty][k] * sB[k][tx];
    __syncthreads();                     // wait before overwriting the tile
}
```

Each global element is now read once and reused TILE times, raising arithmetic intensity (FLOPs per byte). The two `__syncthreads()` calls are mandatory: the first guarantees the tile is fully written before any thread reads it; the second prevents a thread from overwriting the tile while others still read it.

**Measured on an RTX 5060 (seq ≈ 1001, `--tokens 250`):**

| Metric | latency | compute | change |
|--------|---:|---:|:--|
| Duration | 1.39 ms | 1.02 ms | **~1.36× faster** |
| Compute (SM) Throughput | 92.8% | 95.8% | +3 pp |
| Achieved Occupancy | 94.2% | 97.2% | +3 pp |
| Warp execution efficiency | 31.86 | 31.99 | already saturated |

The step-2 gain is modest because GPT-2's contraction dimension (K = 64) leaves little to reuse — a useful lesson about when tiling pays off. Note that SOL *Memory Throughput* does **not** drop here (shared-memory accesses still register on the L1 pipe); the honest step-2 signals are `Duration` and `Compute (SM) Throughput`.
**Key metric to watch:** `Compute (SM) Throughput` and `Duration`

---

### Side note — `matmul_memory` (kernel 2): same lever, shown as code only

`matmul_memory` sets `BLOCK = 32` (1024 threads). This pulls the **same** lever as step 1 (block size), just larger — so it is *not* a separate optimization to stack. In fact it regresses here: a 1024-thread block packs fewer blocks per SM and suffers tail effects, giving **lower** occupancy (62.3% vs 94.2%) and a **longer** duration (1.56 ms vs 1.39 ms) than the 16×16 `matmul_latency`, even though its warp efficiency is just as good (31.65 / 32). It is included so you can read its code and see that "bigger block" is not monotonically better — there is a sweet spot.

**Summary table (measured, RTX 5060, seq ≈ 1001):**

| Step | Kernel | Change | Duration | Warp eff. | Occupancy | SM Tput |
|------|--------|--------|---:|---:|---:|---:|
| baseline | matmul_base | `BLOCK = 1` | 47.18 ms | 1.00 | 48.8% | 90.2% |
| 1 | matmul_latency | `BLOCK = 16` | 1.39 ms | 31.86 | 94.2% | 92.8% |
| 2 | matmul_compute | + `TILE = 16` | 1.02 ms | 31.99 | 97.2% | 95.8% |
| (aside) | matmul_memory | `BLOCK = 32` | 1.56 ms | 31.65 | 62.3% | 83.3% |

> Numbers are from one RTX 5060 run; expect variation across GPUs, but the *ordering* and the ~34× / ~1.36× step pattern should hold.

---

## 8. Assignment — Optimize `matmul_skeleton.cu`

`matmul_skeleton.cu` begins as an exact copy of the naive baseline. Your task is to make it measurably faster than `matmul_base` and to explain why it got faster using the report.

Everything below the `#include` line is yours to change. The two parameters at the top are your starting point:

```c
#define TILE  0    // 0: no tiling  |  N>0: shared-memory tile of size NxN
#define BLOCK 4    // block side length (used when TILE == 0)
```

### 8.1 Recommended Procedure

1. Profile `matmul_base` (kernel 5) and record Duration, SM throughput, Memory throughput, and Achieved Occupancy as your baseline.
2. Profile the two cumulative steps (kernel 3, then kernel 4) and confirm on your own hardware: block size first collapses Duration, then tiling trims it further.
3. Edit `matmul_skeleton.cu`. Apply the levers in the same order: first raise `BLOCK` (e.g. 16) to fix warp utilization — this is the single biggest win. Then, for a further gain, enable tiling (`TILE = 16`) to add data reuse on top.
4. Recompile (see below), then profile `matmul_skeleton` (kernel 6) and compare against your baseline.
5. Write up what you changed, which metrics moved, and the causal reason for each change.

### 8.2 Recompiling After Each Edit

PyTorch caches the compiled extension. After editing the `.cu` file you must clear that kernel's cache, or your change will not take effect:

```cmd
rmdir /s /q "%LOCALAPPDATA%\torch_extensions\torch_extensions\Cache\py312_cu128\matmul_skeleton"
python run.py 6 --tokens 250
```

> The `py312_cu128` segment encodes your Python and CUDA versions; adjust it if yours differ.

### 8.3 Submission Checklist

- [ ] `python run.py 6` completes without errors
- [ ] The ncu report shows the `matmul_skeleton` kernel being captured
- [ ] At least one Speed-of-Light or occupancy metric is better than `matmul_base`
- [ ] A written explanation: what you changed, what the report shows, and why the numbers moved

---

## 9. Troubleshooting

### Compilation hangs forever
A run killed with Ctrl+C can leave a stale lock file that blocks the next build. `run.py` deletes these automatically on startup; if it still hangs, close and reopen the terminal.

### ncu permission error on Windows
Run the terminal as Administrator — GPU performance counters require elevated privileges.

### `torch.cuda.is_available()` returns False
You have a CPU-only PyTorch build, or its CUDA version does not match your driver. Check your install:
```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```
