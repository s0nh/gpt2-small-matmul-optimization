"""
run.py
=============================================================================
Profiling driver for GPT-2 attention matmul kernels.

Runs one eager GPT-2 forward pass with the chosen matmul kernel replacing
attention's QK^T and P@V operations. The matmul calls are wrapped in an NVTX
range ("matmul_op") so Nsight Compute captures exactly those kernels.

Kernel menu:
    1  eager_matmul    stock torch.matmul (cuBLAS/MAGMA) -- no compile
    2  matmul_memory   same lever as #3, larger block; code only  (matmul_memory.cu,  TILE=0 BLOCK=32)
    3  matmul_latency  step 1: raise block size -> warp util/occupancy  (matmul_latency.cu, TILE=0 BLOCK=16)
    4  matmul_compute  step 2: block size + tiling (data reuse)         (matmul_compute.cu, TILE=16)
    5  matmul_base     naive baseline (1 thread/block)                  (matmul_base.cu,    TILE=0 BLOCK=1)
    6  matmul_skeleton YOUR kernel                                      (matmul_skeleton.cu -- edit this)

Cumulative demo order: 5 (base) -> 3 (block size) -> 4 (+ tiling).

Kernels 2-6 JIT-compile on first run (~30-60 s); subsequent runs use cache.
Kernel 1 compiles nothing.

-----------------------------------------------------------------------------
HOW TO RUN
-----------------------------------------------------------------------------
Plain run (verify it works, prints the ncu command):
    python run.py 5

Profile a kernel with Nsight Compute (Admin terminal on Windows):
    ncu --nvtx --nvtx-include "matmul_op/" --set basic -f -c 4 ^
        -o base_report python run.py 5 --tokens 250

Profile all six kernels in one sweep (cmd):
    for %k in (1 2 3 4 5 6) do ncu --nvtx --nvtx-include "matmul_op/" --set basic -f -c 4 -o kernel%k_report python run.py %k --tokens 250

  --nvtx --nvtx-include "matmul_op/"  capture only kernels inside the matmul_op range
  --set basic                         standard metric set (SM/Memory throughput, occupancy)
  -c 4                                profile only the first 4 kernel launches (faster)
  -o <name>                           write <name>.ncu-rep (open in Nsight Compute UI)
=============================================================================
"""
import os
import sys
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import torch.utils.cpp_extension as _cpp_ext
from torch.utils.cpp_extension import load

if not torch.cuda.is_available():
    raise SystemExit("This script requires a CUDA GPU. None detected.")

_cpp_ext.SUBPROCESS_DECODE_ARGS = ("utf-8", "ignore")
ROOT = os.path.dirname(os.path.abspath(__file__))

# 이전 실행이 Ctrl+C로 중단됐을 때 남은 스테일 락 파일을 자동으로 삭제합니다.
# 락 파일이 남아 있으면 다음 실행 시 컴파일이 무한 대기 상태에 빠집니다.
def _clear_stale_locks():
    removed = []
    for lock in Path(_cpp_ext.get_default_build_root()).rglob("lock"):
        lock.unlink(missing_ok=True)
        removed.append(lock.parent.name)
    if removed:
        print(f"[lock] 스테일 락 삭제됨: {', '.join(removed)}", flush=True)

_clear_stale_locks()

# ── Kernel catalogue ──────────────────────────────────────────────────────────
# Each custom variant is its OWN self-contained .cu file. The three bound kernels
# share an identical body template; they differ only in TILE size, BLOCK size,
# the presence of __launch_bounds__, and the kernel name. Option 5 points at a
# separate file you will write later.
# /utf-8 is REQUIRED: the .cu files contain Korean comments, and on a Korean
# Windows MSVC otherwise reads them as CP949 and miscounts multibyte chars,
# swallowing the following line of code. /utf-8 makes MSVC read them as UTF-8.
WIN_CFLAGS = ["-allow-unsupported-compiler", "-Xcompiler", "/Zc:preprocessor", "-Xcompiler", "/utf-8",
              "--threads", "4"]

# Warp execution efficiency: average active threads per executed warp instruction
# (0-32). This is the honest "is the kernel wasting the GPU?" metric -- a 1-thread
# block reads ~1 here (3%), even though SOL throughput % can look deceptively high.
# Not part of --set basic, so we add it explicitly to the recommended ncu command.
WARP_EFF_METRIC = "smsp__thread_inst_executed_per_inst_executed.ratio"
WARP_EFF_LABEL  = "warp execution efficiency"

KERNELS = {
    1: {"name": "eager_matmul",   "source": None},
    2: {"name": "matmul_memory",  "source": "matmul_memory.cu"},
    3: {"name": "matmul_latency", "source": "matmul_latency.cu"},
    4: {"name": "matmul_compute", "source": "matmul_compute.cu"},
    5: {"name": "matmul_base",     "source": "matmul_base.cu"},
    6: {"name": "matmul_skeleton", "source": "matmul_skeleton.cu"},
}


def choose_kernel(argv_value):
    if argv_value is not None:
        return int(argv_value)
    print("\nSelect the matmul kernel to run:")
    for i, k in KERNELS.items():
        print(f"  {i}  {k['name']}")
    while True:
        raw = input("Enter number [1-6] (default 1): ").strip() or "1"
        if raw.isdigit() and int(raw) in KERNELS:
            return int(raw)
        print("  invalid choice, try again.")


def build_matmul_fn(choice):
    """Return (label, matmul_fn). For option 1, matmul_fn is torch.matmul."""
    cfg = KERNELS[choice]
    name = cfg["name"]
    if cfg["source"] is None:                       # option 1: stock eager matmul
        print(f"[kernel] {name}: using stock torch.matmul (no compile).", flush=True)
        return name, torch.matmul

    src = os.path.join(ROOT, cfg["source"])
    if not os.path.exists(src):
        raise SystemExit(
            f"[kernel] {name}: source '{cfg['source']}' not found.\n"
            f"         Option {choice} expects {cfg['source']} exposing batched_matmul(A, B)\n"
            f"         with its CUDA kernel named '{name}'. (You said you'd write matmul_base later.)"
        )
    print(f"[kernel] {name}: compiling {cfg['source']} (first time ~30-60s, then cached) ...", flush=True)
    ext = load(name=name, sources=[src], extra_cuda_cflags=WIN_CFLAGS, verbose=False)
    print("[kernel] compile done.", flush=True)
    return name, ext.batched_matmul


# ── Patch eager attention so attention's matmuls use the chosen kernel ────────
# A module-global flag gates the NVTX range so ONLY the profiled pass's matmuls
# are bracketed (the warmup pass is left out of the report).
CAPTURE = False


def make_attention(matmul_fn):
    is_custom = matmul_fn is not torch.matmul

    def attn(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
        if scaling is None:
            scaling = query.size(-1) ** -0.5
        # Make the transposed key contiguous OUTSIDE the timed range, so the
        # custom kernel's internal .contiguous() copy is a no-op and the
        # matmul_op range holds ONLY the matmul kernel (no copy/fill kernels).
        kt = key.transpose(-1, -2)
        if is_custom:
            kt = kt.contiguous()
        if CAPTURE: torch.cuda.nvtx.range_push("matmul_op")
        qk = matmul_fn(query, kt)                                             # QK^T
        if CAPTURE: torch.cuda.nvtx.range_pop()
        attn_weights = qk * scaling                                          # scaling: outside range
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1).type(value.dtype)
        attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
        if CAPTURE: torch.cuda.nvtx.range_push("matmul_op")
        attn_output = matmul_fn(attn_weights, value)                          # P @ V
        if CAPTURE: torch.cuda.nvtx.range_pop()
        attn_output = attn_output.transpose(1, 2)
        return attn_output, attn_weights
    return attn


def precompile_all():
    """Compile all .cu kernels upfront so later runs are instant (cached)."""
    print("Pre-compiling all kernels (2-4). This runs once; subsequent runs use the cache.", flush=True)
    for choice in [2, 3, 4]:
        cfg = KERNELS[choice]
        src = os.path.join(ROOT, cfg["source"])
        if not os.path.exists(src):
            print(f"  [skip] {cfg['name']}: {cfg['source']} not found", flush=True)
            continue
        print(f"  [{choice}] {cfg['name']}: compiling ...", flush=True)
        load(name=cfg["name"], sources=[src], extra_cuda_cflags=WIN_CFLAGS, verbose=False)
        print(f"  [{choice}] {cfg['name']}: done", flush=True)
    print("Pre-compile complete.", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Profile a chosen attention-matmul kernel in eager GPT-2.")
    ap.add_argument("kernel", nargs="?", default=None, help="kernel number 1-6 (prompts if omitted)")
    ap.add_argument("--tokens", type=int, default=250, help="repeats of the prompt phrase (controls seq length, default ~1000 tokens)")
    ap.add_argument("--precompile", action="store_true", help="compile all .cu kernels upfront then exit")
    args = ap.parse_args()

    if args.precompile:
        precompile_all()
        return

    choice = choose_kernel(args.kernel)
    if choice not in KERNELS:
        raise SystemExit(f"kernel must be one of {list(KERNELS)}")

    print(f"Device : {torch.cuda.get_device_name(0)}", flush=True)
    label, matmul_fn = build_matmul_fn(choice)

    import transformers.models.gpt2.modeling_gpt2 as gpt2_module
    from transformers import GPT2Model, GPT2Tokenizer
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()

    gpt2_module.eager_attention_forward = make_attention(matmul_fn)

    device = "cuda"
    model = GPT2Model.from_pretrained("gpt2", attn_implementation="eager").to(device).eval()
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    inputs = tok("Once upon a time " * args.tokens, return_tensors="pt",
                 max_length=1024, truncation=True).to(device)
    seq = inputs["input_ids"].shape[1]
    print(f"Kernel : {label}   |   seq length : {seq} tokens", flush=True)

    global CAPTURE
    with torch.no_grad():
        model(**inputs); torch.cuda.synchronize()         # warmup (NOT captured)
        CAPTURE = True
        print("Running profiled forward pass (matmul_op range active) ...", flush=True)
        model(**inputs); torch.cuda.synchronize()         # profiled pass
        CAPTURE = False
    print("Done.", flush=True)

    rep = f"{label}_report"
    print("\n" + "-" * 70, flush=True)
    print("To profile THIS selection with Nsight Compute, run:", flush=True)
    print(f'  ncu --nvtx --nvtx-include "matmul_op/" --set basic \\', flush=True)
    print(f'      --metrics {WARP_EFF_METRIC} -f -c 4 \\', flush=True)
    print(f'      -o {rep} python {os.path.basename(__file__)} {choice}', flush=True)
    print(f'  ({WARP_EFF_LABEL}: active threads per warp instruction, out of 32 --', flush=True)
    print(f'   the honest "is this kernel wasting the GPU?" metric; ~1 means catastrophic)', flush=True)
    if choice == 1:
        print(f"  (look for the stock GEMM kernel, e.g. magma_sgemmEx_kernel / gemmSN_* / cutlass)", flush=True)
    else:
        print(f"  (look for the kernel named '{label}' in the report)", flush=True)
    print("-" * 70, flush=True)


if __name__ == "__main__":
    main()
