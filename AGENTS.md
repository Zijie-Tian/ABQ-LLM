# ABQLLM environment

Extracted from `scripts/abqllm/README.md`.

The environment is named **`abqllm`** at
`/mnt/data/tzj/anaconda3/envs/abqllm`. It uses Python 3.10.22,
PyTorch 2.0.0+cu118, Transformers 4.35.0, CUDA Toolkit 11.8 and A100/sm80.
Attention is **eager**; FlashAttention and AutoGPTQ are not installed.

## Recreate the Python environment

This is an intentionally legacy research environment, not a recommendation to
deploy these historical package versions as an internet-facing service.

```bash
# Only create when this name does not already exist.
conda create -n abqllm python=3.10.22 pip -y
conda run -n abqllm python -m pip install --require-hashes \
  -r scripts/abqllm/artifacts-linux-py310.lock
```

`requirements-cu118.txt` is the human-readable derivative of
`algorithm/requirements.txt`. `artifacts-linux-py310.lock` pins the resolved
Linux/CPython 3.10 artifacts and SHA256 hashes; it is not portable to another
Python ABI or OS. It intentionally does not install the unrelated root
`requirements.txt`. Conda bootstrap exports and actual installed package
receipts are in `.omx/logs/abqllm-env/`.

Differences from the algorithm's original pins:

- `torch==2.0.0` becomes the official `torch==2.0.0+cu118` build. Its bundled CUDA
  runtime replaces the original explicit CUDA 11.7 `nvidia-*-cu11` pins.
- `ninja==1.11.1.1` becomes `1.11.1.4`. The old installed wheel's WHEEL metadata
  has a blank line before its Tag fields, and current pip reports it as
  unsupported. The replacement passes `pip check`; no installed metadata was
  patched and no dependency checks were suppressed.

## Run on this machine

From the repository root:

```bash
conda activate abqllm
# Prefer this launcher over inherited shell CUDA/PYTHONPATH settings:
bash scripts/abqllm/run.sh python scripts/abqllm/smoke_environment.py \
  --output .omx/logs/abqllm-env/environment-smoke.json
bash scripts/abqllm/run.sh python scripts/abqllm/smoke_python.py \
  --stage components,tiny_abq --out .omx/logs/abqllm-env/python-smoke.json
bash scripts/abqllm/run.sh python scripts/abqllm/smoke_python.py \
  --stage llama2_baseline --model /mnt/data/tzj/models/Llama-2-7b-hf \
  --out .omx/logs/abqllm-env/llama2-baseline.json
```

The launcher defaults to physical GPU1 by UUID; inside the process it is
`cuda:0`. Override with `ABQLLM_GPU=<GPU UUID or physical index>` when needed.
It does not stop existing GPU processes. CUDA, Python imports and caches are
scoped to this project; the parent shell and other Conda environments are not
changed. Downloads are disabled by default for model/dataset operations.
`ABQLLM_ENV_PREFIX` and `ABQLLM_CUDA_PREFIX` override the local paths.

Run the native correctness smoke separately:

```bash
bash scripts/abqllm/run.sh python scripts/abqllm/smoke_native.py
```

It uses the verified model-like shape **M=1, N=4096, K=4096**. Signed W2A8
BMMA/WMMA and the W4A4 CUTLASS comparison pass on this machine. Unsigned W2A8
has no instantiated test kernels and is reported `NOT_SUPPORTED`, not PASS.
The combined result is `PASS_WITH_NOT_SUPPORTED`; selecting only unsupported
cases exits 2. Early K=128/256 full-template sweeps failed and are retained in
the diagnostic logs: success at K=4096 is not a claim of arbitrary-shape support.

To rebuild the native binaries without the original all-target/high-parallelism
scripts:

```bash
git submodule update --init --depth 1 3rdparty/cutlass
bash scripts/abqllm/run.sh cmake -S engine -B .omx/build/engine-sm80 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=80 -DSM=80 \
  -DENABLE_W2A8=ON -DENABLE_W4A4=ON -DCMAKE_BUILD_TYPE=Release
bash scripts/abqllm/run.sh cmake --build .omx/build/engine-sm80 \
  --target test_any_mma test_any_wmma test_w4a4_int_wmma --parallel 8
```

The root CUTLASS gitlink is `cc85b64cf676c45f98a17e3a47c0aafcf817f088`.
Compilation alone is not the correctness gate; run the numeric tests afterward.

## What the checks mean

- Environment: target interpreter/imports, `pip check`, one visible A100,
  actual FP32/FP16 CUDA matrix multiplication against a CPU reference.
- Components: W16 passthrough; W8A8/W4A4/W2A8 fake quantization, manual numerical
  references, finite gradients, and a valid grouped-quantization case. Two-bit
  weights follow the repository's intentional bit-balance `qmax=4` behavior.
- Tiny ABQ: one-layer random Llama, hidden 128, 4 heads, vocab 256, **2048**
  tokens, 2 calibration samples, 1 epoch; W8A8 and W4A4 with LET/LWC/compensation
  updates, finite training metrics, saved calibration parameters and a
  post-quantization forward pass. Sequence length is retained because upstream
  calibration attention buffers are hardcoded to 2048.
- Llama-2 baseline: existing local 7B weights, FP16 eager, a short prompt and
  up to 8 new greedy tokens. **This is not quantized model inference.**
- Native: independently checked low-bit integer GEMM, not evidence that
  Python fake quant dispatches to these kernels.

No full-dataset PPL/task evaluation, 7B ABQ calibration, speedup claims, or
FasterTransformer quantized E2E are included. The existing FT W2A8 scale-loading
TODO and incomplete Python CUDA binding are not changed by this setup.
Upstream saved ABQ calibration parameters omit compensation tensors; the tiny
test checks those updates in memory and does not claim a full model round-trip.

All runners exit nonzero on failure. Keep failed logs as diagnostic evidence;
an upstream binary's exit code zero alone does not establish correctness.
