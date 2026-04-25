# FLUX.2-dev-NVFP4 White Image Bug: Two-Day Postmortem

**Date**: 2025-04-21 – 2025-04-22  
**Branch**: `fix/nvfp4-nibble-swap-default`  
**Author**: Johnson (johnson@together.ai)  
**Model**: `black-forest-labs/FLUX.2-dev-NVFP4`  
**Hardware**: NVIDIA B200 (Blackwell, sm_100)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Background](#2-background)
   - 2.1 [FLUX.2 Model Architecture](#21-flux2-model-architecture)
   - 2.2 [NVFP4 Quantization Format](#22-nvfp4-quantization-format)
   - 2.3 [The Mixed Checkpoint Design](#23-the-mixed-checkpoint-design)
   - 2.4 [FP4 GEMM Backends: CUTLASS vs flashinfer/cuDNN](#24-fp4-gemm-backends-cutlass-vs-flashinfercudnn)
   - 2.5 [TMA (Tensor Memory Accelerator) Weight Layout](#25-tma-tensor-memory-accelerator-weight-layout)
3. [Symptom and Environment](#3-symptom-and-environment)
4. [Root Cause 1: Wrong Nibble Swap Default](#4-root-cause-1-wrong-nibble-swap-default)
5. [Root Cause 2: Missing Input Scales in Mixed Checkpoint](#5-root-cause-2-missing-input-scales-in-mixed-checkpoint)
6. [Root Cause 3: TMA Permutation Applied to Wrong Backend](#6-root-cause-3-tma-permutation-applied-to-wrong-backend)
7. [cuDNN mm_fp4 Verification Methodology](#7-cudnn-mm_fp4-verification-methodology)
8. [Final End-to-End Verification](#8-final-end-to-end-verification)
9. [Dead Ends and Wrong Turns](#9-dead-ends-and-wrong-turns)
10. [Two-Day Timeline](#10-two-day-timeline)
11. [Debugging Principles](#11-debugging-principles)
12. [Prevention and Future Guards](#12-prevention-and-future-guards)
13. [Postscript: Root Cause 4 — Custom NVFP4 Loader Silently Falls Back to Native Diffusers BF16 (Rebase Regression)](#13-postscript-root-cause-4--custom-nvfp4-loader-silently-falls-back-to-native-diffusers-bf16-rebase-regression)
   - 13.1 [Discovery Path](#131-discovery-path)
   - 13.2 [Sub-Cause A: Mapped vs Raw Module Name in NVFP4 Exclude List](#132-sub-cause-a-mapped-vs-raw-module-name-in-nvfp4-exclude-list)
   - 13.3 [Sub-Cause B: Supplemental Filename Filtered Out by Loader](#133-sub-cause-b-supplemental-filename-filtered-out-by-loader)
   - 13.4 [Sub-Cause C: Stale Supplemental Cache Reused Without Validation](#134-sub-cause-c-stale-supplemental-cache-reused-without-validation)
   - 13.5 [Dead Ends Along the RC4 Investigation](#135-dead-ends-along-the-rc4-investigation)
   - 13.6 [Verification After All RC4 Sub-Fixes](#136-verification-after-all-rc4-sub-fixes)
   - 13.7 [Why the Original 2-Day Investigation Did Not Catch This](#137-why-the-original-2-day-investigation-did-not-catch-this)

---

## 1. Executive Summary

`FLUX.2-dev-NVFP4` generated pure-white images (mean pixel ≈252, std ≈2) on a B200 GPU while BF16 worked correctly. The original two-day investigation found **three independent root causes**, all of which had to be fixed together before correct image quality was restored. A subsequent rebase onto current `main` exposed a **fourth root cause** — a loader-path regression where the custom SGLang NVFP4 loader was silently falling back to native diffusers BF16, making the original three fixes dead code on the rebased branch. Restoring the custom loader required three small sub-fixes (RC4-A, B, C). All four root causes are now addressed:

| # | Root Cause | Location | Impact |
|---|-----------|----------|--------|
| 1 | `swap_weight_nibbles=False` — BFL checkpoint nibble order is opposite to what mm_fp4 kernels expect | `quantization_utils.py` | GEMM outputs random noise (cos_sim≈0 vs BF16) |
| 2 | Mixed checkpoint omits `input_scale` for 64 tensors in double-block attention/MLP layers | `flux_2_nvfp4.py` | Activation scaling defaults to 1.0 → wrong quantization range for those layers |
| 3 | CUTLASS TMA blockwise scale permutation always applied, even for flashinfer/cuDNN backend which expects raw row-major scales | `modelopt_quant.py` | Scale values scrambled → GEMM #4 output 116σ outlier → cascade → all-white |
| 4 | Custom NVFP4 loader silently falls back to native diffusers BF16 on current `main` (3 sub-causes) | `quantization_utils.py`, `flux_2_nvfp4.py` | RC1–RC3 fixes never run; transformer loaded as BF16 (~120 GB) instead of NVFP4 (~22.9 GB); end-to-end output is garbage on the wrong code path |

After all four root causes are fixed:
- NVFP4 (this branch, current main): `mean=214.43, std=79.61`
- BF16 (reference):                   `mean=215.9,  std=82.58`

The images are visually and statistically indistinguishable. Loader logs confirm `Loaded transformer: Flux2Transformer2DModel (sgl-diffusion version). model size: 22.9 GB` with no fallback line and no `Found unloaded parameters in meta state dict` warning.

---

## 2. Background

### 2.1 FLUX.2 Model Architecture

FLUX.2 is a **multimodal diffusion transformer** (MM-DiT) that generates images conditioned on rich multimodal text embeddings. Unlike older CLIP-conditioned models, FLUX.2 uses a **Mistral3-based VLM** (vision-language model) as the text encoder, enabling much more precise instruction following.

#### Architecture Overview

```
Text Input  ──► Mistral3 VLM (42 GB, bf16) ──► text_tokens [B, S_txt, D]
                                                         │
Image Latent ──► VAE Encoder ──► latent_tokens [B, S_img, D]
                                                         │
                              ┌──────────────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │  8 Double Blocks   │  ← MM-DiT: text+image tokens processed
                    │  (MM-DiT style)    │    separately but with cross-attention
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │  48 Single Blocks  │  ← concatenated text+image tokens
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │   VAE Decoder      │  ← latent → pixel space
                    └────────────────────┘
```

#### Double Block (MM-DiT) Internals

Each of the 8 double blocks handles both image and text token streams:

```
image tokens ──►  img_attn.to_qkv  ──► [joint attention] ──►  img_attn.to_out
text tokens  ──►  txt_attn.to_qkv  ──►                    ──►  txt_attn.to_out
                                              │
image tokens ──►  img_mlp.0  ──► GELU  ──► img_mlp.2  ──► residual add
text tokens  ──►  txt_mlp.0  ──► GELU  ──► txt_mlp.2  ──► residual add
```

In the NVFP4 mixed checkpoint, **txt_mlp.0** and **txt_mlp.2** in all 8 double blocks are the FP4-quantized MLP layers. They become the critical path for this bug.

#### Single Block Internals

Each of the 48 single blocks concatenates text and image tokens, then applies:
- `attn.to_qkv`, `attn.to_out` (attention)
- `mlp.0`, `mlp.2` (feedforward)

These are also FP4-quantized in the NVFP4 checkpoint.

#### Sequence Lengths

图像 tokenization 分两步完成：

**第一步：VAE 空间压缩（像素空间 → latent 空间）**

VAE encoder 是带有残差块和 attention 的卷积网络，做的是**连续的空间下采样**，不是硬切 patch：

```
512×512×3 (RGB) ──► VAE Encoder ──► 64×64×32 (latent)
```

每个 latent 位置对应原图 8×8 像素的感受野，但 VAE 编码的是学习到的语义特征，而不是像素均值。

**第二步：2×2 Patchify（latent 空间 → transformer token）**

在进入 transformer 之前，对 `64×64×32` 的 latent 做空间折叠——每个 2×2 的 spatial block 拼到 channel 维度：

```
64×64×32 ──► reshape ──► 32×32×128
                    (2×2 block → 4 个位置 × 32 ch = 128 ch/token)
```

得到 `32×32 = 1024` 个 token，每个 token 覆盖原图 **16×16 像素**（VAE 8× × patchify 2× = 16×）。

**512×512 汇总：**
- `S_img = (512/8/2) × (512/8/2) = 32 × 32 = 1024` image latent tokens
- 每个 token ↔ 原图 16×16 像素，128 维 latent 向量
- `S_txt` ≤ 512（T5 encoder 上限），实际取决于 prompt 长度，通常 64–512

Token counts by resolution (after VAE 8× + 2×2 pack = 16× effective):
| Resolution | S_img |
|------------|-------|
| 512×512    | 1024  |
| 768×768    | 2304  |
| 1024×1024  | 4096  |

The transformer operates at `D=6144` (hidden dim) for FLUX.2.

---

### 2.2 NVFP4 Quantization Format

NVFP4 (from NVIDIA ModelOpt) is a **mixed-precision quantization scheme** using FP4 for weights and FP8 for per-group activation/weight scales.

#### FP4 E2M1 Number Format

FP4 uses 4 bits: `[sign(1) | exponent(2) | mantissa(1)]`

The representable positive values are:
```
0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
```
(Plus their negatives and ±0.)  
This gives a dynamic range of [−6.0, 6.0] with 16 discrete values total.

#### Nibble Packing

Two FP4 values are packed into one byte. The convention (as expected by CUTLASS and cuDNN kernels):
```
byte[i] = fp4_lo | (fp4_hi << 4)

where:
  fp4_lo (bits[3:0])  = element at even column  (col 0, 2, 4, ...)
  fp4_hi (bits[7:4])  = element at odd  column  (col 1, 3, 5, ...)
```

This is the kernel-side contract. **BFL checkpoints use the opposite convention** (lo=odd, hi=even), which is root cause 1.

#### Per-Group Scaling

For a weight matrix W of shape [N_out, K_in], the quantization is:

```
group_size = 16
w_scale: FP8 E4M3fn tensor, shape [N_out, K_in // 16]

For each group of 16 elements along K:
  w_dequant[n, k*16:(k+1)*16] = fp4_decode(w_packed[n, k*16:(k+1)*16]) × w_scale[n, k]
```

Each group of 16 consecutive input features shares one FP8 scale factor.

#### Global Scale Factors

Beyond per-group scales, NVFP4 uses two scalar multipliers per layer:

| Tensor | Type | Meaning |
|--------|------|---------|
| `input_scale` | float32 scalar | Maps activation range to FP4 input range |
| `weight_scale_2` | float32 scalar | Global weight re-scaling factor |

The combined GEMM computes:
```
y = (x_fp4 ⊗ w_fp4) × (x_scale ⊗ w_scale) × alpha

where:
  alpha = input_scale × weight_scale_2
  x_fp4 = quantize_fp4(x_bf16 × input_scale_inv)  # input_scale_inv = 1/input_scale
  w_fp4 = quantize_fp4(w_bf16 / (w_scale × weight_scale_2))
  ⊗ = FP4 GEMM with per-group scale accumulation
```

This alpha compresses the full pipeline into one fused GEMM call.

#### Full Dequantization Formula

To reconstruct BF16 output from NVFP4 tensors:
```python
# Given: w_packed [N, K//2], w_scale [N, K//16], weight_scale_2 scalar
w_lo = (w_packed & 0x0F)              # even cols, shape [N, K//2]
w_hi = (w_packed >> 4)                # odd cols,  shape [N, K//2]
w_fp4 = interleave(w_lo, w_hi)        # shape [N, K], fp4 codes

w_bf16 = fp4_decode(w_fp4)            # shape [N, K], normalized fp4 values
w_bf16 *= repeat(w_scale, 16, dim=1)  # per-group scale
w_bf16 *= weight_scale_2              # global scale
```

---

### 2.3 The Mixed Checkpoint Design

BFL released two NVFP4 checkpoints:

| File | Size | Format |
|------|------|--------|
| `flux2-dev-nvfp4.safetensors` | ~21 GB | Full FP4: all linear layers quantized |
| `flux2-dev-nvfp4-mixed.safetensors` | ~15 GB | Mixed: attention in BF16, MLP in FP4 |

The **mixed checkpoint** is intended for deployment where attention quality is prioritized and memory savings come from quantizing the FFN layers only.

The mixed checkpoint includes `weight_scale` and `weight_scale_2` for all quantized layers, but **omits `input_scale`** for 64 tensors in the 8 double blocks:

```
double_blocks.{0..7}:
  img_attn.to_out.0.input_scale    ← MISSING (img_attn is BF16, so this is expected)
  img_attn.to_qkv.input_scale      ← MISSING
  txt_mlp.0.input_scale            ← MISSING  ← BUG: txt_mlp IS FP4 quantized
  txt_mlp.2.input_scale            ← MISSING  ← BUG: txt_mlp IS FP4 quantized
  ... (weight_scale, weight_scale_2 etc)
```

When `input_scale` is missing, the loader defaults to `input_scale=1.0`. For activations that have a significantly different scale (e.g., if the true `input_scale` is 171.48), using 1.0 means the activations are never rescaled into the FP4 representable range — they saturate immediately and produce garbage.

---

### 2.4 FP4 GEMM Backends: CUTLASS vs flashinfer/cuDNN

Two distinct FP4 GEMM implementations are supported:

#### CUTLASS Backend (sgl_kernel)

```python
import sgl_kernel
sgl_kernel.cutlass_scaled_fp4_mm(out, x_fp4, w_fp4_t, x_scale, w_scale_interleaved, alpha)
```

- Available only when `sgl_kernel` is installed (typically: H100/A100 with custom build)
- Weight scales must be in **CUTLASS TMA blockwise interleaved layout** (see §2.5)
- Highest throughput on Hopper when available

#### flashinfer/cuDNN Backend

```python
import flashinfer
flashinfer.mm_fp4(x_fp4, w_fp4_t, x_scale_fp8, w_scale_fp8_t, alpha, out_dtype)
```

- Available on Blackwell (B200, sm_100) via cuDNN 9+ integration in flashinfer
- Default on B200 when `sgl_kernel` is not installed
- Weight scales must be in **raw row-major layout**: `w_scale` shape `[N, K//group_size]` FP8
- `w_scale` is passed as `.T` (transposed) → `[K//group_size, N]` as the `b_descale` argument to cuDNN

#### Backend Selection Logic

```python
# platforms/cuda.py
def _get_fp4_gemm_op():
    try:
        import sgl_kernel
        return sgl_kernel.cutlass_scaled_fp4_mm, None  # (op, flashinfer_backend=None)
    except ImportError:
        flashinfer_backend = get_modelopt_flashinfer_fp4_backend()
        return None, flashinfer_backend  # (op=None, flashinfer_backend="cudnn" or "auto")

def get_modelopt_flashinfer_fp4_backend():
    # Blackwell (sm_100) → "cudnn"
    # Others           → "auto"
    if torch.cuda.get_device_capability()[0] >= 10:
        return "cudnn"
    return "auto"
```

The key invariant: **`flashinfer_backend is None` ↔ CUTLASS path. `flashinfer_backend is not None` ↔ flashinfer/cuDNN path.**

---

### 2.5 TMA (Tensor Memory Accelerator) Weight Layout

TMA is a Hopper/Blackwell hardware feature that allows asynchronous, high-throughput bulk tensor loads from global memory to shared memory. The CUTLASS FP4 kernel uses TMA descriptors for weight scale loading, which requires the scales to be **pre-arranged in a specific blockwise interleaved layout** rather than simple row-major.

#### Why TMA Needs a Special Layout

TMA descriptors encode tensor dimensionality and strides at descriptor-creation time. For the CUTLASS mm_fp4 tile size of 128×128 (output rows × input cols), the kernel loads a 128-row block of scales in a single TMA transaction. To map cleanly onto this, scales are pre-permuted so that a contiguous tile in the TMA descriptor corresponds to the 128×4 block that each CUTLASS warp group needs.

#### The Permutation

```python
# Applied in process_weights_after_loading() before this fix
# scales shape: [B, M, K] where M=output rows, K=K//group_size
padded_scales = padded_scales.reshape(B, M // 128, 4, 32, K // 4, 4)
padded_scales = padded_scales.permute(0, 1, 4, 3, 2, 5)
# result: scales now in CUTLASS TMA blockwise interleaved order
```

Breaking down what this does:
- `M // 128` = number of 128-row tiles
- `4, 32` = within each tile, group into 4 groups of 32 rows
- `K // 4, 4` = along K, group into K//4 tiles of 4 scale values
- The permute moves the K tile dimension (`4`) before the 32-row dimension, interleaving the scale groups in the pattern CUTLASS TMA expects

**This transformation is correct for CUTLASS. It is completely wrong for cuDNN**, which reads scales in raw row-major order and performs its own internal layout transformation.

---

## 3. Symptom and Environment

### Environment

```
GPU:     NVIDIA B200 (Blackwell, sm_100, 192 GB HBM3e)
CUDA:    12.8
PyTorch: 2.6
sgl_kernel: NOT INSTALLED → flashinfer/cuDNN path always active
flashinfer: installed, cuDNN backend for sm_100
diffusers: 0.36.0.dev0
Model: black-forest-labs/FLUX.2-dev-NVFP4 (mixed checkpoint)
```

### Symptom

Every image generated by FLUX.2-dev-NVFP4 was pure white, regardless of prompt, seed, or number of inference steps:

```
NVFP4 image stats: min=253, max=255, mean=253.9, std=1.8
BF16  image stats: min=0,   max=255, mean=215.9, std=82.58
```

The VAE decoder was receiving latents saturated toward the maximum value. The denoising process was not converging — the output of each FLUX.2 transformer block was diverging rather than refining the latent.

### First Sanity Check: BF16 Works

Before investigating quantization, confirmed the pipeline itself (scheduler, VAE, tokenizer) works correctly:

```python
# Script: gen_bf16_sweep3.py
# Load FLUX.2-dev in BF16, generate at 4 and 8 steps, seed=42
# Result:
#   bf16_steps4.png: min=0, max=255, mean=215.9, std=82.58
#   bf16_steps8.png: min=0, max=255, mean=218.4, std=80.21
```

This confirmed: the non-transformer components are fine. The bug is entirely in the FP4 transformer forward pass.

---

## 4. Root Cause 1: Wrong Nibble Swap Default

### 4.1 Discovery Path

The investigation started from code review. In `quantization_utils.py`, the function `_build_nvfp4_config_from_safetensors_files()` constructs the NVFP4 quant config:

```python
# State at start of investigation (commit 5a89768):
result = quant_cls.from_config({
    "quant_algo": "NVFP4",
    "group_size": group_size,
    "ignore": exclude_modules,
    "checkpoint_uses_packed_qkv": checkpoint_uses_packed_qkv,
    "swap_weight_nibbles": False,  # ← THIS IS WRONG
})
```

The commit message for 5a89768 said: *"HF BFL checkpoints store weights in standard FP4 packing (low nibble = even col); no swap needed"* — but this claim was wrong and unverified.

The question was: **what nibble convention does the BFL checkpoint actually use?**

### 4.2 Technical Explanation: BFL Nibble Convention

#### Nibble Packing 基础

FP4 每个值 4 bits，一个 byte（8 bits）恰好打包两个。对于权重矩阵某一行，相邻两列配对：

```
col 0, col 1 → byte[0]
col 2, col 3 → byte[1]
col 4, col 5 → byte[2]  ...
```

**CUTLASS/cuDNN kernel 期望的格式：**

```
byte[i] = fp4_even_col | (fp4_odd_col << 4)
         ↑ 低 4 bits（lo nibble）  ↑ 高 4 bits（hi nibble）
```

具体示例，假设 col 0 = 1.0（FP4 编码 `0x4`），col 1 = 2.0（FP4 编码 `0x6`）：

```
byte[0] = 0x4 | (0x6 << 4) = 0x64 = 0b 0110 0100
                                           ^^^^      ← 高 nibble: col1 = 2.0
                                               ^^^^  ← 低 nibble: col0 = 1.0
```

Kernel 读取：
```python
fp4_col0 = byte & 0x0F  # → 0x4 = 1.0  ✓
fp4_col1 = byte >> 4    # → 0x6 = 2.0  ✓
```

#### BFL Checkpoint 的格式（相反）

BFL NVFP4 checkpoints 是 NVIDIA ModelOpt quantizer 生成的，打包约定与上面相反——偶数列在高 nibble，奇数列在低 nibble：

```
BFL byte[i] = fp4_odd_col | (fp4_even_col << 4)
             ↑ 低 4 bits      ↑ 高 4 bits
```

同样的 col0=1.0, col1=2.0，BFL 打包：

```
byte[0] = 0x6 | (0x4 << 4) = 0x46 = 0b 0100 0110
                                           ^^^^      ← 高 nibble: col0 = 1.0（存在高位）
                                               ^^^^  ← 低 nibble: col1 = 2.0（存在低位）
```

Kernel 拿到 `0x46`，按 CUTLASS 约定解读：
```python
fp4_col0 = 0x46 & 0x0F  # → 0x6 = 2.0  ✗（应该是 1.0）
fp4_col1 = 0x46 >> 4    # → 0x4 = 1.0  ✗（应该是 2.0）
```

每对相邻列的权重值都互换了，整个权重矩阵的列顺序全乱 → cos_sim ≈ 0。

#### Nibble Swap 修复

```python
w_swapped = ((w_packed >> 4) | (w_packed << 4)).to(torch.uint8)
```

对 `0x46` 做 swap：
```
0x46 >> 4 → 0x04
0x46 << 4 → 0x60  （uint8 自动截断高位）
OR        → 0x64  ✓ 恢复成 CUTLASS/cuDNN 期望的格式
```

把每个 byte 的高低 nibble 对调，BFL 格式就与 kernel 期望对齐。

### 4.3 The Fix

```python
# quantization_utils.py, line ~313
"swap_weight_nibbles": True,  # BFL packs: lo nibble=odd col, hi=even col; swap needed
```

The `swap_weight_nibbles` flag is consumed by ModelOpt's weight loading code, which applies the byte swap during safetensors loading, before any other processing.

### 4.4 Proof

A unit-test script was written to test GEMM output quality for a single linear layer:

**Script functionality** (`repro_nvfp4_nibble_swap.py`):
1. Load a real NVFP4 weight tensor from the BFL checkpoint (e.g., `transformer_blocks.0.mlp.0.weight`)
2. Load the corresponding BF16 weight from the companion BF16 model
3. Dequantize the FP4 weight with and without nibble swap
4. Compute cosine similarity between dequantized FP4 and the BF16 reference weight
5. Run random-input GEMM through flashinfer.mm_fp4 with both configurations
6. Compare output cosine similarity vs BF16 matmul reference

Key results from this script:

```
Without swap (False):
  w_dequant vs w_bf16:  cos_sim = -0.0002  ← complete garbage
  GEMM output cos_sim = -0.0003

With swap (True):
  w_dequant vs w_bf16:  cos_sim =  0.9909  ← correct
  GEMM output cos_sim =  0.9901
```

The weight matrix without swap has random sign relationships — the nibble transposition effectively randomizes which values land in which position across the entire weight matrix.

---

## 5. Root Cause 2: Missing Input Scales in Mixed Checkpoint

### 5.1 Discovery Path

After adding root cause 1 fix (swap=True), images were still white. The next step was to audit what tensors were actually loaded vs what the model expected.

During model loading, the following warning appeared:

```
[WARNING] Checkpoint keys not loaded (no matching model parameter):
  'transformer_blocks.0.attn.to_out.0.input_scale',
  'transformer_blocks.0.attn.to_out.0.weight_scale',
  'transformer_blocks.0.attn.to_out.0.weight_scale_2',
  'transformer_blocks.0.attn.to_qkv.input_scale',
  ...
  ... and 28 more skipped keys.
```

This warning was actually misleading — it refers to the `transformer_blocks` (single blocks), not `double_blocks`. The single-block attention layers are in BF16 in the mixed checkpoint, so these keys correctly have no matching model parameters.

The real missing tensors required **active inspection of the checkpoint contents** vs what the model expected:

**Script functionality** (inline inspection code):
1. Open both safetensors files with `safetensors.torch.load_file()`
2. Get all keys from the mixed checkpoint
3. Get all keys from the full (non-mixed) checkpoint
4. Compute: `set(full_keys) - set(mixed_keys)` — tensors in full but not in mixed
5. Filter to `input_scale` tensors only

Result: **64 missing tensors**:
```
double_blocks.0.img_attn.proj.input_scale        ← img_attn is BF16, no linear layer → OK to skip
double_blocks.0.img_attn.qkv.input_scale         ← same
double_blocks.0.txt_mlp.0.input_scale            ← THIS IS A BUG: txt_mlp.0 IS FP4!
double_blocks.0.txt_mlp.2.input_scale            ← THIS IS A BUG: txt_mlp.2 IS FP4!
double_blocks.1.img_attn.proj.input_scale        ...
double_blocks.1.img_attn.qkv.input_scale         ...
double_blocks.1.txt_mlp.0.input_scale            ← BUG
double_blocks.1.txt_mlp.2.input_scale            ← BUG
... (× 8 double blocks = 16 txt_mlp missing + 48 img_attn missing but OK)
```

Of the 64 missing tensors: 48 are for img_attn layers that are BF16 in the mixed checkpoint (benign), and **16 are for txt_mlp layers that ARE FP4** (the bug).

### 5.2 Technical Explanation

When `input_scale` is absent, the model's weight loading code sets it to the default value of `1.0`.

For a layer like `double_blocks.0.txt_mlp.0`:
- True `input_scale` from full checkpoint: approximately `171.48`
- Default loaded value: `1.0`
- Effect: activations are never divided by `input_scale` before FP4 quantization

The FP4 quantization of activations works as:
```python
x_scaled = x_bf16 * input_scale_inv   # input_scale_inv = 1 / input_scale
x_fp4 = quantize_fp4(x_scaled)        # clamp to [-6.0, 6.0] and quantize
```

With `input_scale=1.0` (wrong), `input_scale_inv=1.0`, so activations go directly into FP4 quantization without rescaling. If the typical activation magnitude is much larger than 1 (which it is — typical BF16 activations have std ≈ 0.3–1.0 in this model, but `input_scale_inv=171` means the checkpoint expects activations to be in a range 171× smaller), the quantization clips almost everything to the ±6.0 representable range, producing saturated garbage outputs.

Actually the direction is: `input_scale` is the factor that was *used during calibration* to scale the input before quantizing. With a wrong value of 1.0 instead of 171.48, the fused GEMM also multiplies outputs by `alpha = input_scale × weight_scale_2` incorrectly.

### 5.3 The Fix

The fix is in `flux_2_nvfp4.py`, function `_build_supplemental_safetensors_dir()`:

```python
def _build_supplemental_safetensors_dir(mixed_path: str) -> str:
    """
    Build a temp dir containing:
      1. A symlink/copy of the mixed checkpoint
      2. A 'supplemental.safetensors' with the missing input_scale tensors
         extracted from the companion full checkpoint
    """
    # Hash the mixed path for a stable temp dir name
    dir_hash = hashlib.md5(mixed_path.encode()).hexdigest()[:16]
    supp_dir = f"/tmp/sglang_nvfp4_supp_{dir_hash}"
    
    # Find companion full checkpoint (same directory, different filename)
    full_path = mixed_path.replace("-mixed", "")
    
    # Load both checkpoints, find missing input_scale tensors
    mixed_tensors = load_safetensors_metadata(mixed_path)
    full_tensors  = load_safetensors_metadata(full_path)
    
    missing_keys = [k for k in full_tensors if k not in mixed_tensors
                    and "input_scale" in k]
    
    # Extract missing tensors from full checkpoint, save as supplemental
    supplemental = {k: full_tensors[k] for k in missing_keys}
    save_file(supplemental, f"{supp_dir}/supplemental.safetensors")
    
    # Also symlink the mixed checkpoint into the dir so loader sees both
    os.symlink(mixed_path, f"{supp_dir}/flux2-dev-nvfp4-mixed.safetensors")
    
    return supp_dir
```

The loader is then passed `transformer_weights_path=supp_dir`, which contains both files. The model loader reads them in sequence, with the supplemental file filling in the missing tensors.

### 5.4 Proof

After this fix, model loading shows:
```
[INFO] Built NVFP4 quant config from 1 safetensors: group_size=16, 43 excluded modules
[INFO] Loading Flux2Transformer2DModel from 2 safetensors file(s)
```

The "2 safetensors file(s)" confirms the supplemental is being loaded. The 48 benign img_attn keys still generate "not loaded" warnings (expected — those layers are BF16 and have no corresponding model parameters), but the 16 txt_mlp input_scale tensors are now correctly loaded.

---

## 6. Root Cause 3: TMA Permutation Applied to Wrong Backend

### 6.1 Where We Were After Fixes 1+2

After applying both fixes above, NVFP4 still produced white images. The stats:

```
After fix 1+2: mean=252.1, std=2.4  ← still white
```

This was surprising — individually, fix 1 brought cos_sim from ≈0 to ≈0.95–0.99 for individual GEMMs. The model should have been producing something reasonable. The persistence of white images indicated a remaining systematic error that was amplifying through the 56-block denoising.

### 6.2 Adding GEMM Debug Instrumentation

The key insight: **white images are caused by numerical overflow in the denoising latents**. The VAE decoder maps latents to pixels via tanh-like saturation; if latents saturate to ±∞, the decoder produces uniform white or black. So the question shifted from "are weights correct?" to "which GEMM first produces a catastrophically large output?"

Debug instrumentation was added to `ModelOptFp4LinearMethod.apply()`:

```python
# Added to modelopt_quant.py apply() method (temporary, removed after root cause found):
_debug_call_count += 1
if _debug_call_count <= 92:  # 92 = total FP4 GEMMs in one forward pass
    x_in = x.view(-1, x.shape[-1])
    out_f = out.float()
    print(f"[NVFP4 GEMM #{_debug_call_count:3d}] "
          f"shape={list(x_in.shape)}→{list(layer.weight.shape)} "
          f"alpha={layer.alpha:.6e} "
          f"input_scale_inv={1/layer.input_scale:.6e} "
          f"x_std={x_in.std():.4f} "
          f"out_std={out_f.std():.4f} "
          f"out_max={out_f.abs().max():.4f}", 
          flush=True, file=sys.stderr)
```

This printed statistics for every FP4 GEMM call during one inference forward pass (4 denoising steps × several blocks, but only the first pass was needed to find the outlier).

### 6.3 Finding the Outlier: GEMM #4

Running NVFP4 inference with debug prints:

```
[NVFP4 GEMM #1] shape=[1, 1024, 6144]→[1024, 36864] alpha=3.044826e-07 input_scale_inv=1.714833e+02 x_std=0.3232 out_std=0.5196 out_max=4.3750
[NVFP4 GEMM #2] shape=[1, 1024, 18432]→[1024, 6144] alpha=4.632132e-06 input_scale_inv=1.750000e+01 x_std=0.1199 out_std=0.4322 out_max=4.2812
[NVFP4 GEMM #3] shape=[1, 512, 6144]→[512, 36864]  alpha=2.933666e-07 input_scale_inv=1.462857e+02 x_std=0.2677 out_std=0.4508 out_max=10.3750
[NVFP4 GEMM #4] shape=[1, 512, 18432]→[512, 6144]  alpha=7.361174e-06 input_scale_inv=1.600000e+01 x_std=0.1380 out_std=0.4277 out_max=49.7500
```

GEMM #4 stands out immediately: `out_max=49.75` while the others are all <11.

**Mapping GEMM #4 to the model graph**:
- GEMM #1: `double_blocks.0.txt_mlp.0` (txt → 6144 → 36864, first MLP expansion)
- GEMM #2: `double_blocks.0.txt_mlp.2` (36864 → 6144, MLP contraction)  
  Wait — the shapes are `[1, 1024, 18432]→[1024, 6144]`. With packed_qkv=True, this is actually the packed QKV projection for text attention in double_block.0.
- ...

The precise layer mapping required tracing the forward pass order, but the critical observation was **GEMM #4 with `out_max=49.75`**. Given that the typical output std was ~0.43, an out_max of 49.75 represents a **~116σ outlier**.

**Why 116σ matters**: In a diffusion model, the denoising network computes residuals that are added to the current noisy latent. A 116σ spike in one residual contribution causes the latent to diverge massively, which cascades through subsequent denoising steps (each step conditions on the previous latent). By step 2 of 4, the latent is dominated by NaN-adjacent values, and the VAE decoder produces white.

### 6.4 Hypothesis: Scale Layout Mismatch

At this point, fix 1 (nibble swap) was confirmed correct — cos_sim=0.99 on individual weight matrices. Fix 2 (input_scale) was confirmed loaded — the loader was reading both safetensors files. Yet GEMM #4 was still producing outliers.

The remaining possible issue in `process_weights_after_loading()` was the **CUTLASS TMA blockwise permutation** applied to weight scales:

```python
# Existing code in process_weights_after_loading():
padded_scales = padded_scales.reshape(B, M_padded // 128, 4, 32, K_padded // 4, 4)
padded_scales = padded_scales.permute(0, 1, 4, 3, 2, 5)
# ↑ This runs unconditionally, regardless of which GEMM backend will be used
```

The hypothesis: **the cuDNN backend expects raw row-major scales, but we're feeding it TMA-permuted scales**. This would not be immediately obvious from the dequantized weight test (root cause 1 analysis) because that test used a fresh dequantization of the packed weights, bypassing the scale permutation entirely.

### 6.5 Dead Ends Along the Way

**Dead end 1: Suspected remaining input_scale issue**

Initially suspected that despite the supplemental safetensors, some input_scales were still defaulting to 1.0. Added a print to the weight loading code to log all loaded input_scale values. Result: all 16 txt_mlp input_scales were correctly loading non-1.0 values. Ruled out.

**Dead end 2: Method B in mm_fp4 shape test**

To verify the cuDNN backend's expected scale format, attempted to call `flashinfer.fp4_quantize(w_bf16)` and directly compare the output to checkpoint scales:

```python
# Method B attempt:
w_q, w_scale_fresh = flashinfer.fp4_quantize(w_bf16.T)  # quantize transposed weight
out_b = flashinfer.mm_fp4(x_fp4, w_q, x_sf, w_scale_fresh, alpha, dtype)
```

This failed with `ValueError: K dimension mismatch` — `fp4_quantize` on the transposed weight produced a scale tensor with shape incompatible with the expected mm_fp4 argument shape. The error was a shape mismatch in how we constructed the inputs, not a fundamental incompatibility. This dead end consumed time but the key insight came from Method A (direct scale comparison, below).

**Dead end 3: Suspected log crash**

The verify run output was buffered — the log showed only 21 lines for ~10 minutes (text encoder loads 44.7 GB via RunAI Streamer and doesn't flush partial output). This looked like a crash or hang. Verified the process was alive via `ps aux | grep verify`, checked GPU memory was in use (`nvidia-smi` showed 7850 MiB on device 0). Confirmed: buffered output, not a crash.

**Dead end 4: TMA permutation correctness assumed for all backends**

The code comment at the permutation site said:
```python
# Blockwise interleave for CUTLASS TMA layout required by CUTLASS kernel
```

But the code had no `if CUTLASS_BACKEND:` guard. The natural assumption was "this must be needed by both kernels." This assumption was wrong and needed the explicit test to break.

### 6.6 The 4-Combination Proof Matrix

To isolate scale layout vs nibble swap, a systematic test was written:

**Script functionality** (`test_nibble_vs_noscale.py`):
1. Load a real NVFP4 layer (e.g., `double_blocks.0.txt_mlp.2`)
2. Load the corresponding BF16 weight
3. Generate a random BF16 input and compute the BF16 matmul as reference (`y_true`)
4. Run flashinfer.mm_fp4 for all 4 combinations:
   - `(swap=True,  scales=TMA_permuted)` ← current code state
   - `(swap=False, scales=TMA_permuted)`
   - `(swap=True,  scales=raw_rowmajor)` ← hypothesis: this should be correct
   - `(swap=False, scales=raw_rowmajor)`
5. For each: compute cosine similarity vs `y_true`, output std, output max
6. Also compute reference: `flashinfer.fp4_quantize(w_bf16)` and compare byte-by-byte to `w_swap`

**Results**:

```
y_true (BF16 matmul):  std=0.2562, max=2.0524

Combination                         std     max     cos_sim vs y_true
──────────────────────────────────────────────────────────────────────
swap=True  + TMA_permuted (current) 0.2657  1.3672  0.9468   ← wrong
swap=False + TMA_permuted           0.2656  1.3672 -0.0001   ← complete noise
swap=True  + raw_rowmajor           0.2560  2.1875  0.9909   ← CORRECT ✓
swap=False + raw_rowmajor           0.2560  2.0469 -0.0002   ← complete noise

Reference: fp4_quantize(w_bf16)     std=0.6408, max=3.2188  cos=0.9718
```

This matrix tells us two things with certainty:

1. **Nibble swap is correct**: In every row, `swap=True` dominates `swap=False`. Without swap, cos_sim≈0 regardless of scale layout. This confirms root cause 1.

2. **Raw row-major scales are correct for cuDNN**: `swap=True + raw_rowmajor` achieves cos_sim=0.9909, higher than any TMA variant. The TMA permutation reduces correctness from 0.9909 to 0.9468 — even though we were swapping correctly, the permuted scales still injected a 5% quality degradation, plus causing the specific 116σ outlier in GEMM #4 that triggered the white image cascade.

**Why does TMA permutation give cos=0.9468 instead of ≈0?** The permutation reorders scale values within the same tensor — it doesn't randomly zero them. For most of the weight matrix, the "nearby" scale happens to be close in value to the correct scale (since adjacent groups in a well-trained weight matrix tend to have similar magnitudes). The permutation introduces a systematic but not catastrophic error for most GEMMs. The 116σ outlier in GEMM #4 is a case where the permuted scale lands a value that is 10-100× different from the correct one for a particular group.

### 6.7 Byte-Level Verification

As a final confirmation that the checkpoint byte format was fully understood, a byte-level comparison was performed:

```python
# Take swapped checkpoint weights
w_ckpt_swapped = ((w_ckpt_packed >> 4) | (w_ckpt_packed << 4)).to(torch.uint8)

# Fresh quantization of BF16 reference weights
w_fresh_fp4, _ = flashinfer.fp4_quantize(w_bf16.cuda())
w_fresh_packed = w_fresh_fp4.cpu()

# Compare
print("w_swap bytes (first 8):", w_ckpt_swapped[:8].numpy())
print("w_fp4_fresh  (first 8):", w_fresh_packed[:8].numpy())
```

Result:
```
w_swap bytes (first 8): [157 186 121  36 133 172 204  72]
w_fp4_fresh  (first 8): [157 186 121  36 133 172 204  72]
```

**Identical**. This is strong evidence that:
- The checkpoint stores weights in BFL convention (lo=odd col)
- After nibble swap, the byte layout exactly matches what `flashinfer.fp4_quantize()` would produce from the BF16 weights
- The NVFP4 quantization calibration in ModelOpt uses the same FP4 values as flashinfer's fresh quantization (they agree on the FP4 codebook and the groupwise scaling)

This also implicitly validates that the weight_scale values in the checkpoint are correct — if the scales were wrong, the byte-level agreement would still hold but the GEMM outputs would still be wrong. The cos_sim=0.9909 from the scale test confirms scales are also correct in row-major form.

### 6.8 The Fix

```python
# modelopt_quant.py — process_weights_after_loading()
padded_scales = torch.zeros((B, M_padded, K_padded), dtype=scales.dtype)
padded_scales[:B, :M, :K] = scales

_, flashinfer_backend = _get_fp4_gemm_op()
if flashinfer_backend is None:
    # CUTLASS (sgl_kernel) path only: apply TMA blockwise interleave
    padded_scales = padded_scales.reshape(
        B, M_padded // 128, 4, 32, K_padded // 4, 4
    )
    padded_scales = padded_scales.permute(0, 1, 4, 3, 2, 5)

padded_scales = padded_scales.contiguous().cuda()
```

The guard `if flashinfer_backend is None` (= CUTLASS path) means:
- **B200 without sgl_kernel**: flashinfer_backend="cudnn" → permutation skipped → raw row-major scales → correct
- **H100/A100 with sgl_kernel**: flashinfer_backend=None → permutation applied → TMA layout → correct

The fix is backend-agnostic: each backend gets the layout it actually needs.

### 6.9 Proof

After fix 3 (with fixes 1+2 also in place), re-running the GEMM debug:

```
[NVFP4 GEMM #1] out_std=0.5196  out_max=4.38
[NVFP4 GEMM #2] out_std=0.4322  out_max=4.28
[NVFP4 GEMM #3] out_std=0.4508  out_max=10.38
[NVFP4 GEMM #4] out_std=0.4277  out_max=4.62   ← was 49.75, now normal
```

GEMM #4 dropped from 49.75 to 4.62. The 116σ outlier is gone.

---

## 7. cuDNN mm_fp4 Verification Methodology

This section documents the general approach for verifying FP4 GEMM correctness against a BF16 reference, applicable to any future kernel debugging.

### 7.1 The Correctness Test Pattern

The core pattern for testing any FP4 GEMM:

```python
import torch
import flashinfer

def test_fp4_gemm_layer(
    w_packed: torch.Tensor,   # [N, K//2] uint8, from checkpoint
    w_scale:  torch.Tensor,   # [N, K//group_size] fp8_e4m3fn
    w_scale2: float,          # weight_scale_2 scalar
    input_scale: float,       # input_scale scalar
    w_bf16:   torch.Tensor,   # [N, K] bfloat16, BF16 reference weight
    M: int = 32               # batch/sequence dimension for test
) -> dict:
    K = w_bf16.shape[1]
    N = w_bf16.shape[0]
    
    # Random BF16 input (simulate typical activation distribution)
    x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda') * 0.3
    
    # BF16 reference output
    y_true = (x @ w_bf16.T.cuda()).float()
    
    # Quantize input as flashinfer would
    x_sf_inv = input_scale  # alpha already folds this
    x_fp4, x_scale = flashinfer.fp4_quantize(x)
    alpha = input_scale * w_scale2
    
    # Test: raw row-major scales
    out_raw = flashinfer.mm_fp4(
        x_fp4, w_packed.cuda().T,
        x_scale, w_scale.cuda().T,
        alpha, torch.bfloat16
    ).float()
    
    cos_raw = torch.nn.functional.cosine_similarity(
        y_true.flatten().unsqueeze(0),
        out_raw.flatten().unsqueeze(0)
    ).item()
    
    return {
        "cos_raw": cos_raw,
        "out_max": out_raw.abs().max().item(),
        "out_std": out_raw.std().item(),
        "y_true_std": y_true.std().item(),
    }
```

**Interpreting results**:
- `cos_sim > 0.99`: excellent, GEMM is correct
- `cos_sim 0.93–0.99`: acceptable for FP4 quantization (10% quality loss is normal)
- `cos_sim 0.0 ± 0.01`: nibble order is wrong (swap issue)
- `cos_sim < 0.0`: systematic sign flip somewhere
- `out_max >> 10`: scale layout issue causing outlier groups
- `cos_sim ≈ 0.95 but out_max >> expected`: partially correct scales, some groups scrambled (TMA permutation symptom)

### 7.2 Testing Scale Layout Correctness

The 4-combination matrix test is the definitive scale layout test. Key implementation notes:

**Applying TMA permutation manually**:
```python
def apply_tma_permutation(scales: torch.Tensor) -> torch.Tensor:
    """Apply CUTLASS TMA blockwise interleave. Input: [M, K] fp8."""
    M, K = scales.shape
    M_p = round_up(M, 128)
    K_p = round_up(K, 4)
    s = torch.zeros(1, M_p, K_p, dtype=scales.dtype)
    s[0, :M, :K] = scales
    s = s.reshape(1, M_p // 128, 4, 32, K_p // 4, 4)
    s = s.permute(0, 1, 4, 3, 2, 5)
    return s.contiguous().reshape(M_p, K_p)
```

**Testing both nibble configurations**:
```python
w_swap   = ((w_packed >> 4) | (w_packed << 4)).to(torch.uint8)
w_noswap = w_packed  # as-is from checkpoint

for swap_label, w in [("swap", w_swap), ("noswap", w_noswap)]:
    for scale_label, w_sc in [("raw", w_scale_raw), ("tma", w_scale_tma)]:
        out = flashinfer.mm_fp4(x_fp4, w.T, x_sf, w_sc.T, alpha, dtype)
        cos = cosine_similarity(y_true, out)
        print(f"{swap_label}+{scale_label}: cos={cos:.4f}")
```

### 7.3 Byte-Level Nibble Correctness Test

To verify the nibble convention without needing a BF16 reference weight:

```python
# Get fresh quantization of the dequantized weight
w_dequant = dequantize_nvfp4(w_packed, w_scale, w_scale2, swap=True)
w_fresh, _ = flashinfer.fp4_quantize(w_dequant.cuda().float())
w_fresh = w_fresh.cpu()

# Compare bytes
w_ckpt_swapped = ((w_packed >> 4) | (w_packed << 4)).to(torch.uint8)
match = (w_ckpt_swapped == w_fresh.to(torch.uint8)).float().mean()
print(f"Byte match rate: {match:.4f}")  # Should be ~1.0 if nibble convention correct
```

Note: perfect match is expected only for non-saturation groups. FP4 is lossy — a small fraction of groups may round differently. Match rate > 0.97 is strong evidence of correct convention.

### 7.4 Finding the Outlier GEMM

In a 56-block model with ~92 FP4 GEMMs per forward pass, a systematic debug print is needed. The key statistics to log per GEMM:

```python
# What to print:
print(f"[GEMM #{n}] "
      f"out_max={out.abs().max():.4f} "   # primary outlier detector
      f"out_std={out.std():.4f} "          # normalization context
      f"nsigma={out.abs().max()/out.std():.1f}")  # sigma count
```

**Decision threshold**: `nsigma > 10` on any single GEMM is suspicious. `nsigma > 50` is a definite bug. In this case, GEMM #4 had `nsigma ≈ 116`, making it unambiguously the root cause of cascade divergence.

---

## 8. Final End-to-End Verification

After all three fixes, a full pipeline verification was run:

**Verification script** (`gen_nvfp4_verify.py`):
1. Call `_build_supplemental_safetensors_dir(mixed_path)` to construct the temp dir
2. Load `DiffGenerator` with `model_path="black-forest-labs/FLUX.2-dev-NVFP4"`, pointing to supplemental dir
3. Generate: prompt="Doraemon is eating dorayaki", 512×512, 4 steps, seed=42
4. Print pixel statistics

**Results**:
```
NVFP4 (all 3 fixes):   min=0, max=255, mean=215.8, std=80.84
BF16  (reference):     min=0, max=255, mean=215.9, std=82.58
```

The image shows Doraemon holding a dorayaki — semantically correct, high quality, visually comparable to the BF16 output. The mean/std difference is within expected FP4 quantization quality loss (std difference of 1.74 = 2.1%).

**Performance note**: The NVFP4 inference took 21.7 seconds for 4 steps (5.4 s/step) vs BF16 which would take ~15 s/step (rough estimate). Blackwell without sgl_kernel uses a cuDNN FP4 path that is not yet as optimized as the CUTLASS path for Hopper. Expected improvement when sgl_kernel CUTLASS support is added for B200.

---

## 9. Dead Ends and Wrong Turns

Documenting these because they illustrate how model/kernel quality bugs mislead investigation.

### DE-1: "cos_sim=0.95 means weights are probably fine"

**What happened**: Early single-GEMM test showed cos_sim=0.94–0.95 for NVFP4 vs BF16 dequantization. We briefly accepted this as "reasonable FP4 quality loss" and moved on.  
**Why it was wrong**: The 0.94 was with TMA-permuted scales and correct nibble swap. Raw scales + correct swap gives 0.99. The 5% delta is actually the TMA permutation damaging the scales. We should have asked "why is it 0.94 and not 0.99?" immediately.  
**Lesson**: For GEMM correctness verification, 0.99 is achievable with FP4. 0.94 is a hint that something is still off, not a passing grade.

### DE-2: "The supplemental safetensors fix should be enough for input_scale"

**What happened**: After fix 2 (supplemental), we assumed all input_scales were loaded. Didn't immediately verify by checking the actual loaded values.  
**Why it was wrong**: The supplemental correctly loaded the tensors, but we wasted time doubting this because the images were still white.  
**Lesson**: When multiple root causes exist, fixing one doesn't validate the others. Verify each fix independently before concluding "X is fine."

### DE-3: "The TMA permutation comment says CUTLASS, but surely cuDNN also needs layout transformation"

**What happened**: The TMA permutation code comment mentioned "CUTLASS TMA layout." We assumed cuDNN would need a similar pre-transformation. The question "does cuDNN do its own internal layout transformation?" was answered by experiment (the 4-combination test) rather than documentation first.  
**Why it was wrong**: cuDNN is a high-level library that handles layout internally. Unlike raw CUTLASS kernels, cuDNN kernels accept row-major tensors and perform internal reorganization. This is standard cuDNN API design.  
**Lesson**: When integrating two different GPU kernels for the same operation, **never assume they share the same input data format.** Always verify separately.

### DE-4: Method B shape mismatch in mm_fp4 test

**What happened**: Attempted to verify cuDNN scale format by calling `fp4_quantize(w.T)` and feeding the result directly to mm_fp4. Got `K dimension mismatch` error. Spent ~20 min debugging the shape error.  
**Why it happened**: `fp4_quantize` on `w.T` (shape [K, N]) produces packed weights [K, N//2] and scales [K, N//16]. But mm_fp4 expects `w_packed_t` (the transposed packed weight, shape [N//2, K]) and scales [N, K//16]. The shapes are not simply transposable.  
**Resolution**: Switched to Method A (test the checkpoint's own packed weights with both scale layouts), which was cleaner and more relevant anyway.  
**Lesson**: When testing a kernel, use real checkpoint data in the test rather than re-quantizing from scratch — it avoids shape confusion and tests the actual execution path.

### DE-5: Log appeared to show a crash (buffered output)

**What happened**: The verify run log showed only 21 lines for ~12 minutes. The last line was "RunAI Streamer: Overall time to stream 44.7 GiB: 6.51s" (text encoder loaded). After that, nothing. Looked like a crash.  
**Why it happened**: Python's `logging` module (used by the sglang server) buffers output when redirected to a file via `nohup`. After the first batch of log messages, all subsequent messages were held in the buffer until either the buffer was full or the process exited.  
**How we confirmed it wasn't a crash**: `ps aux | grep verify` showed the process alive with moderate CPU; `nvidia-smi` showed 7850 MiB GPU memory in use (transformer loaded). After another few minutes, the full log appeared at once (buffer flushed at exit).  
**Lesson**: For long-running inference jobs, use `python -u` (unbuffered) or `PYTHONUNBUFFERED=1` to get real-time log output: `PYTHONUNBUFFERED=1 nohup python3 verify.py > verify.log 2>&1 &`

### DE-6: Assuming the first sigma-outlier GEMM was in single_blocks

**What happened**: Initial mental model was that single_blocks (48 of them) were more likely to have issues because they have more FP4 layers and process concatenated sequences. We initially mapped GEMMs to single_blocks before realizing GEMM #4 was actually in `double_blocks.0`.  
**Why it happened**: Forward pass execution order: double_blocks first, then single_blocks. GEMMs 1-4 map to the first double block's txt_mlp (GEMMs 1-2 = double_block.0.txt_mlp.0 and txt_mlp.2, GEMMs 3-4 = double_block.0... actually depends on QKV packing).  
**Lesson**: When numbering GEMMs for debugging, also log the layer name, not just a counter. This was fixed in the instrumentation by printing the weight tensor shape, which allowed later inference of the layer.

---

## 10. Two-Day Timeline

```
Day 1 (2025-04-21)
──────────────────
Early:   Bug reported: FLUX.2-dev-NVFP4 generates white images on B200
         First check: BF16 model works, pipeline is fine

Mid:     Code review → found swap_weight_nibbles=False in commit 5a89768
         Wrote nibble swap test → confirmed cos_sim: False→0.00, True→0.99
         Added swap_weight_nibbles=True to working tree (not yet committed)

Mid:     Audit mixed checkpoint keys → found 64 missing input_scale tensors
         Identified: 16 of these are for FP4-quantized txt_mlp layers
         Wrote _build_supplemental_safetensors_dir() → committed (d3cbe94)

Late:    Ran NVFP4 with fixes 1+2 → still white (mean=252, std=2.4)
         Added GEMM debug instrumentation to modelopt_quant.py apply()

Day 2 (2025-04-22)
──────────────────
01:52:  Ran NVFP4 with debug prints → collected GEMM statistics
02:04:  Debug run complete. Identified GEMM #4: out_max=49.75 (116σ outlier)
02:10:  Hypothesis: TMA scale permutation applied to cuDNN backend

02:20:  Wrote 4-combination test (swap × scale_layout for flashinfer.mm_fp4)
02:35:  Results: swap+raw=0.9909, swap+TMA=0.9468 → TMA permutation confirmed wrong
02:40:  Byte comparison: w_swapped bytes = fp4_quantize(w_bf16) bytes → IDENTICAL

03:00:  Applied fix: conditional TMA permutation gated on flashinfer_backend is None
03:15:  Removed debug prints from modelopt_quant.py
03:27:  Launched full end-to-end verify run (gen_nvfp4_verify.py)

03:38:  Text encoder loaded (11 min, 44.7 GB → RunAI Streamer)
03:39:  Image generated: mean=215.8, std=80.84 ✓

03:45:  Committed fix 1 (swap_weight_nibbles=True)
03:46:  Committed fix 3 (conditional TMA permutation)
        Note: fix 2 was committed day before (d3cbe94)
```

---

## 11. Debugging Principles

These principles are generalizable to any kernel/quantization/model quality debugging.

### P1: "White image" = overflow in latent space, not a pixel problem

When a diffusion model generates white or black images, the root cause is almost never in the VAE decoder. It's always that the denoising network's output latents have diverged. The question is: **which component first produces out-of-range values?**

Corollary: attach statistics at the output of each transformer block (or each GEMM), not at the final pixel output. The divergence point is usually within the first 1-2 blocks.

### P2: Sigma count > 50 on any tensor is a smoking gun

Normal neural network intermediate tensors follow approximately Gaussian distributions. A value exceeding 50σ is not noise — it's a systematic error. Log `abs(output).max() / output.std()` for every GEMM when debugging quality issues.

### P3: Test each component in isolation before the full pipeline

The full NVFP4 pipeline has 3 moving parts: nibble swap, supplemental scales, TMA layout. Testing all three together makes it impossible to attribute blame. The correct approach:
1. Test weight dequantization accuracy (cos_sim of w_fp4 vs w_bf16)
2. Test single-GEMM output accuracy (cos_sim of y_fp4 vs y_bf16 with fixed random input)
3. Test full pipeline only after both pass

### P4: The scale format and the weight format are independent concerns

FP4 GEMM has two separate format questions:
- **Weight bytes**: nibble packing convention (lo=even or lo=odd)
- **Scale bytes**: layout (row-major or TMA-interleaved)

These are orthogonal. Fixing one doesn't affect the other. The 4-combination matrix test (swap × scale_layout) is the canonical way to decouple them.

### P5: Different backends for the same operation ≠ same data format

CUTLASS, cuDNN, and FlashInfer are three different implementations of FP4 GEMM. Each has its own:
- Weight scale layout expectations
- Packed weight byte ordering expectations
- Alpha/scale composition conventions

Never assume "this worked in CUTLASS, so it'll work in cuDNN with the same inputs." Always verify with a unit test first, using real checkpoint data.

### P6: Backend-conditional code paths must be explicit and tested

The pattern:
```python
_, flashinfer_backend = _get_fp4_gemm_op()
if flashinfer_backend is None:  # CUTLASS path
    # CUTLASS-specific layout transformation
    ...
```

Any code that preprocesses weights differently for different backends should have a visible guard. "Works on H100" + "doesn't work on B200" is the classic symptom of a missing backend guard.

### P7: Commit messages must be accurate about semantics, not just syntax

Commit 5a89768: *"set swap_weight_nibbles=False for HF BFL checkpoints"* — asserted a semantic claim ("HF BFL checkpoints use standard packing") that was factually wrong. The commit introduced a bug because the message's assertion was never verified.

A correct commit message would reference the evidence: *"tested on layer X, cos_sim=0.99 without swap."* The absence of such evidence in the message is a red flag.

### P8: When the log stops, prove it's buffering before assuming a crash

Command: `PYTHONUNBUFFERED=1 nohup python3 script.py > out.log 2>&1 &`

Always use unbuffered output for long-running diagnostic scripts. Without it, debugging the debugger wastes time.

---

## 12. Prevention and Future Guards

### Guard 1: Single-GEMM Quality Regression Test

Add a fast (~30s) test that verifies NVFP4 GEMM quality for one representative layer:

```
python -m pytest .../test/unit/test_nvfp4_gemm.py -v
```

Test should:
- Load a specific NVFP4 layer from checkpoint (or a synthetic one with known properties)
- Run flashinfer.mm_fp4 with current code path
- Assert cos_sim vs BF16 reference > 0.98
- Run once per backend (CUTLASS if available, flashinfer always)

### Guard 2: Backend-Specific Weight Processing Tests

Any function in `process_weights_after_loading()` that transforms weight data should have separate test cases for each backend path. The TMA permutation bug existed for months because there was no test that ran both paths and compared outputs.

### Guard 3: Checkpoint Completeness Validation at Load Time

The supplemental safetensors workaround is correct, but a better long-term fix is a validation step at load time:

```python
def validate_nvfp4_checkpoint(model, checkpoint_tensors):
    for name, module in model.named_modules():
        if isinstance(module, ModelOptFp4Linear):
            assert f"{name}.input_scale" in checkpoint_tensors, \
                f"Missing input_scale for FP4 layer {name}"
```

This surfaces the missing-tensor bug as an explicit error rather than silently defaulting to 1.0.

### Guard 4: Log GEMM Statistics in Debug Mode

Keep (but gate) the GEMM debug instrumentation behind a flag:

```python
if os.environ.get("SGLANG_NVFP4_DEBUG_GEMMS"):
    log_gemm_stats(n, x, out, layer)
```

Setting `SGLANG_NVFP4_DEBUG_GEMMS=1` should be the first step in any future NVFP4 quality investigation.

### Guard 5: `swap_weight_nibbles` Must Be Documented with Evidence

The value of `swap_weight_nibbles` must be explained with reference to a specific verification test. The current comment reads:

```python
"swap_weight_nibbles": True,
# BFL NVFP4 checkpoints pack: low nibble = odd col, high nibble = even col
# Verified: byte comparison w_swapped == fp4_quantize(w_bf16) for
# double_blocks.0.txt_mlp.2 in flux2-dev-nvfp4-mixed.safetensors
```

If a future checkpoint uses a different convention, this comment should be updated and the verification test should be re-run.

### Guard 6: Verify the Custom NVFP4 Code Path Actually Ran

Image-pixel verification (`mean≈215, std≈80`) confirms GEMM-level correctness but does NOT distinguish "custom NVFP4 path produced the right image" from "silent fallback to native diffusers BF16 produced the right image." Loader-path drift on `main` (RC4) caused exactly this confusion.

A cheap CI check after any NVFP4 verification run:

```bash
LOG=/tmp/nvfp4_verify.log

# 1. Custom NVFP4 path actually ran
grep -q "Loaded transformer: Flux2Transformer2DModel (sgl-diffusion version)" "$LOG" \
  || { echo "FAIL: custom transformer loader did not run"; exit 1; }

# 2. Loaded model size is in the NVFP4 range, not BF16
grep "Loaded transformer.*model size" "$LOG" | awk '{print $NF, $(NF-1)}' \
  | awk '$2=="GB" && ($1 > 30) { exit 1 }' \
  || { echo "FAIL: model size > 30 GB suggests BF16 fallback"; exit 1; }

# 3. No silent fallback fired
grep -q "falling back to native version" "$LOG" \
  && { echo "FAIL: ComponentLoader fell back to native diffusers"; exit 1; }

# 4. No tensors silently dropped during load
grep -q "Found unloaded parameters in meta state dict" "$LOG" \
  && { echo "FAIL: meta state dict had unloaded parameters"; exit 1; }

# 5. Both supplemental files reached the loader (when supplemental dir is in use)
grep -q "Loading Flux2Transformer2DModel from 2 safetensors file(s)" "$LOG" \
  || echo "WARN: only 1 safetensors file loaded; supplemental may have been filtered"
```

These five greps catch every RC4 sub-cause individually and require no GPU work — they're pure log inspection. Wire them into the same CI step that runs the end-to-end image generation, and any future loader-path regression on `main` becomes a hard failure instead of a silent BF16 substitution.

---

## 13. Postscript: Root Cause 4 — Custom NVFP4 Loader Silently Falls Back to Native Diffusers BF16 (Rebase Regression)

### 13.1 Discovery Path

Three days after the original investigation closed, the fix branch was rebased onto current `main` so it could be opened as a PR. The rebase itself was clean — no conflicts on the three NVFP4 fix files. The first sanity-check generation, however, produced obviously wrong output:

```
NVFP4 (rebased, all 3 fixes): mean=219.19, std=12.12
BF16 reference:               mean=215.9,  std=82.58
```

Pixel mean was close to BF16 but **standard deviation collapsed from ~80 to ~12** — the rendered image was a near-uniform cream/beige smear with no structure. Critically, this was *not* the original "white image" symptom — std=12 is not std=2, and the image was *not* saturated to 255. The failure mode was different.

The transformer-loader log explained why:

```
[ERROR] AssertionError: Failed to shard/load parameter proj_out.weight:
        full_tensor.shape=(128, 6144), meta_sharded_param.shape=(128, 3072),
        temp_param.shape=(128, 3072), param_cls=ModelWeightParameter
[ERROR] Error while loading customized transformer, falling back to native version
[INFO]  Loaded transformer: Flux2Transformer2DModel (native version).
        model size: 120.04 GB, consumed GPU mem: 120.04 GB
```

Three observations from this log immediately reframed the problem:

1. **The custom SGLang NVFP4 loader was failing**, not the kernel or the data. Failure was at *weight-load time* (shape assertion), well before any GEMM ran.
2. **`ComponentLoader` was silently catching the exception and falling back to native diffusers**. The fallback path loads the model as plain BF16 from the diffusers `Flux2Transformer2DModel` class — which has no NVFP4-aware weight loader, no `swap_weight_nibbles`, no input_scale handling, no TMA-permute decision. **Every single fix from RC1, RC2, and RC3 was dead code on this path.**
3. **`model size: 120.04 GB`** is the giveaway. The custom NVFP4 path produces a transformer that occupies ~22.9 GB on GPU. The native BF16 path, with FLUX.2-dev's full bf16 weights expanded, lands around 120 GB. The 5× ratio is exactly what we'd expect from `bfloat16 / nvfp4-packed`.

The remaining work was to figure out why the custom loader was failing on `proj_out.weight` specifically — a shape mismatch of `(128, 6144)` (checkpoint) vs `(128, 3072)` (meta). The factor-of-two ratio was suspiciously aligned with the FP4 packing factor: NVFP4 stores two FP4 values per byte, so a logical `[N, K]` weight is packed as `[N, K/2]` on disk. Either the checkpoint shipped this layer **unpacked** (so it's actually BF16) and the meta wanted it **packed** (NVFP4), or vice versa. That mismatch is the core RC4-A.

Three sub-causes had to be fixed for the custom path to succeed end-to-end:

- **RC4-A:** `proj_out` was incorrectly tagged as NVFP4 when it should stay BF16. The `exclude_modules` list was being built with the wrong key namespace.
- **RC4-B:** Even after RC4-A, an `Found unloaded parameters in meta state dict` warning appeared for the same 16 `txt_mlp.*.input_scale` tensors that RC2 was supposed to backfill. The supplemental file was being generated correctly but silently dropped by the loader's `*-mixed.safetensors` filename filter.
- **RC4-C:** A stale `/tmp` cache from earlier debugging runs was persisting across invocations even when the supplemental's contents no longer matched the current expected-extras set, masking the test signal during iteration.

---

### 13.2 Sub-Cause A: Mapped vs Raw Module Name in NVFP4 Exclude List

#### 13.2.1 Symptom

Custom NVFP4 loader fails at meta-load time with:

```
AssertionError: Failed to shard/load parameter proj_out.weight:
                full_tensor.shape=(128, 6144), meta_sharded_param.shape=(128, 3072)
```

The `(128, 6144)` shape is the BF16 storage of `proj_out`; the `(128, 3072)` is what the meta layout expects when `proj_out` is treated as NVFP4 (FP4 is packed 2-per-byte → K halved).

#### 13.2.2 Investigation

`_build_nvfp4_config_from_safetensors_files()` in `quantization_utils.py` builds the NVFP4 quant config from safetensors metadata when the checkpoint doesn't ship a `quantization_config` in its diffusers config. The function walks BFL-side metadata to compute which modules stay BF16 (the `exclude_modules` list), e.g. `final_layer.linear` (BFL's name for the last projection layer).

The pre-existing code on `main` looked like:

```python
for module_bfl in exclude_bfl_modules:
    raw_weight_name = f"{module_bfl}.weight"
    if mapping_fn is not None:
        mapped, _, _ = mapping_fn(raw_weight_name)
        if mapped != raw_weight_name:
            exclude_modules.append(module_bfl)   # ← BUG: appends BFL name, not mapped name
            continue
    # ... fallback paths
```

The `exclude_modules` list is later passed into `ModelOptFp4Config` and used by the runtime quant method to decide which submodules of the SGLang `Flux2Transformer2DModel` should remain BF16. **But the runtime quant config is keyed by SGLang-side (mapped) module names, not BFL-side raw names.**

For FLUX.2:

| BFL raw name | Mapped (SGLang) name |
|--------------|----------------------|
| `final_layer.linear` | `proj_out` |
| `double_blocks.X.img_mlp.0` | `transformer_blocks.X.ff.linear_in` |
| `double_blocks.X.txt_mlp.2` | `transformer_blocks.X.ff_context.linear_out` |

By appending `final_layer.linear` instead of `proj_out`, the BF16 exclusion never took effect. `proj_out` continued to be tagged as NVFP4. The loader then tried to interpret the BF16 checkpoint bytes for `proj_out.weight` as NVFP4-packed bytes and tripped the meta-shape assertion.

#### 13.2.3 Why the Factor of Two

NVFP4 weight packing:
- Logical shape: `[out_features, in_features]` (the BF16 weight matrix shape)
- Stored on disk: `[out_features, in_features // 2]` (2 FP4 values per byte)

`proj_out` has logical shape `[128, 6144]`. Stored as BF16 → `[128, 6144]`, two bytes per element. Stored as NVFP4 → `[128, 3072]`, one byte per pair.

When the custom loader tried to materialize `proj_out` as NVFP4, the meta tensor was allocated at `[128, 3072]` (the packed shape). The full BF16 tensor coming from the checkpoint was `[128, 6144]`. The shape assertion fired immediately.

#### 13.2.4 The Fix

```python
for module_bfl in exclude_bfl_modules:
    raw_weight_name = f"{module_bfl}.weight"
    if mapping_fn is not None:
        mapped, _, _ = mapping_fn(raw_weight_name)
        if mapped != raw_weight_name:
            # exclude_modules holds module names (no ".weight" suffix) under
            # the mapped (SGLang-side) layout — the load-time NVFP4 quant
            # config is keyed by runtime module names.
            mapped_module = (
                mapped[: -len(".weight")] if mapped.endswith(".weight") else mapped
            )
            exclude_modules.append(mapped_module)   # e.g. "proj_out"
            continue
```

The `.weight` suffix is stripped because `exclude_modules` is a list of *module* names (no parameter suffix). `mapping_fn` returns the full mapped key including `.weight`, so we strip it before appending.

#### 13.2.5 Proof

A focused unit test (`test_quantization_utils.py:test_nvfp4_excludes_flux2_bf16_fallback_layers_after_name_mapping`) constructs a synthetic FLUX.2 NVFP4 checkpoint with `final_layer.linear.weight` in BF16, runs `build_nvfp4_config_from_safetensors_list()` with the FluxArchConfig name-mapping dict, and asserts that:

```python
self.assertIn("proj_out", quant_config.exclude_modules)
self.assertNotIn("final_layer.linear", quant_config.exclude_modules)
```

After the fix, the test passes. End-to-end, the `proj_out.weight` shape assertion at meta-load time disappears, and the custom NVFP4 loader proceeds past it.

---

### 13.3 Sub-Cause B: Supplemental Filename Filtered Out by Loader

#### 13.3.1 Symptom

After RC4-A fixed the meta-load assertion, end-to-end generation reached the denoising loop but produced the same `mean≈219, std≈12` cream-smear as before — and the loader log emitted a new (or re-emitted) warning that had not been there during the original two-day investigation:

```
[WARNING] Found unloaded parameters in meta state dict: {
  'transformer_blocks.0.ff_context.linear_in.input_scale',
  'transformer_blocks.0.ff_context.linear_out.input_scale',
  'transformer_blocks.1.ff_context.linear_in.input_scale',
  ...
  ... 16 tensors total, all transformer_blocks.{0..7}.ff_context.linear_{in,out}.input_scale
}
```

These are *exactly* the 16 `txt_mlp.*.input_scale` tensors that RC2's `_build_supplemental_safetensors_dir()` was supposed to backfill — under the BFL→SGLang name mapping (`txt_mlp.0` → `ff_context.linear_in`, `txt_mlp.2` → `ff_context.linear_out`).

#### 13.3.2 Investigation

The supplemental file was being generated. Inspection of the temp dir confirmed it contained the expected 64 keys:

```bash
$ ls -la /tmp/sglang_nvfp4_supp_<hash>/
flux2-dev-nvfp4-mixed.safetensors -> /home/.../models--black-forest-labs--FLUX.2-dev-NVFP4/.../flux2-dev-nvfp4-mixed.safetensors
supplemental.safetensors                  ← 64 keys, all expected

$ python -c "from safetensors import safe_open; \
             print(len(list(safe_open('/tmp/sglang_nvfp4_supp_<hash>/supplemental.safetensors', \
             framework='pt').keys())))"
64
```

The keys included exactly the 16 `double_blocks.*.txt_mlp.*.input_scale` tensors that should map to the unloaded `transformer_blocks.*.ff_context.linear_*.input_scale` parameters.

So the supplemental file existed, with the right contents, in the right directory. **Why wasn't it being loaded?**

Closer inspection of the loader log revealed a line that had not been noticed during the original investigation:

```
[INFO] Using 1 mixed transformer safetensors file(s) and ignoring 1 sibling
       non-mixed file(s):
       ['/tmp/sglang_nvfp4_supp_<hash>/flux2-dev-nvfp4-mixed.safetensors']
```

The loader was picking up only the *symlinked* `flux2-dev-nvfp4-mixed.safetensors` and **silently dropping `supplemental.safetensors` as a "non-mixed sibling"**.

#### 13.3.3 The Loader Filter

`runtime/loader/transformer_load_utils.py` defines:

```python
_MIXED_SAFETENSORS_RE = re.compile(r".*-mixed(?:-\d+-of-\d+)?\.safetensors$")

def _prefer_mixed_safetensors_files(safetensors_list: list[str]) -> list[str]:
    """Prefer mixed-precision transformer exports over sibling full exports.

    Some raw ModelOpt NVFP4 repos ship both `foo-mixed.safetensors` and
    `foo.safetensors`. They are alternative full transformer exports, not
    shards, so loading both trips duplicate tensor-name validation.
    """
    mixed_files = [
        path for path in safetensors_list
        if _MIXED_SAFETENSORS_RE.match(os.path.basename(path))
    ]
    if not mixed_files or len(mixed_files) == len(safetensors_list):
        return safetensors_list
    logger.info(
        "Using %d mixed transformer safetensors file(s) and ignoring %d sibling "
        "non-mixed file(s): %s",
        len(mixed_files), len(safetensors_list) - len(mixed_files), mixed_files
    )
    return mixed_files
```

The filter is intended to prevent loading both `foo-mixed.safetensors` and `foo.safetensors` when a raw ModelOpt export ships both. Its regex accepts only filenames ending in `-mixed.safetensors` (or sharded `-mixed-<i>-of-<n>.safetensors`).

The original RC2 fix named the supplemental file `supplemental.safetensors`. That name **does not match the regex**. So in the supp dir's two-file content (`flux2-dev-nvfp4-mixed.safetensors` symlink + `supplemental.safetensors`), the filter saw one match and one non-match — and silently dropped the non-match.

This is a textbook case of an *invisible* interaction between two unrelated subsystems:
- RC2's fix produced a deterministic temp dir with two files, designed to be loaded together.
- The loader's filter (added independently for a different reason — duplicate-key avoidance) silently filters out anything not named `*-mixed.safetensors`.

The two subsystems didn't share a contract, and there was no error or even warning when one stripped the other's output.

#### 13.3.4 The Fix

Rename the supplemental file so it passes the filter:

```python
mixed_link = os.path.join(supp_dir, mixed_name)
# Name must match the loader's *-mixed.safetensors filter
# (_prefer_mixed_safetensors_files in transformer_load_utils.py); otherwise
# the supplemental file is silently dropped as a "non-mixed sibling".
supp_path = os.path.join(supp_dir, "supplemental-mixed.safetensors")
```

The `-mixed.safetensors` suffix is the minimum change required for the regex to match. No filter logic was changed — the contract change is purely on the filename side.

#### 13.3.5 Proof

After RC4-A + RC4-B, with a fresh cache:

```
[INFO] Using mixed-precision NVFP4 weights: ...flux2-dev-nvfp4-mixed.safetensors
[INFO] Supplemented mixed NVFP4 checkpoint with 64 missing tensor(s) from
       flux2-dev-nvfp4.safetensors → /tmp/sglang_nvfp4_supp_<hash>
[INFO] Loading Flux2Transformer2DModel from 2 safetensors file(s)
        ↑ both files now loaded together
```

No more `Found unloaded parameters in meta state dict` warning. The 16 `transformer_blocks.*.ff_context.linear_*.input_scale` tensors flow through the BFL→SGLang name mapping at load time and land in the right runtime parameters.

#### 13.3.6 Why the Filter Didn't Match

A quick check before settling on the rename: the regex is `.*-mixed(?:-\d+-of-\d+)?\.safetensors$`. Walk through both filenames:

| Filename | Matches? | Why |
|----------|----------|-----|
| `flux2-dev-nvfp4-mixed.safetensors` | ✓ | suffix `-mixed.safetensors` matches `.*-mixed\.safetensors$` |
| `supplemental.safetensors` | ✗ | no `-mixed` before `.safetensors` |
| `supplemental-mixed.safetensors` | ✓ | suffix `-mixed.safetensors` matches |

Renaming is sufficient and minimally invasive.

---

### 13.4 Sub-Cause C: Stale Supplemental Cache Reused Without Validation

#### 13.4.1 Symptom

During iterative debugging of RC4-A and RC4-B, the supplemental `/tmp/sglang_nvfp4_supp_<hash>/` directory persisted across runs. The original RC2 fix used a deterministic hash of the mixed-file path as the temp-dir name and reused the dir if `supplemental.safetensors` already existed:

```python
# Original RC2 implementation:
mixed_link = os.path.join(supp_dir, mixed_name)
supp_path = os.path.join(supp_dir, "supplemental.safetensors")

if os.path.isfile(supp_path) and os.path.exists(mixed_link):
    return supp_dir   # ← reused without inspecting contents
```

The intent was to avoid re-extracting 64 tensors on every server start. The implementation was fine for the original RC2 fix. It became a problem in two scenarios:

1. **A supplemental file generated by an older builder** (e.g. before the `txt_mlp.*.input_scale` set was identified, or before the RC4-A fix changed which keys are needed) survives unchanged across runs.
2. **The supplemental file is renamed** (RC4-B). The cache still has the old `supplemental.safetensors`; the new code now writes `supplemental-mixed.safetensors`. Both files coexist; the cache check sees the new file is missing, regenerates it, but the old one is still there — and the loader still applies the `*-mixed.safetensors` filter, so the stale `supplemental.safetensors` is dropped (no harm), and the new `supplemental-mixed.safetensors` is loaded (correct). This case happens to recover, but it's accidental.

The first scenario is the dangerous one: a stale supplemental whose key set no longer matches the current expectation produces a quiet correctness failure with no warning.

#### 13.4.2 The Fix

Compute the expected extra-key set up-front, validate the cached supplemental against it, and rebuild on mismatch with a loud warning:

```python
with safe_open(mixed_file, framework="pt", device="cpu") as f:
    mixed_keys = set(f.keys())
with safe_open(non_mixed_path, framework="pt", device="cpu") as f:
    non_mixed_keys = set(f.keys())

expected_extra_keys = non_mixed_keys - mixed_keys
if not expected_extra_keys:
    return None

# ... mixed_link, supp_path setup ...

if os.path.isfile(supp_path) and os.path.exists(mixed_link):
    with safe_open(supp_path, framework="pt", device="cpu") as f:
        supplemental_keys = set(f.keys())
    if supplemental_keys == expected_extra_keys:
        return supp_dir
    logger.warning(
        "Rebuilding stale FLUX.2 NVFP4 supplemental cache at %s: expected %d "
        "tensor(s), found %d",
        supp_dir, len(expected_extra_keys), len(supplemental_keys),
    )

# ... fall through to fresh extraction ...
```

The validation is a set-equality check, not a subset check — extras are as bad as missing keys (they indicate a stale or different builder).

#### 13.4.3 Why This Matters Beyond the Iteration Loop

Without RC4-C, the failure mode for a stale cache is:
- `supplemental.safetensors` exists with old contents
- `_build_supplemental_safetensors_dir()` returns the supp dir without rebuilding
- The loader (with RC4-B fix) loads the supplemental — but its keys are stale
- Some expected tensors are missing → `Found unloaded parameters in meta state dict` warning
- The model runs but with wrong activation scaling for the missing layers

This is exactly the original RC2 failure mode, re-introduced by a stale cache. The original RC2 detection (key-set diff with the full checkpoint) was correct but not idempotent against builder evolution.

#### 13.4.4 Proof

A focused unit test (`test_flux2_nvfp4.py:test_rebuilds_stale_supplemental_cache`) constructs a mixed checkpoint with one missing tensor, builds the supplemental once, **manually overwrites the supplemental with empty contents** to simulate a stale builder, and asserts that a second call to `_build_supplemental_safetensors_dir()` rebuilds the file:

```python
# 1. Build supplemental for the first time
supp_dir = Path(_build_supplemental_safetensors_dir(str(mixed_file)))
self.assertTrue((supp_dir / "supplemental-mixed.safetensors").is_file())

# 2. Stale-cache simulation: overwrite with empty contents
save_file({}, str(supp_dir / "supplemental-mixed.safetensors"))

# 3. Second build should detect the mismatch and rebuild
rebuilt_dir = Path(_build_supplemental_safetensors_dir(str(mixed_file)))
self.assertEqual(supp_dir, rebuilt_dir)

# 4. Verify the rebuilt file has the expected key
with safe_open(rebuilt_dir / "supplemental-mixed.safetensors",
               framework="pt", device="cpu") as f:
    self.assertIn("double_blocks.0.txt_mlp.0.input_scale", set(f.keys()))
```

The test passes after RC4-C is applied; before the fix, the second `_build_supplemental_safetensors_dir()` call would short-circuit on `os.path.isfile(supp_path)` and return the empty supplemental, leaving `double_blocks.0.txt_mlp.0.input_scale` unloaded.

---

### 13.5 Dead Ends Along the RC4 Investigation

#### DE-RC4-1: "rotary_emb=None must be a model wrapper bug"

**What happened**: The first attempt to run the rebased branch hit `AttributeError: 'NoneType' object has no attribute 'forward'` at `flux.py:520` when calling `rotary_emb.forward(img_ids)`. `denoising.py:624` calls `getattr(self.transformer, "rotary_emb", None)` and silently returns `None` when the attribute is missing. Quick code archaeology suggested the `Flux2Transformer2DModel` DiT has `self.rotary_emb = Flux2PosEmbed(...)` in its `__init__`, so the attribute *should* exist. We hypothesized that `self.transformer` was being wrapped by some new layer (FSDP, device-manager, layerwise-offload mixin) that didn't forward `rotary_emb` through.

**Why it was wrong**: There's no wrapper. `self.transformer` is the raw `Flux2Transformer2DModel` instance — except it's actually the *native diffusers* `Flux2Transformer2DModel`, loaded via the silent fallback path. The native diffusers class has `pos_embed`, not `rotary_emb`. `getattr(...)` returned None because the attribute genuinely didn't exist on the *fallen-back* class.

**Lesson**: When a `getattr(obj, attr, default)` returns the default, ask "is this the object I think it is?" before "is this attribute being stripped by a wrapper?" Type identity matters — `type(self.transformer).__name__` is the first thing to print.

#### DE-RC4-2: "Codex's `rotary_emb = pos_embed` alias is the right fix"

**What happened**: An automated debugging session proposed patching the loader to alias `rotary_emb = pos_embed` whenever the native diffusers fallback fired, so `denoising.py`'s `getattr(...)` would find a usable RoPE module. This was technically a valid workaround — the pipeline would no longer crash.

**Why it was wrong**: It papered over the *real* bug — that the fallback was firing at all. The alias would let the pipeline run, but it would run as **native diffusers BF16** with no NVFP4 quantization. RC1, RC2, and RC3 would all be dead code. The image-quality test would pass because BF16 always works, but the PR's actual purpose (NVFP4 correctness on Blackwell) would never be exercised. We rejected the alias and went looking for the proximate cause of the fallback (the `proj_out` shape assertion) instead.

**Lesson**: A patch that silences a symptom in the wrong code path is worse than no patch at all — it removes the test signal that could have caught the underlying regression. Always ask "does my fix put us on the code path I'm trying to test?" before merging.

#### DE-RC4-3: "After RC4-A, the supplemental should just work"

**What happened**: After RC4-A removed the `proj_out` shape assertion, we ran end-to-end and saw a successful generation — but with the wrong pixel stats (`mean≈219, std≈12` again). It was tempting to declare RC4-A insufficient and start hunting for a fourth correctness bug elsewhere.

**Why it was wrong**: The actual issue (RC4-B, supplemental filename) had been silently present all along — its symptom (`Found unloaded parameters` warning) was being printed but had been overlooked because the `proj_out` assertion was the louder signal. Once RC4-A cleared the assertion, the pre-existing warning became the *current* loudest signal, but only if you went looking for it. Re-reading the full loader log line by line is what surfaced the `Using 1 mixed safetensors file(s) and ignoring 1 sibling non-mixed file(s)` line.

**Lesson**: When a fix unblocks progress but the symptom doesn't fully resolve, don't assume "additional unrelated bug." Re-grep the full log for warnings that may have been masked by louder errors. Each fix can expose the next, just like in the original two-day investigation.

#### DE-RC4-4: "The supplemental cache is fine to leave in /tmp"

**What happened**: During the RC4-A and RC4-B iteration, we noticed a stale `/tmp/sglang_nvfp4_supp_<hash>/` directory from earlier runs. We initially left it in place, reasoning that the deterministic-hash naming would make it self-correcting.

**Why it was wrong**: Self-correction only works if the cache check validates contents, not just filenames. The original RC2 implementation only checked for the file's *existence*. After RC4-B renamed `supplemental.safetensors` to `supplemental-mixed.safetensors`, both files coexisted in the cache; the new code path created the new file (good) but didn't notice that the old file was still around (irrelevant in this case but masked the symptom in others). More importantly, a supplemental from a *different builder* could survive indefinitely with no warning. RC4-C made the cache check a content-equality check; the dead end is what motivated that fix.

**Lesson**: Caches without content validation are silent footguns. Either invalidate aggressively (re-extract every run) or validate by content (set-equality of expected keys).

---

### 13.6 Verification After All RC4 Sub-Fixes

After RC4-A + RC4-B + RC4-C, on a freshly-cleaned `/tmp` cache, end-to-end FLUX.2-dev-NVFP4 generation on B200 produces:

**Loader log (key lines)**:
```
[INFO] Using mixed-precision NVFP4 weights: .../flux2-dev-nvfp4-mixed.safetensors
[INFO] Supplemented mixed NVFP4 checkpoint with 64 missing tensor(s) from
       flux2-dev-nvfp4.safetensors → /tmp/sglang_nvfp4_supp_<hash>
[INFO] Loading Flux2Transformer2DModel from 2 safetensors file(s)
[INFO] Loaded transformer: Flux2Transformer2DModel (sgl-diffusion version).
       model size: 22.9 GB
```

**No** `falling back to native version` line.
**No** `Found unloaded parameters in meta state dict` warning.

**Image stats**:
```
NVFP4 (this branch): min=0  max=255  mean=214.43  std=79.61
BF16 reference:      min=0  max=255  mean=215.9   std=82.58
                                     Δ=0.7%       Δ=3.6%
```

**Wall time**: 24.56s for 12 denoising steps (768×768) on a single B200, with the custom NVFP4 path (22.9 GB on GPU) versus what the broken native fallback would have been (120 GB on GPU + slower BF16 path).

**Diagnostic-summary table for RC4** (matching the format used for RC1–RC3):

| Sub-cause | Diagnostic tool | Signature |
|---|---|---|
| RC4-A: Mapped vs raw module name | Custom-loader log | `AssertionError: Failed to shard/load parameter proj_out.weight, full=(128, 6144), meta=(128, 3072)` |
| RC4-B: Supplemental filename | Loader-info line | `Using 1 mixed transformer safetensors file(s) and ignoring 1 sibling non-mixed file(s)` |
| RC4-C: Stale cache | Persistent `/tmp` dir + `Found unloaded parameters` returns after iteration | Old supplemental key set ≠ current expected set |

---

### 13.7 Why the Original 2-Day Investigation Did Not Catch This

A natural question after RC4 is: how did the original investigation produce a "PR-ready" branch that was dead code on current `main`? Three reasons:

1. **The branch base was old.** The original investigation ran against `a4cf2ea` — the `sgl-project/sglang:main` tip *at the time*. On that base, the custom NVFP4 loader worked, the `*-mixed.safetensors` filter behaved as intended, and the original three fixes ran exactly as designed. Run B from the rebase verification (worktree at `e43fa66`, the original fix tip) reproduced the correct image (`mean=215.8, std=80.84`) — proving the original fixes were correct *for that base*.

2. **The drift on `main` was indirect.** Between `a4cf2ea` and current `main` (at the time of the PR-prep rebase, `465abad`), no commit on `main` directly touched the three NVFP4 fix files. The diff is ~9.7k lines across upstream features, CI workflows, model additions, etc. The relevant change was much subtler:

   - At some point on `main`, the `_prefer_mixed_safetensors_files()` filter was added or strengthened. The original RC2 implementation chose `supplemental.safetensors` as a name that wouldn't trigger duplicate-key validation; that constraint was satisfied at the time. The new filter introduced a different constraint — *the filename must match `*-mixed.safetensors`* — that the supplemental name did not satisfy. Two independent decisions interacted.
   - Similarly, the `exclude_modules` consumer side of `quantization_utils.py` evolved on `main`. The original raw-BFL-name behavior may have happened to work when the consumer was lenient about which namespace the names lived in; current `main`'s consumer is keyed strictly by mapped names.

   Neither change triggered a merge conflict during the rebase, because neither touched the three NVFP4 fix files. The contract drift was *implicit*.

3. **Verification was image-pixel-based, not loader-log-based.** The original verification declared success when `mean=215.8, std=80.84` matched BF16. That's a strong signal of GEMM-level correctness, but it's a weak signal of *loader-path* correctness. If the rebased branch had silently switched to the BF16 fallback path, image stats would still match BF16 (the BF16 path generates a BF16 image), and the test would still pass. The signal that catches RC4 is the loader log: `model size: 22.9 GB` (custom path) vs `120.04 GB` (fallback). RC4 was caught only because someone looked at the log and noticed the wrong number.

**Generalized lesson**: When a quantization fix lives on a code path that has a silent fallback, image-quality verification alone is insufficient. The verification must include a check that *the intended code path actually ran*. For NVFP4: assert `model size ≈ 22.9 GB`, assert no `falling back to native version` line, assert no `Found unloaded parameters` warning. These are cheap log-grep tests that should run as part of any future NVFP4 CI verification. See updated **Guard 6** in Section 12 for the implementation sketch.

---

## Appendix: Key Commands

### Check NVFP4 image quality (quick)
```bash
python3 -c "
import numpy as np; from PIL import Image
img = Image.open('/path/to/output.png')
arr = np.array(img)
print(f'min={arr.min()}, max={arr.max()}, mean={arr.mean():.1f}, std={arr.std():.2f}')
"
# Target for healthy NVFP4 512×512: mean≈200-220, std≈70-90
# White image: mean>250, std<5
```

### Find outlier GEMM (with debug flag)
```bash
SGLANG_NVFP4_DEBUG_GEMMS=1 python3 gen_script.py 2>&1 | grep "NVFP4 GEMM" | awk -F'out_max=' '{print $2, $0}' | sort -n | tail -5
# Look for any GEMM where out_max >> 10 or nsigma >> 20
```

### Verify backend selection
```python
from sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant import _get_fp4_gemm_op
op, flashinfer_backend = _get_fp4_gemm_op()
print(f"CUTLASS: {op is not None}, flashinfer backend: {flashinfer_backend}")
# B200 without sgl_kernel: CUTLASS: False, flashinfer backend: cudnn
# H100 with sgl_kernel:    CUTLASS: True,  flashinfer backend: None
```

### Check checkpoint for missing input_scales
```python
from safetensors import safe_open
mixed = "/path/to/flux2-dev-nvfp4-mixed.safetensors"
full  = "/path/to/flux2-dev-nvfp4.safetensors"
with safe_open(mixed, framework="pt") as f: mixed_keys = set(f.keys())
with safe_open(full,  framework="pt") as f: full_keys  = set(f.keys())
missing = [k for k in full_keys - mixed_keys if "input_scale" in k]
print(f"Missing input_scale tensors: {len(missing)}")
for k in sorted(missing): print(" ", k)
```

### Run full NVFP4 end-to-end verify
```bash
PYTHONUNBUFFERED=1 python3 gen_nvfp4_verify.py 2>&1 | tee /tmp/nvfp4_verify.log
# Check: last lines should show mean≈210-220, std≈70-90
```
