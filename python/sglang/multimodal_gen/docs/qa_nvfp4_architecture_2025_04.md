# NVFP4 Architecture Q&A — April 2025

Companion to `postmortem_nvfp4_white_image_2025_04.md`.
Questions and answers from post-fix investigation of the FLUX.2-dev NVFP4 checkpoint structure.

---

## Q1: In single blocks, are both attention and feedforward FP4-quantized?

**Yes — both are NVFP4**, but they are **fused** in the checkpoint rather than stored as separate `attn.*` / `mlp.*` tensors.

FLUX single blocks fuse all projections into two linear layers:

| Tensor | What it contains |
|--------|-----------------|
| `single_blocks.N.linear1` | QKV projection + MLP fc1 (concatenated along output dim) |
| `single_blocks.N.linear2` | Attention out projection + MLP fc2 (concatenated) |

Both tensors carry `weight`, `weight_scale`, `weight_scale_2`, and `input_scale` — the full NVFP4 set.

The `attn.to_qkv` / `mlp.0` naming visible in the BF16 checkpoint is the un-fused view. At runtime, the single-block forward pass splits `linear1` / `linear2` back into the attention and MLP branches.

---

## Q2: In MM-DiT (double blocks), are attention and feedforward both FP4?

**No — attention is BF16; only the MLP layers are FP4.**

Verified directly from checkpoint keys in `flux2-dev-nvfp4-mixed.safetensors`:

| Layer | Quantization | Evidence |
|-------|-------------|---------|
| `img_attn.qkv` | **BF16** | `weight` only — no scale tensors |
| `img_attn.proj` | **BF16** | `weight` only — no scale tensors |
| `txt_attn.qkv` | **BF16** | `weight` only — no scale tensors |
| `txt_attn.proj` | **BF16** | `weight` only — no scale tensors |
| `img_mlp.0` / `.2` | **NVFP4** | `weight` + `weight_scale` + `weight_scale_2` + `input_scale` |
| `txt_mlp.0` / `.2` | **Partial** | `weight` + `weight_scale` + `weight_scale_2`, but **no `input_scale`** |

The missing `input_scale` on `txt_mlp` is the direct manifestation of **root cause 2** (see postmortem §3.2): the mixed checkpoint omits `input_scale` for `txt_mlp` layers in `double_blocks.0`–`double_blocks.7` (16 tensors total). Without them, runtime defaults to `input_scale=1.0`, causing incorrect activation quantization for those layers.

---

## Summary: Quantization map across block types

| Block type | Attention | MLP / feedforward |
|------------|-----------|-------------------|
| Single blocks (×48) | NVFP4 (fused in `linear1`/`linear2`) | NVFP4 (fused in `linear1`/`linear2`) |
| Double blocks / MM-DiT (×19) | **BF16** | NVFP4 (`img_mlp` full; `txt_mlp` weight-only in mixed ckpt) |
