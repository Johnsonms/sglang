---
name: NVFP4 nibble-swap investigation
description: FLUX.2-dev-NVFP4 white image fix — all 3 root causes found, fixed, committed, pushed. Postmortem written. Ready for PR tomorrow.
type: project
originSessionId: a9f23663-8f09-4791-aeec-7cff48434a0e
---
## Status: COMPLETE — all fixes committed and pushed to remote

Branch: `fix/nvfp4-nibble-swap-default`  
Remote: `johnson` → `https://github.com/Johnsonms/sglang`  
Pushed: 2025-04-22

---

## Commits on branch (on top of main)

```
bff620e  diffusion: add NVFP4 white image postmortem (2025-04)
e43fa66  diffusion: skip CUTLASS TMA scale permutation for flashinfer/cuDNN backend
47881e9  diffusion: fix swap_weight_nibbles=True for BFL NVFP4 checkpoints
d3cbe94  diffusion: fix FLUX.2-dev-NVFP4 mixed checkpoint missing input_scale tensors
```

---

## Three root causes

### Root cause 1: swap_weight_nibbles=False (commit 47881e9)
BFL NVFP4 checkpoint packs: lo nibble=odd col, hi nibble=even col.  
mm_fp4 kernels (CUTLASS + cuDNN) expect: lo nibble=element-0 (even col).  
Fix: `quantization_utils.py` → `"swap_weight_nibbles": True`  
Proof: cos_sim −0.0002 → 0.9909 vs BF16

### Root cause 2: Missing input_scale tensors in mixed checkpoint (commit d3cbe94)
`flux2-dev-nvfp4-mixed.safetensors` omits `input_scale` for 64 tensors.  
16 of those are for FP4-quantized txt_mlp layers in double_blocks.0–7.  
Without them, runtime defaults to input_scale=1.0 → wrong activation quantization.  
Fix: `_build_supplemental_safetensors_dir()` in `flux_2_nvfp4.py` extracts missing tensors from companion full checkpoint.

### Root cause 3: TMA permutation applied for cuDNN backend (commit e43fa66)
`process_weights_after_loading()` always applied CUTLASS TMA blockwise interleave to weight scales.  
flashinfer mm_fp4 (cuDNN on B200) expects raw row-major scales [N, K//group_size].  
TMA permutation scrambled scales → GEMM #4 (txt_mlp.2): out_max=49.75 (116σ outlier) → cascade → white images.  
Fix: `modelopt_quant.py` → conditional TMA permute: `if flashinfer_backend is None:` (CUTLASS only)

---

## Final verification result

```
NVFP4 (all 3 fixes):  min=0, max=255, mean=215.8, std=80.84
BF16  (reference):    min=0, max=255, mean=215.9, std=82.58
```
Image: Doraemon eating dorayaki — correct, high quality. ✓

---

## Postmortem document

Path: `python/sglang/multimodal_gen/docs/postmortem_nvfp4_white_image_2025_04.md`  
Committed: bff620e  
1157 lines covering: FLUX.2 architecture, NVFP4 format deep dive, all 3 root causes with discovery paths, dead ends, verification methodology, debugging principles, prevention guards.  
**User plans to review this tomorrow and use it for learning + interview prep.**

---

## Next steps (for tomorrow)

1. Go through postmortem document with user — they want to walk through it step by step
2. Create PR: `fix/nvfp4-nibble-swap-default` → `main` on `sgl-project/sglang`
3. CI test: `python -m pytest python/sglang/multimodal_gen/test/server/test_server_c.py -v -k "nvfp4"`

---

## Key file locations

- `python/sglang/multimodal_gen/runtime/utils/quantization_utils.py` line ~313 — swap_weight_nibbles
- `python/sglang/multimodal_gen/runtime/layers/quantization/modelopt_quant.py` — process_weights_after_loading(), TMA fix
- `python/sglang/multimodal_gen/runtime/pipelines/flux_2_nvfp4.py` — _build_supplemental_safetensors_dir()
- `python/sglang/multimodal_gen/docs/postmortem_nvfp4_white_image_2025_04.md` — postmortem

## HF model paths

- BF16:  `/home/johnson/scratch/huggingface/models--black-forest-labs--FLUX.2-dev/snapshots/26afe3a78bb242c0a8bb181dcc8937bb16e5c66c`
- NVFP4: `/home/johnson/scratch/huggingface/models--black-forest-labs--FLUX.2-dev-NVFP4/snapshots/142b87e70bc3006937b7093d89ff287b5f59f071/flux2-dev-nvfp4-mixed.safetensors`
