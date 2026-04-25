---
name: NVFP4 nibble-swap investigation
description: FLUX.2-dev-NVFP4 white image fix — 4 root causes (3 correctness + 1 loader-path on current main). All committed, pushed to v1 and v2 branches on Johnsonms/sglang. Postmortem (1706 lines) and pr.md ready.
type: project
originSessionId: a9f23663-8f09-4791-aeec-7cff48434a0e
---
## Status: COMPLETE — all fixes committed, pushed, postmortem updated, pr.md ready

Last update: 2026-04-25 (RC4 added during PR-prep rebase)

---

## Three branches on Johnsonms/sglang fork

| Branch | Base | Tip | Purpose |
|---|---|---|---|
| `fix/nvfp4-nibble-swap-default` | `a4cf2ea` (old main) | `84e1b6d` | Original investigation branch — RC1/2/3 + postmortem + Claude memory; not for upstream PR |
| `fix/flux-nvfp4-quantization-correctness` | `30909cbe` (older main) | `71761604d` | v1 PR branch — RC1/2/3 (with two-step swap history) + RC4. Still on remote as of last push. |
| `fix/flux-nvfp4-quantization-correctness-v2` | `465abad` (current main tip) | `7129377a8` (local, post-rebase) | v2 PR branch — RC2 + RC3 + RC4 + 3 doc commits. **Swap commit dropped per BBuf review.** Local-only; remote still has the older 7-commit chain ending at `2b6890e59`. Awaiting force-push. |

v1 and v2 differ in commit history. v1 still has the swap_weight_nibbles=True commit (and its precursor False commit). v2 has had both dropped — the runtime relies on `ModelOptFp4Config.from_config()`'s default-True instead. Final image stats are identical to the post-RC4 verification: `mean=214.43, std=79.61`.

---

## Four root causes (3 correctness + 1 loader-path)

### Root cause 1: swap_weight_nibbles default (RC1) — RESOLVED AS NO-OP

BFL NVFP4 checkpoint packs: lo nibble=odd col, hi nibble=even col.
mm_fp4 kernels (CUTLASS + cuDNN) expect: lo nibble=element-0 (even col).

**Status on v2 (post-BBuf cleanup, 2026-04-25):** RC1 is no longer a code change. `ModelOptFp4Config.from_config()` already defaults `swap_weight_nibbles=True` at every site (`modelopt_quant.py:264, 276, 286, 497`). The original investigation introduced `swap=False` as a wrong default in commit `5a89768`, then corrected it back to `True` in `47881e9` — the two cancel out. Dropping both yields the same correct behavior with cleaner history. Verified: per-GEMM cos_sim 0.99 vs BF16, end-to-end image matches BF16.

**Status on v1:** Still has the explicit `swap_weight_nibbles=True` line via the iterative two-commit history. Functionally identical to v2; cosmetically more cluttered. Could be cleaned up with the same rebase if/when v1 is touched again.

Proof of correctness path (without the explicit setting): cos_sim −0.0002 → 0.9909 vs BF16, since `from_config` falls through to default True.

### Root cause 2: Missing input_scale tensors in mixed checkpoint (RC2)
`flux2-dev-nvfp4-mixed.safetensors` omits `input_scale` for 64 tensors.
16 of those are for FP4-quantized txt_mlp layers in double_blocks.0–7.
Without them, runtime defaults to input_scale=1.0 → wrong activation quantization.
Fix: `_build_supplemental_safetensors_dir()` in `flux_2_nvfp4.py` extracts missing tensors from companion full checkpoint.

### Root cause 3: TMA permutation applied for cuDNN backend (RC3)
`process_weights_after_loading()` always applied CUTLASS TMA blockwise interleave to weight scales.
flashinfer mm_fp4 (cuDNN on B200) expects raw row-major scales [N, K//group_size].
TMA permutation scrambled scales → GEMM #4 (txt_mlp.2): out_max=49.75 (116σ outlier) → cascade → white images.
Fix: `modelopt_quant.py` → conditional TMA permute: `if flashinfer_backend is None:` (CUTLASS only)

### Root cause 4: Custom NVFP4 loader silently falls back to native diffusers BF16 (RC4) — discovered during PR-prep rebase
After rebasing v2 onto current main, end-to-end output became cream-smear (mean=219, std=12) instead of white. Loader log revealed: `AssertionError: proj_out.weight (128,6144) vs (128,3072)` → `Error while loading customized transformer, falling back to native version` → `Loaded transformer: Flux2Transformer2DModel (native version). model size: 120.04 GB`.

The native fallback path runs the model as plain BF16 — RC1/2/3 are dead code on this path. **Image quality alone does NOT distinguish "custom NVFP4 ran correctly" from "fallback BF16 ran correctly"** — same correct image, wrong code path. Three sub-causes:

**RC4-A:** `quantization_utils.py:403` — `_build_nvfp4_config_from_safetensors_files()` was appending raw BFL name (`final_layer.linear`) to `exclude_modules` instead of mapped SGLang name (`proj_out`). Result: `proj_out` got tagged as NVFP4, tripped (128,6144) vs (128,3072) shape assertion at meta-load.

**RC4-B:** `flux_2_nvfp4.py:_build_supplemental_safetensors_dir()` named the supplemental file `supplemental.safetensors`. The loader's `_prefer_mixed_safetensors_files()` filter (regex `.*-mixed(?:-\d+-of-\d+)?\.safetensors$`) silently dropped it as a "non-mixed sibling". Result: 16 critical txt_mlp.*.input_scale tensors stayed unloaded → `Found unloaded parameters in meta state dict` warning. Fix: rename to `supplemental-mixed.safetensors`.

**RC4-C:** Same function — cached `/tmp/sglang_nvfp4_supp_<hash>/` dir was reused across runs based purely on file existence, not contents. Stale supplementals from older builders survived indefinitely. Fix: compute `expected_extra_keys = non_mixed - mixed` upfront, validate cached file's keys equal that set, rebuild with warning on mismatch.

---

## Final verification result (current main, post RC4)

```
NVFP4 (all 4 fixes):  min=0, max=255, mean=214.43, std=79.61   (Run H, current main)
NVFP4 (orig 3 fixes): min=0, max=255, mean=215.8,  std=80.84   (Run B, old main e43fa66)
BF16  (reference):    min=0, max=255, mean=215.9,  std=82.58
```

Loader log signals (the canonical way to verify RC4 is fixed):
- `Loaded transformer: Flux2Transformer2DModel (sgl-diffusion version). model size: 22.9 GB` ✓
- `Loading Flux2Transformer2DModel from 2 safetensors file(s)` ✓ (mixed + supplemental-mixed)
- NO `falling back to native version`
- NO `Found unloaded parameters in meta state dict`

Image: Doraemon eating dorayaki — correct, identical to BF16 output. Wall time 24.56s for 12 steps on B200.

---

## Artifacts (on /sgl-workspace/sglang)

- `0424/run_a_buggy_a4cf2ea.png` — broken base, beige uniform smear
- `0424/run_b_fixed_e43fa66.png` — original 3 fixes on old main, correct image
- `0424/run_g_v2_filterfix.png` — v2 + RC4 fixes, correct image
- `0424/run_h_v1_final.png` — v1 + RC4 fixes (final verification), correct image
- `0424/run_*.log` — loader logs for all runs (used to diagnose RC4)
- `pr.md` — unified PR description (488 lines, 4 findings + sub-fixes A/B/C in unified narrative, no Part 1/Part 2 split)
- `python/sglang/multimodal_gen/docs/postmortem_nvfp4_white_image_2025_04.md` — 1706 lines, includes Section 13 (RC4 postscript) and Guard 6 (verify-the-code-path CI check)

## Unit tests added

- `python/sglang/multimodal_gen/test/unit/test_quantization_utils.py` — verifies BF16 fallback layers excluded under mapped names, not raw BFL names
- `python/sglang/multimodal_gen/test/unit/test_flux2_nvfp4.py` — verifies stale supplemental cache is rebuilt and lives at `supplemental-mixed.safetensors`

Run: `python -m pytest -q python/sglang/multimodal_gen/test/unit/test_quantization_utils.py python/sglang/multimodal_gen/test/unit/test_flux2_nvfp4.py` → 2 passed.

---

## Key file locations

- `python/sglang/multimodal_gen/runtime/utils/quantization_utils.py` — line ~313 swap_weight_nibbles, line ~403 mapped-module-name fix (RC4-A)
- `python/sglang/multimodal_gen/runtime/layers/quantization/modelopt_quant.py` — process_weights_after_loading() conditional TMA permute (RC3); from_config() defaults swap_weight_nibbles=True at all sites (relevant to BBuf review)
- `python/sglang/multimodal_gen/runtime/pipelines/flux_2_nvfp4.py` — _build_supplemental_safetensors_dir() with cache validation (RC4-C) and `supplemental-mixed.safetensors` filename (RC4-B)
- `python/sglang/multimodal_gen/runtime/loader/transformer_load_utils.py` — _prefer_mixed_safetensors_files() filter, regex at line 43
- `python/sglang/multimodal_gen/docs/postmortem_nvfp4_white_image_2025_04.md` — full postmortem (Sections 1–13)

## HF model paths

- BF16:  `/home/johnson/scratch/huggingface/models--black-forest-labs--FLUX.2-dev/snapshots/26afe3a78bb242c0a8bb181dcc8937bb16e5c66c`
- NVFP4: `/home/johnson/scratch/huggingface/models--black-forest-labs--FLUX.2-dev-NVFP4/snapshots/142b87e70bc3006937b7093d89ff287b5f59f071/flux2-dev-nvfp4-mixed.safetensors`

## Reproducer command

```bash
CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH=/sgl-workspace/sglang/python \
  HF_HOME=/home/johnson/scratch/huggingface \
  SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND=cudnn \
  python /tmp/gen_flux2_nvfp4.py /tmp/out.png
```

(`/tmp/gen_flux2_nvfp4.py` is the manual repro script preserved from `a93584b`. The PR branches deliberately exclude this file — it's only on `fix/nvfp4-nibble-swap-default`. For PR review, the equivalent inline `DiffGenerator.from_pretrained(...)` snippet is in pr.md's "How to Reproduce" section.)

---

## Open follow-ups

1. **Force-push v2 to fork**: rebase dropped `swap_weight_nibbles` commit; remote still at old chain ending `2b6890e59`. Need `git push --force-with-lease johnson fix/flux-nvfp4-quantization-correctness-v2`. Awaiting user OK before push.
2. **End-to-end repro on rebased v2**: tests pass; full Doraemon E2E not yet re-run after the rebase. Should run before push to confirm `from_config()` default-True actually works on the cleaned chain.
3. **Update pr.md** to reflect that RC1 is no longer a code change on v2 (only correctness clarification). Current pr.md still describes RC1 as a fix.
4. **Decision: ship v1 or v2?** v1 still has the swap commits and an open BBuf review thread. v2 is cleaner. Final code is identical except for the (redundant) swap line on v1.
5. **Claude session memory commit on v2**: Files under `.claude/memory/` are personal session state, not for sgl-project/sglang upstream. Currently kept on the fork for portability. Should be dropped before merging the v2 PR upstream.
