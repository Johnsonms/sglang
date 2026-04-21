"""
Single-shot FLUX.2-dev-NVFP4 image generation. Run twice:
  1. With buggy code  (swap_weight_nibbles default=True)  → before image
  2. With fixed code  (swap_weight_nibbles default=False) → after image

Usage:
    HF_HOME=/home/johnson/scratch/huggingface \
    SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND=cudnn \
    python python/sglang/multimodal_gen/test/manual/gen_flux2_nvfp4.py <output_filename>
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../.."))

from sglang.multimodal_gen import DiffGenerator

output = sys.argv[1] if len(sys.argv) > 1 else "flux2_nvfp4_out.png"

gen = DiffGenerator.from_pretrained(model_path="black-forest-labs/FLUX.2-dev-NVFP4")
result = gen.generate(sampling_params_kwargs={
    "prompt": "Doraemon is eating dorayaki",
    "height": 768,
    "width": 768,
    "num_inference_steps": 12,
    "save_output": True,
    "output_file_name": os.path.basename(output),
    "output_path": os.path.dirname(output) or ".",
})
print(f"Saved → {getattr(result, 'output_file_path', output)}")
