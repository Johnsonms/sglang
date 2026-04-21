"""
Repro script for the NVFP4 nibble-swap default bug.

NVFP4 bytes are packed as: low-nibble = element[2*col], high-nibble = element[2*col+1].
The nibble swap (w >> 4) | (w << 4) reverses this within each byte.

The CUTLASS/CuDNN FP4 GEMM kernel expects the swapped layout, so swap_weight_nibbles
must be True for checkpoints that use the standard ModelOpt packing (e.g.
black-forest-labs/FLUX.2-dev-NVFP4).  Checkpoints built with
build_modelopt_nvfp4_transformer.py --pattern-preset flux1-nvfp4 already store
weights in the kernel-ready layout and set swap_weight_nibbles=False explicitly.

Default: swap_weight_nibbles=True  → kernel-ready layout  (correct for HF NVFP4 checkpoints)
Explicit False in config.json      → weights already in kernel layout (correct for built checkpoints)

Run (CPU-only, no GPU required):
    python python/sglang/multimodal_gen/test/manual/repro_nvfp4_nibble_swap.py
"""

import sys
import torch

sys.path.insert(
    0,
    __file__.split("python/sglang")[0] + "python",
)

from sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    _prepare_nvfp4_weight_bytes,
)


def pack_nvfp4_bytes(values: list[int]) -> torch.Tensor:
    """Pack a list of FP4 nibbles (0-15) into bytes using standard ModelOpt layout.

    Low nibble  = values[2*i]
    High nibble = values[2*i + 1]
    """
    assert len(values) % 2 == 0, "Need even number of FP4 values"
    packed = []
    for i in range(0, len(values), 2):
        lo = values[i] & 0xF
        hi = values[i + 1] & 0xF
        packed.append(lo | (hi << 4))
    return torch.tensor(packed, dtype=torch.uint8)


def show_swap_effect(values: list[int]) -> None:
    packed = pack_nvfp4_bytes(values)
    swapped = _prepare_nvfp4_weight_bytes(packed.unsqueeze(0), swap_weight_nibbles=True).squeeze(0)
    not_swapped = _prepare_nvfp4_weight_bytes(packed.unsqueeze(0), swap_weight_nibbles=False).squeeze(0)
    print(f"  Original packed bytes  (standard ModelOpt):  {[hex(x) for x in packed.tolist()]}")
    print(f"  After swap=True        (kernel-ready layout): {[hex(x) for x in swapped.tolist()]}")
    print(f"  After swap=False       (unchanged):           {[hex(x) for x in not_swapped.tolist()]}")


def show_default_config_swap():
    cfg_default = ModelOptFp4Config.from_config(
        {"quant_algo": "NVFP4", "group_size": 16, "ignore": []}
    )
    cfg_explicit_false = ModelOptFp4Config.from_config(
        {"quant_algo": "NVFP4", "group_size": 16, "ignore": [], "swap_weight_nibbles": False}
    )
    print(f"  Config WITHOUT swap_weight_nibbles field  → swap={cfg_default.swap_weight_nibbles}")
    print(f"  Config WITH    swap_weight_nibbles=false  → swap={cfg_explicit_false.swap_weight_nibbles}")
    return cfg_default.swap_weight_nibbles


print("=" * 60)
print("NVFP4 nibble-swap layout demo")
print("=" * 60)

fp4_values = [3, 7, 0xA, 0xF]

print()
print("Step 1: Default config value for swap_weight_nibbles")
default_swap = show_default_config_swap()

print()
print("Step 2: Effect of nibble swap on packed bytes")
print(f"  FP4 values to pack: {fp4_values}")
show_swap_effect(fp4_values)

print()
print("=" * 60)
print("Summary")
print("=" * 60)
print("  HF NVFP4 checkpoints (e.g. FLUX.2-dev-NVFP4): swap=True  (default)")
print("  Built checkpoints with flux1-nvfp4 preset:     swap=False (explicit in config.json)")
print(f"  Current default: swap={default_swap}")
