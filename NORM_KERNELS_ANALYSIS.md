# Norm Kernels Migration Analysis

## Overview

This document analyzes the norm kernels that exist in the codebase and their migration status from compiled to JIT implementations.

## Kernel Inventory

### 4 Norm Kernels Identified

| Kernel Name | Purpose | Current Status |
|-------------|---------|----------------|
| **rmsnorm** | Root Mean Square normalization | ✅ Both compiled & JIT available |
| **fused_add_rmsnorm** | Fused residual add + RMSNorm | ✅ Both compiled & JIT available |
| **gemma_rmsnorm** | Gemma-style RMSNorm (weight+1) | ✅ Both compiled & JIT available |
| **gemma_fused_add_rmsnorm** | Gemma-style fused add + RMSNorm | ✅ Both compiled & JIT available |

## Current Implementation Status

### 1. **Compiled Version (sgl-kernel)**

**Location:** `/sgl-workspace/sglang/sgl-kernel/`

**Source Files:**
- `csrc/norm.cu` - FlashInfer's norm.cu (included via CMakeLists.txt line 333)
- `python/sgl_kernel/elementwise.py` - Python wrappers

**Registration:**
- ✅ Torch ops registered in `csrc/common_extension.cc` (lines 66-76)
- ✅ Function declarations in `include/sgl_kernel_ops.h` (lines 129-133)
- ✅ Compiled into `common_ops` shared library

**Python API:**
```python
import sgl_kernel

# All use torch.ops.sgl_kernel.*
sgl_kernel.rmsnorm(input, weight, eps)
sgl_kernel.fused_add_rmsnorm(input, residual, weight, eps)
sgl_kernel.gemma_rmsnorm(input, weight, eps)
sgl_kernel.gemma_fused_add_rmsnorm(input, residual, weight, eps)
```

### 2. **JIT Version (FlashInfer)**

**Location:** `/sgl-workspace/sglang/flashinfer/`

**Source Files:**
- `csrc/norm.cu` - Same source code
- `csrc/flashinfer_norm_ops.cu` - JIT registration
- `flashinfer/norm.py` - Python wrappers with JIT compilation

**JIT Module:**
- Uses `gen_jit_spec()` to create JIT-compiled module
- Compiled on-demand when first called
- Cached via `@functools.cache`

**Python API:**
```python
import flashinfer

# All use JIT compilation
flashinfer.rmsnorm(input, weight, eps)
flashinfer.fused_add_rmsnorm(input, residual, weight, eps)
flashinfer.gemma_rmsnorm(input, weight, eps)
flashinfer.gemma_fused_add_rmsnorm(input, residual, weight, eps)
```

## Kernel Details

### 1. rmsnorm

**Algorithm:** `out[i] = (input[i] / RMS(input)) * weight[i]`

**Signature:**
```python
def rmsnorm(
    input: torch.Tensor,      # (batch_size, hidden_size)
    weight: torch.Tensor,     # (hidden_size,)
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor
```

**Use Cases:**
- Standard layer normalization in transformers
- Post-attention normalization
- MLP layer normalization

### 2. fused_add_rmsnorm

**Algorithm:**
1. `residual[i] += input[i]`
2. `input[i] = (residual[i] / RMS(residual)) * weight[i]`

**Signature:**
```python
def fused_add_rmsnorm(
    input: torch.Tensor,      # (batch_size, hidden_size)
    residual: torch.Tensor,   # (batch_size, hidden_size)
    weight: torch.Tensor,     # (hidden_size,)
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
) -> None  # In-place operation
```

**Use Cases:**
- Transformer residual connections + normalization
- Reduces memory bandwidth by fusing operations
- Critical for performance in decoder layers

### 3. gemma_rmsnorm

**Algorithm:** `out[i] = (input[i] / RMS(input)) * (weight[i] + 1)`

**Signature:**
```python
def gemma_rmsnorm(
    input: torch.Tensor,      # (batch_size, hidden_size)
    weight: torch.Tensor,     # (hidden_size,)
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor
```

**Use Cases:**
- Gemma model family (Google)
- Similar to standard RMSNorm but with weight offset

### 4. gemma_fused_add_rmsnorm

**Algorithm:**
1. `residual[i] += input[i]`
2. `input[i] = (residual[i] / RMS(residual)) * (weight[i] + 1)`

**Signature:**
```python
def gemma_fused_add_rmsnorm(
    input: torch.Tensor,      # (batch_size, hidden_size)
    residual: torch.Tensor,   # (batch_size, hidden_size)
    weight: torch.Tensor,     # (hidden_size,)
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
) -> None  # In-place operation
```

**Use Cases:**
- Gemma model residual connections
- Fused operation for better performance

## Migration Strategy

Since both compiled and JIT versions already exist, we need to:

### ✅ Good News!
Unlike the renorm kernels, **norm.cu is ALREADY included** in the compiled build (CMakeLists.txt line 333), so we don't need to restore it.

### 📋 Migration Tasks

1. **Add `_compiled` suffix versions** in `sgl-kernel/python/sgl_kernel/elementwise.py`
   - Keep current functions as-is (they use compiled versions)
   - Add new `*_compiled` named exports for clarity

2. **Add FlashInfer JIT versions** to sgl_kernel
   - Import flashinfer.norm functions
   - Create wrappers without `_compiled` suffix

3. **Create benchmark script** `sgl-kernel/benchmark/bench_norm.py`
   - Compare: PyTorch reference vs Compiled vs JIT
   - Test all 4 kernels
   - Measure performance across different configurations

4. **Update `__init__.py`** exports
   - Export both `*_compiled` and regular versions

## Expected Performance Characteristics

Based on the renorm kernel benchmark results:

| Implementation | Expected Performance | Notes |
|----------------|---------------------|-------|
| **PyTorch** | Baseline (slowest) | Multiple kernel launches, memory overhead |
| **Compiled** | 10-50x faster | Single fused kernel, optimized |
| **JIT** | 1.5-5x faster than compiled | Runtime optimization, latest algorithms |

### Key Factors:
- **Hidden size impact:** Larger hidden sizes favor JIT
- **Batch size impact:** JIT maintains advantage across batch sizes
- **PDL (Programmatic Dependent Launch):** Hopper architecture optimization
- **Memory bandwidth:** Fused operations reduce data movement

## Testing Requirements

### Correctness Tests

For each kernel, test:
```python
configs = [
    # (batch_size, hidden_size)
    (1, 4096),      # Single sample
    (32, 4096),     # Small batch
    (128, 4096),    # Medium batch
    (256, 8192),    # Large batch, large hidden
]

eps_values = [1e-6, 1e-5]  # Different epsilon values
```

### Performance Tests

Configurations to benchmark:
```python
batch_sizes = [1, 16, 32, 64, 128, 256]
hidden_sizes = [2048, 4096, 5120, 8192, 11008, 14336]
```

Typical hidden sizes for popular models:
- LLaMA-7B: 4096
- LLaMA-13B: 5120
- LLaMA-70B: 8192
- Mixtral-8x7B: 4096
- Qwen-72B: 8192

## Dependencies

### Compiled Version
- ✅ Already included in sgl-kernel build
- ✅ Torch ops registered
- ✅ CMakeLists.txt configured

### JIT Version
- ✅ FlashInfer package available in `/sgl-workspace/sglang/flashinfer/`
- ✅ JIT infrastructure ready
- ⚠️ Need to import in sgl_kernel

## Next Steps

### Immediate Actions

1. **Create benchmark baseline** (Python reference implementations)
2. **Set up benchmark infrastructure** similar to `bench_renorm.py`
3. **Add `_compiled` suffixes** to existing functions
4. **Import FlashInfer JIT versions**
5. **Run benchmarks** and collect results
6. **Document recommendations** based on performance data

### Questions to Answer

1. Do JIT kernels outperform compiled versions? (Expected: Yes, by 1.5-5x)
2. How does performance scale with hidden_size?
3. Is PDL beneficial on Hopper? (Expected: Yes)
4. Should we default to JIT for production? (Likely: Yes)

## Code Structure Comparison

### Current (main branch):
```
sgl_kernel.rmsnorm → torch.ops.sgl_kernel.rmsnorm (compiled)
```

### After Migration:
```
sgl_kernel.rmsnorm_compiled → torch.ops.sgl_kernel.rmsnorm (compiled)
sgl_kernel.rmsnorm → flashinfer.rmsnorm (JIT)
```

This maintains backward compatibility while enabling performance comparison.

## Files to Modify

1. `sgl-kernel/python/sgl_kernel/elementwise.py` - Add JIT wrappers
2. `sgl-kernel/python/sgl_kernel/__init__.py` - Export new functions
3. `sgl-kernel/benchmark/bench_norm.py` - Create new benchmark
4. `sgl-kernel/benchmark/NORM_BENCHMARK_RESULTS.md` - Document results

## Summary

All 4 norm kernels are ready for migration:
- ✅ Source code exists in both compiled and JIT forms
- ✅ Compiled versions are already built and registered
- ✅ JIT infrastructure is available
- 🔄 Need to add comparison and benchmarking
- 📊 Expected outcome: JIT versions will be 1.5-5x faster

The migration follows the same pattern as renorm kernels, with the advantage that norm.cu is already in the build.
