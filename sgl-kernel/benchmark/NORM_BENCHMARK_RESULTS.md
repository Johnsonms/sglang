# Norm Kernel Benchmark Results

## Executive Summary

This document presents comprehensive benchmark results comparing three implementations of norm kernels:
- **PyTorch Reference**: Native PyTorch implementation
- **SGL Kernel (Compiled)**: Statically compiled FlashInfer kernels
- **SGL Kernel (JIT)**: Just-In-Time compiled FlashInfer kernels

**Key Findings:**
- ✅ All correctness checks passed for float16 data type
- 🚀 Compiled and JIT kernels show **3-5x speedup** over PyTorch reference
- ⚡ JIT kernels perform on par with compiled kernels (within 5% performance)
- 📊 Both implementations maintain consistent performance across batch sizes
- ⚠️ JIT kernels currently only support float16 (bfloat16 support pending)

## Test Environment

- **GPU**: NVIDIA B200 (Blackwell architecture)
- **CUDA**: 12.x
- **PyTorch**: Latest version
- **Data Type**: float16 (for benchmarks)
- **Test Date**: 2026-02-15

## Benchmark Configurations

```python
batch_sizes = [1, 32, 128, 256]
hidden_sizes = [2048, 4096, 8192, 11008]
eps = 1e-6
dtype = torch.float16
```

## Detailed Results

### 1. RMSNorm Performance

**Operation**: `out = (input / RMS(input)) * weight`

| Batch | Hidden | Torch (μs) | Compiled (μs) | JIT (μs) | Speedup (Compiled) | Speedup (JIT) |
|-------|--------|------------|---------------|----------|-------------------|---------------|
| 1     | 2048   | 39.98      | 11.33         | 11.33    | **3.53x**         | **3.53x**     |
| 1     | 4096   | 39.97      | 12.13         | 11.42    | **3.30x**         | **3.50x**     |
| 1     | 8192   | 39.94      | 13.31         | 13.31    | **3.00x**         | **3.00x**     |
| 1     | 11008  | 42.02      | 14.43         | 14.30    | **2.91x**         | **2.94x**     |
| 32    | 2048   | 51.97      | 11.33         | 11.33    | **4.59x**         | **4.59x**     |
| 32    | 4096   | 57.34      | 11.42         | 11.36    | **5.02x**         | **5.05x**     |
| 32    | 8192   | 45.92      | 13.31         | 13.31    | **3.45x**         | **3.45x**     |
| 32    | 11008  | 50.02      | 14.30         | 14.21    | **3.50x**         | **3.52x**     |
| 128   | 2048   | 55.39      | 11.39         | 11.39    | **4.86x**         | **4.86x**     |
| 128   | 4096   | 60.48      | 13.15         | 12.43    | **4.60x**         | **4.87x**     |
| 128   | 8192   | 51.04      | 13.34         | 13.34    | **3.83x**         | **3.83x**     |
| 128   | 11008  | 55.14      | 15.39         | 15.39    | **3.58x**         | **3.58x**     |
| 256   | 2048   | 56.45      | 12.26         | 12.22    | **4.61x**         | **4.62x**     |
| 256   | 4096   | 62.50      | 13.34         | 13.34    | **4.68x**         | **4.68x**     |
| 256   | 8192   | 58.66      | 15.39         | 15.39    | **3.81x**         | **3.81x**     |
| 256   | 11008  | 66.53      | 17.50         | 17.47    | **3.80x**         | **3.81x**     |

**Analysis:**
- Compiled and JIT versions show virtually identical performance
- Speedup ranges from **2.91x to 5.05x** over PyTorch
- Higher speedups at larger batch sizes (32, 128, 256)
- Performance is consistent across hidden sizes

### 2. Fused Add + RMSNorm Performance

**Operation**: `residual += input; input = (residual / RMS(residual)) * weight`

| Batch | Hidden | Torch (μs) | Compiled (μs) | JIT (μs) | Speedup (Compiled) | Speedup (JIT) |
|-------|--------|------------|---------------|----------|-------------------|---------------|
| 1     | 2048   | 47.94      | 15.46         | 15.49    | **3.10x**         | **3.09x**     |
| 1     | 4096   | 47.23      | 17.50         | 17.44    | **2.70x**         | **2.71x**     |
| 1     | 8192   | 51.07      | 18.40         | 18.43    | **2.78x**         | **2.77x**     |
| 1     | 11008  | 51.17      | 20.48         | 20.48    | **2.50x**         | **2.50x**     |
| 32    | 2048   | 62.46      | 17.31         | 17.31    | **3.61x**         | **3.61x**     |
| 32    | 4096   | 68.58      | 19.30         | 18.43    | **3.55x**         | **3.72x**     |
| 32    | 8192   | 57.20      | 19.42         | 19.42    | **2.95x**         | **2.95x**     |
| 32    | 11008  | 58.37      | 21.50         | 21.50    | **2.71x**         | **2.71x**     |
| 128   | 2048   | 64.54      | 17.44         | 17.41    | **3.70x**         | **3.71x**     |
| 128   | 4096   | 70.66      | 19.42         | 19.42    | **3.64x**         | **3.64x**     |
| 128   | 8192   | 62.43      | 21.47         | 21.33    | **2.91x**         | **2.93x**     |
| 128   | 11008  | 66.62      | 21.47         | 21.50    | **3.10x**         | **3.10x**     |
| 256   | 2048   | 65.34      | 19.42         | 19.42    | **3.36x**         | **3.36x**     |
| 256   | 4096   | 74.75      | 19.49         | 19.46    | **3.84x**         | **3.84x**     |
| 256   | 8192   | 68.77      | 25.63         | 23.58    | **2.68x**         | **2.92x**     |
| 256   | 11008  | 82.98      | 29.70         | 27.65    | **2.79x**         | **3.00x**     |

**Analysis:**
- JIT version shows slight advantage in larger configurations (256 batch, 8192+ hidden)
- Speedup ranges from **2.50x to 3.84x** over PyTorch
- Fused operation reduces memory bandwidth by combining residual add and norm
- Most efficient at medium to large batch sizes

### 3. Gemma RMSNorm Performance

**Operation**: `out = (input / RMS(input)) * (weight + 1)`

| Batch | Hidden | Torch (μs) | Compiled (μs) | JIT (μs) | Speedup (Compiled) | Speedup (JIT) |
|-------|--------|------------|---------------|----------|-------------------|---------------|
| 1     | 2048   | 43.04      | 11.33         | 11.33    | **3.80x**         | **3.80x**     |
| 1     | 4096   | 43.01      | 12.16         | 11.42    | **3.54x**         | **3.77x**     |
| 1     | 8192   | 46.94      | 13.31         | 13.31    | **3.53x**         | **3.53x**     |
| 1     | 11008  | 47.94      | 15.12         | 14.37    | **3.17x**         | **3.34x**     |
| 32    | 2048   | 54.21      | 11.33         | 11.36    | **4.79x**         | **4.77x**     |
| 32    | 4096   | 58.75      | 11.36         | 11.36    | **5.17x**         | **5.17x**     |
| 32    | 8192   | 50.24      | 13.31         | 13.31    | **3.77x**         | **3.77x**     |
| 32    | 11008  | 54.27      | 14.27         | 14.24    | **3.80x**         | **3.81x**     |
| 128   | 2048   | 57.34      | 11.39         | 11.36    | **5.03x**         | **5.05x**     |
| 128   | 4096   | 63.68      | 13.15         | 12.38    | **4.84x**         | **5.14x**     |
| 128   | 8192   | 55.36      | 13.34         | 13.34    | **4.15x**         | **4.15x**     |
| 128   | 11008  | 60.51      | 15.39         | 15.39    | **3.93x**         | **3.93x**     |
| 256   | 2048   | 60.42      | 12.26         | 12.24    | **4.93x**         | **4.94x**     |
| 256   | 4096   | 67.62      | 13.34         | 13.34    | **5.07x**         | **5.07x**     |
| 256   | 8192   | 62.40      | 15.39         | 15.39    | **4.05x**         | **4.05x**     |
| 256   | 11008  | 72.70      | 17.54         | 17.47    | **4.14x**         | **4.16x**     |

**Analysis:**
- Gemma variant shows similar performance to standard RMSNorm
- Additional weight offset (+1) has negligible performance impact
- Speedup ranges from **3.17x to 5.17x** over PyTorch
- Highest speedups at batch size 32 and 128

### 4. Gemma Fused Add + RMSNorm Performance

**Operation**: `residual += input; input = (residual / RMS(residual)) * (weight + 1)`

| Batch | Hidden | Torch (μs) | Compiled (μs) | JIT (μs) | Speedup (Compiled) | Speedup (JIT) |
|-------|--------|------------|---------------|----------|-------------------|---------------|
| 1     | 2048   | 52.06      | 15.55         | 15.46    | **3.35x**         | **3.37x**     |
| 1     | 4096   | 51.30      | 17.50         | 17.44    | **2.93x**         | **2.94x**     |
| 1     | 8192   | 55.15      | 18.43         | 18.43    | **2.99x**         | **2.99x**     |
| 1     | 11008  | 56.29      | 20.48         | 20.48    | **2.75x**         | **2.75x**     |
| 32    | 2048   | 64.54      | 17.28         | 17.28    | **3.74x**         | **3.74x**     |
| 32    | 4096   | 72.67      | 17.57         | 18.37    | **4.14x**         | **3.96x**     |
| 32    | 8192   | 60.54      | 19.42         | 19.42    | **3.12x**         | **3.12x**     |
| 32    | 11008  | 64.34      | 21.47         | 21.47    | **3.00x**         | **3.00x**     |
| 128   | 2048   | 68.64      | 17.41         | 17.41    | **3.94x**         | **3.94x**     |
| 128   | 4096   | 74.75      | 19.42         | 19.42    | **3.85x**         | **3.85x**     |
| 128   | 8192   | 66.56      | 21.47         | 21.34    | **3.10x**         | **3.12x**     |
| 128   | 11008  | 72.70      | 21.47         | 21.50    | **3.39x**         | **3.38x**     |
| 256   | 2048   | 70.59      | 19.42         | 19.42    | **3.63x**         | **3.63x**     |
| 256   | 4096   | 78.85      | 19.49         | 19.46    | **4.05x**         | **4.05x**     |
| 256   | 8192   | 73.70      | 25.63         | 23.58    | **2.88x**         | **3.12x**     |
| 256   | 11008  | 87.07      | 29.70         | 27.62    | **2.93x**         | **3.15x**     |

**Analysis:**
- Similar patterns to standard fused_add_rmsnorm
- JIT version shows advantage at largest configuration (256×11008)
- Speedup ranges from **2.75x to 4.14x** over PyTorch
- Gemma variant maintains performance parity with standard norm

## Performance Summary

### Average Speedups

| Kernel                      | Compiled vs PyTorch | JIT vs PyTorch | JIT vs Compiled |
|-----------------------------|---------------------|----------------|-----------------|
| rmsnorm                     | 3.92x              | 3.95x          | 1.01x           |
| fused_add_rmsnorm           | 3.11x              | 3.15x          | 1.01x           |
| gemma_rmsnorm               | 4.22x              | 4.26x          | 1.01x           |
| gemma_fused_add_rmsnorm     | 3.36x              | 3.38x          | 1.01x           |
| **Overall Average**         | **3.65x**          | **3.69x**      | **1.01x**      |

### Key Observations

1. **Compiled vs JIT Performance**: Near-identical (within 1% on average)
   - JIT compilation achieves parity with static compilation
   - No performance penalty for runtime compilation
   - Both implementations likely use same optimized CUDA kernels

2. **Batch Size Impact**:
   - Larger batch sizes (32, 128, 256) show better speedups
   - Batch size 1 still achieves 2.5-3.8x improvement
   - Optimal performance at batch sizes 32-128

3. **Hidden Size Impact**:
   - Performance scales well across hidden sizes (2048-11008)
   - Slight performance advantage at smaller hidden sizes (2048, 4096)
   - Larger hidden sizes (11008) maintain good speedup

4. **Fused Operations**:
   - Fused kernels show lower absolute speedup due to more complex operations
   - Still achieve 2.5-3.8x improvement over PyTorch
   - Memory bandwidth savings not reflected in latency alone

## Unit Test Results

```
Test Suite: tests/test_norm.py
Total Tests: 224
Passed: 192 (85.7%)
Failed: 32 (14.3%)
```

### Test Breakdown

- ✅ **All float16 tests passed** (192/192)
  - rmsnorm: 64/64 passed
  - fused_add_rmsnorm: 32/32 passed (float16)
  - gemma_rmsnorm: 64/64 passed
  - gemma_fused_add_rmsnorm: 32/32 passed (float16)

- ⚠️ **bfloat16 tests failed for JIT kernels** (32/32)
  - fused_add_rmsnorm (bfloat16): 32 failures
  - Error: "failed to dispatch data type"
  - Root cause: JIT kernels currently only support float16

### Known Limitations

1. **Data Type Support**:
   - Compiled kernels: ✅ float16, ✅ bfloat16
   - JIT kernels: ✅ float16, ❌ bfloat16 (pending)

2. **Workaround**:
   - Use compiled versions for bfloat16: `sgl_kernel.*_compiled()`
   - Use JIT versions for float16: `sgl_kernel.*()`

## Recommendations

### For Production Use

1. **Default to JIT kernels** for float16 workloads:
   ```python
   import sgl_kernel

   # Recommended - JIT version (float16 only)
   output = sgl_kernel.rmsnorm(input, weight, eps)
   output = sgl_kernel.fused_add_rmsnorm(input, residual, weight, eps)
   ```

2. **Use compiled kernels** for bfloat16 workloads:
   ```python
   # For bfloat16 support
   output = sgl_kernel.rmsnorm_compiled(input, weight, eps)
   output = sgl_kernel.fused_add_rmsnorm_compiled(input, residual, weight, eps)
   ```

3. **Prefer fused operations** when possible:
   - Use `fused_add_rmsnorm` instead of separate add + rmsnorm
   - Reduces memory bandwidth and improves end-to-end performance

### Performance Tuning

1. **Batch Size**:
   - Target batch sizes of 32-128 for optimal performance
   - Acceptable performance even at batch size 1

2. **Hidden Size**:
   - All tested hidden sizes (2048-11008) show good performance
   - No special tuning needed for different model sizes

3. **Data Type**:
   - Float16 recommended for best compatibility
   - Bfloat16 support coming in future JIT releases

## Future Work

1. **JIT Enhancements**:
   - [ ] Add bfloat16 support to JIT kernels
   - [ ] Test on additional GPU architectures (Hopper, Ada)
   - [ ] Benchmark with PDL (Programmatic Dependent Launch)

2. **Additional Kernels**:
   - [ ] LayerNorm variants
   - [ ] GroupNorm support
   - [ ] Mixed precision normalization

3. **Performance Optimization**:
   - [ ] Multi-stream execution
   - [ ] Kernel fusion with adjacent operations
   - [ ] Memory layout optimization

## Conclusion

The norm kernel migration successfully provides both compiled and JIT implementations with excellent performance characteristics:

- ✅ **3.65-3.69x average speedup** over PyTorch reference
- ✅ **JIT and compiled versions perform identically** (within 1%)
- ✅ **All float16 correctness tests pass**
- ✅ **Consistent performance across batch sizes and hidden sizes**
- ⚠️ **bfloat16 support pending for JIT kernels**

**Recommendation**: Use JIT kernels by default for float16 workloads. The performance is identical to compiled kernels while providing runtime flexibility and latest algorithm updates.

---

*Benchmark conducted on NVIDIA B200 GPU with CUDA 12.x, PyTorch latest, February 2026*
