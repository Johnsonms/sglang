# Fused Precomputation Kernel for NSA Decode Mode Metadata

## Overview

This optimization implements a fused CUDA kernel for `_precompute_decode_mode` in the Native Sparse Attention (NSA) backend. The kernel consolidates multiple operations into a single kernel launch, reducing CPU launch overhead and improving performance.

## Problem

Previously, `_precompute_decode_mode` performed multiple separate operations:
1. dtype conversion (seq_lens → int32)
2. Cumulative sum with padding (cache_seqlens → cu_seqlens_k)
3. NSA seqlens computation (clamp operation)
4. Cumulative sum for NSA (nsa_cache_seqlens → nsa_cu_seqlens_k)
5. Page table gathering from req_to_token
6. Page table transformation (if real_page_size > 1)

Each operation potentially triggered separate kernel launches, resulting in:
- Multiple CPU-GPU synchronization points
- Kernel launch overhead (5-6+ launches)
- Inefficient memory access patterns

## Solution

The `precompute_decode_metadata_cuda` kernel fuses all operations into a single kernel launch that:

1. **Uses shared memory** for intermediate results (cache_seqlens, nsa_cache_seqlens, cumulative sums)
2. **Implements parallel prefix sum** (Blelloch scan) for efficient cumulative sum computation
3. **Performs all transformations in parallel** across thread blocks
4. **Handles variable batch sizes** up to 256 (configurable via MAX_SHARED_BS)

### Key Design Decisions

#### Shared Memory Strategy
- Batch sizes ≤256 fit comfortably in shared memory (48KB typical)
- Intermediate results stay in fast shared memory, minimizing global memory traffic
- All threads synchronize at key points to ensure correctness

#### Parallel Prefix Sum
- Uses Blelloch scan algorithm for O(n) work complexity
- Efficiently computes cumulative sums in shared memory
- Supports both cu_seqlens_k and nsa_cu_seqlens_k computation

#### Single Block Execution
- Uses one thread block with 256 threads
- Simplifies synchronization (only __syncthreads() needed)
- Efficient for small-to-medium batch sizes typical in decode mode

## Performance Benefits

### Expected Improvements
- **Reduced kernel launches**: 5-6 launches → 1 launch
- **Lower CPU overhead**: Single launch instead of multiple dispatch calls
- **Better memory efficiency**: Shared memory reduces global memory traffic
- **Improved latency**: Critical for low-latency multi-step speculative decoding

### When the Kernel is Used
The kernel is automatically used when:
1. Batch size ≤ 256
2. `sgl_kernel` extension is compiled with the new kernel
3. Python can import `precompute_decode_metadata_cuda`

Otherwise, the code falls back to the original Python implementation.

## Implementation Details

### File Locations

**CUDA Kernel:**
- File: `csrc/elementwise/precompute_decode_metadata.cu`
- Main kernel: `precompute_decode_metadata_kernel` (lines 68-146)
- Wrapper: `precompute_decode_metadata_cuda` (lines 151-294)

**Python Integration:**
- File: `python/sglang/srt/layers/attention/nsa/nsa_backend_mtp_precompute.py`
- Function: `_precompute_decode_mode` (lines 115-218)
- Fallback path included for compatibility

**Build Configuration:**
- CMakeLists.txt: Added to SOURCES list (line 283)
- Extension binding: `csrc/common_extension.cc` (lines 73-78)
- Header: `include/sgl_kernel_ops.h` (lines 151-163)

### Kernel Parameters

```cpp
void precompute_decode_metadata_cuda(
    at::Tensor seq_lens,              // [bs] - input sequence lengths
    at::Tensor req_pool_indices,      // [bs] - request pool indices
    at::Tensor req_to_token,          // [total_req, stride] - page table mapping
    at::Tensor cache_seqlens,         // [bs] - output: cache sequence lengths
    at::Tensor cu_seqlens_k,          // [bs+1] - output: cumulative seqlens
    at::Tensor page_indices,          // [bs, max_len] - output: gathered pages
    at::Tensor nsa_cache_seqlens,     // [bs] - output: NSA cache seqlens
    at::Tensor nsa_cu_seqlens_k,      // [bs+1] - output: NSA cumulative seqlens
    c10::optional<at::Tensor> real_page_table,  // [bs, cols] - output: transformed table
    int64_t max_len,                  // Maximum sequence length
    int64_t nsa_index_topk,           // NSA top-k parameter
    int64_t real_page_size            // Page size for transformation
);
```

### Workflow

The kernel executes the following steps in parallel:

1. **Load seq_lens** into shared memory (with dtype conversion if needed)
2. **Compute nsa_cache_seqlens** via clamp operation (shared memory)
3. **Compute cu_seqlens_k** using parallel prefix sum
4. **Compute nsa_cu_seqlens_k** using parallel prefix sum
5. **Write results** from shared memory to global memory
6. **Gather page_indices** from req_to_token in parallel
7. **Transform page table** (if real_page_size > 1) in parallel

### Thread Synchronization

- Uses `__syncthreads()` between computation phases
- Single block execution ensures all threads stay synchronized
- No inter-block communication needed

## Usage

### Automatic Usage

The kernel is automatically used when conditions are met:

```python
# In _precompute_decode_mode
try:
    from sgl_kernel import precompute_decode_metadata_cuda
    if bs <= 256:
        use_fused_kernel = True
except ImportError:
    use_fused_kernel = False  # Fallback to Python
```

### Compilation

Build sgl-kernel with the new kernel:

```bash
cd sgl-workspace/sglang/sgl-kernel
pip install -e .
```

### Verification

Check if the kernel is available:

```python
from sgl_kernel import precompute_decode_metadata_cuda
print("Kernel available!")
```

## Limitations

### Batch Size Limit
- Current implementation: MAX_SHARED_BS = 256
- Can be increased if more shared memory is available
- For larger batches, falls back to Python implementation

### Not Included in Kernel
- **FlashMLA metadata computation**: Still computed separately after kernel
- Reason: Complex operation requiring external kernel call
- Future: Could be integrated for further optimization

### Single Block Design
- Optimal for batch sizes ≤256
- For larger batches, multi-block design would be needed
- Trade-off: simplicity vs. scalability

## Future Improvements

1. **Dynamic Batch Size Handling**
   - Auto-detect available shared memory
   - Adjust MAX_SHARED_BS based on GPU capabilities

2. **Multi-Block Support**
   - Support batch sizes >256 with multiple blocks
   - Requires more complex synchronization

3. **Combined Precompute + Copy Kernel**
   - Fuse precomputation with metadata copying
   - Single kernel for entire CUDA graph replay path
   - Maximum performance for multi-step speculative decoding

4. **FlashMLA Integration**
   - Include FlashMLA metadata computation in kernel
   - Eliminate final separate kernel call

5. **Adaptive Launch Configuration**
   - Dynamically choose number of threads based on workload
   - Optimize occupancy for different batch sizes

## Testing

### Verification Steps

1. **Compilation**: Ensure kernel compiles without errors
2. **Functional test**: Verify results match Python implementation
3. **Performance test**: Measure kernel launch overhead reduction
4. **Integration test**: Test with multi-step speculative decoding

### Benchmark

Compare fused kernel vs. Python implementation:

```python
import torch
import time

# Benchmark code here
# TODO: Add detailed benchmark script
```

## Related Work

- **Fused Metadata Copy**: `csrc/elementwise/fused_metadata_copy.cu`
  - Optimizes the copying phase during CUDA graph replay
  - Complementary to this precomputation optimization

- **NSA Backend**: `python/sglang/srt/layers/attention/nsa_backend.py`
  - Uses the precomputed metadata
  - Benefits from reduced precomputation overhead

## References

- Original issue: Multiple kernel launches in precomputation phase
- Optimization target: Multi-step speculative decoding with CUDA graphs
- Related PR: Fused metadata copy optimization
