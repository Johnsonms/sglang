# Unified Decode Metadata - Direct Pointer Variant

## Overview

`unified_decode_metadata_cuda_direct` is an optimized variant of `unified_decode_metadata_cuda` specifically designed for small numbers of backends (≤3). Instead of passing pointers through a GPU tensor, it passes them directly as kernel arguments.

## Performance Benefits

### Before (unified_decode_metadata_cuda):
```python
# Create CPU tensor
backend_pointers = torch.empty((7, N), dtype=torch.int64, device='cpu')

# Fill with pointers (loop on CPU)
for i in range(N):
    metadata = backends[i].decode_cuda_graph_metadata[bs]
    backend_pointers[0, i] = metadata.cache_seqlens_int32.data_ptr()
    backend_pointers[1, i] = metadata.cu_seqlens_k.data_ptr()
    # ... 5 more assignments

# Transfer to GPU
backend_pointers = backend_pointers.to(device)  # 1 CPU->GPU transfer

# Call kernel
unified_decode_metadata_cuda(seq_lens, ..., backend_pointers, ...)
```

**Cost**: 1 tensor allocation + 1 CPU->GPU transfer

### After (unified_decode_metadata_cuda_direct):
```python
# Just pass the metadata objects directly - NO tensor operations!
metadata_list = [backend.decode_cuda_graph_metadata[bs] for backend in backends]

unified_decode_metadata_cuda_direct(
    seq_lens, req_pool_indices, req_to_token,
    metadata_list, nsa_index_topk, real_page_size)
```

**Cost**: 0 tensor operations, 0 GPU transfers

## Usage in nsa_backend.py

Replace the current implementation:

```python
# OLD CODE (lines 1962-2012 in nsa_backend.py)
backend_pointers = torch.empty((7, self.speculative_num_steps), dtype=torch.int64, device='cpu')

for i in range(self.speculative_num_steps):
    backend = self.attn_backends[i]
    backend.set_nsa_prefill_impl(forward_batch=None)
    metadata = backend.decode_cuda_graph_metadata[bs]

    backend_pointers[0, i] = metadata.cache_seqlens_int32.data_ptr()
    backend_pointers[1, i] = metadata.cu_seqlens_k.data_ptr()
    backend_pointers[2, i] = metadata.page_table_1.data_ptr()
    backend_pointers[3, i] = metadata.nsa_cache_seqlens_int32.data_ptr()
    backend_pointers[4, i] = metadata.nsa_cu_seqlens_k.data_ptr()
    backend_pointers[5, i] = metadata.real_page_table.data_ptr() if real_page_size > 1 else 0
    backend_pointers[6, i] = metadata.nsa_seqlens_expanded.data_ptr()

unified_decode_metadata_cuda(seq_lens, req_pool_indices, first_backend.req_to_token,
                            backend_pointers, max_len, ...)
```

With the new direct version:

```python
# NEW CODE (much simpler!)
from sgl_kernel import unified_decode_metadata_cuda_direct

# Prepare metadata list
metadata_list = []
for i in range(self.speculative_num_steps):
    backend = self.attn_backends[i]
    backend.set_nsa_prefill_impl(forward_batch=None)
    metadata_list.append(backend.decode_cuda_graph_metadata[bs])

# Single clean call - no tensor allocation!
unified_decode_metadata_cuda_direct(
    seq_lens,
    req_pool_indices,
    first_backend.req_to_token,
    metadata_list,
    first_backend.nsa_index_topk,
    first_backend.real_page_size
)
```

## API Comparison

### unified_decode_metadata_cuda (Original)
- **Max backends**: 8
- **Backend pointers**: Passed as int64 tensor [7, N] on GPU
- **Overhead**: 1 CPU tensor allocation + 1 CPU->GPU transfer
- **Use case**: When N > 3 or flexible backend count needed

### unified_decode_metadata_cuda_direct (New)
- **Max backends**: 3 (hardcoded for optimal performance)
- **Backend pointers**: Passed directly as 21 int64 kernel arguments
- **Overhead**: 0 tensor operations, 0 GPU transfers
- **Use case**: Small fixed backend count (typical for speculative decoding)

## Technical Details

### Kernel Arguments
The direct variant passes 21 pointer arguments directly:
- 7 pointers for backend 0
- 7 pointers for backend 1
- 7 pointers for backend 2
- Unused backends get 0 (nullptr)

### Memory Layout
Pointers are passed in kernel constant memory (no GPU RAM allocation needed):
```
Backend 0: [cache_seqlens, cu_seqlens_k, page_indices, nsa_cache_seqlens, nsa_cu_seqlens_k, real_page_table, seqlens_expanded]
Backend 1: [... same 7 pointers ...]
Backend 2: [... same 7 pointers ...]
```

### Limitations
- Maximum 3 backends (hardcoded for performance)
- If you need > 3 backends, use the original `unified_decode_metadata_cuda`

## Build Instructions

The direct variant is automatically compiled when building sgl-kernel:

```bash
cd /sgl-workspace/sglang/sgl-kernel
pip install -e . --force-reinstall
```

Files involved:
- `csrc/elementwise/unified_decode_metadata_direct.cu` - CUDA kernel implementation
- `include/sgl_kernel_ops.h` - C++ function declaration
- `csrc/common_extension.cc` - PyTorch op registration
- `python/sgl_kernel/attention.py` - Python wrapper
- `CMakeLists.txt` - Build configuration

## Performance Impact

Expected improvements for 3-backend speculative decoding:
- Latency reduction: ~5-10 microseconds per call (eliminates tensor operations)
- Memory pressure: Eliminates 168 bytes of CPU allocation + GPU transfer per call
- Code clarity: Simpler, more readable Python code

For CUDA graph replay at high frequency, these savings compound significantly.
