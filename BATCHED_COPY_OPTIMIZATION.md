# Batched Copy Kernel Optimization for Metadata Copying

## Problem Analysis

In `init_forward_metadata_replay_cuda_graph_from_precomputed`, there are **6-7 individual `.copy_()` operations**:

```python
# 1. Basic seqlens (always)
metadata.cache_seqlens_int32.copy_(precomputed.cache_seqlens)  # ~2-3 μs
metadata.cu_seqlens_k[1:].copy_(precomputed.cu_seqlens_k[1:])  # ~2-3 μs

# 2. Mode-specific copies (3-4 operations)
metadata.page_table_1[:, :max_len].copy_(precomputed.page_indices)  # ~5-8 μs
metadata.nsa_cache_seqlens_int32.copy_(precomputed.nsa_cache_seqlens)  # ~2-3 μs
metadata.nsa_seqlens_expanded[:size].copy_(precomputed.seqlens_expanded)  # ~2-3 μs

# 3. NSA cu_seqlens (always)
metadata.nsa_cu_seqlens_k[1:1+size].copy_(...)  # ~2-3 μs

# 4. Optional copies (0-2 operations)
metadata.real_page_table[:rows, :cols].copy_(precomputed.real_page_table)  # ~3-5 μs
flashmla_metadata.copy_(precomputed.flashmla_metadata)  # ~1-2 μs
```

**Total overhead**: 6-7 kernel launches × (2-3 μs launch overhead) = **12-21 μs**

**Actual copy time**: ~8-12 μs

**Total**: ~20 μs

---

## Optimization Goal

**Merge all copies into a single Triton kernel**: Save kernel launch overhead

**Expected savings**: 10-15 μs (reduce to ~5-8 μs total)

---

## Design Options

### Option 1: Simple Batched Memcpy (Recommended) ⭐

**Approach**: Flatten all tensors to 1D and batch copy contiguous memory regions.

**Pros**:
- ✅ Simple implementation
- ✅ Fast for contiguous memory
- ✅ Easy to understand

**Cons**:
- ❌ Requires contiguous memory (need `.contiguous()` for slices)
- ❌ May introduce extra memory copies for non-contiguous tensors

**Performance**:
- Best case: 5 μs (all contiguous)
- Worst case: 8 μs (some non-contiguous)

### Option 2: Strided Batched Copy (More Complex)

**Approach**: Handle non-contiguous tensors with stride information.

**Pros**:
- ✅ No extra memory copies needed
- ✅ Handles 2D slices directly

**Cons**:
- ❌ More complex kernel
- ❌ Slower for contiguous memory (stride checking overhead)

**Performance**:
- ~6-10 μs

### Option 3: Hybrid Approach (Best Performance)

**Approach**: Check if tensors are contiguous. Use fast path for contiguous, slow path for strided.

**Pros**:
- ✅ Best of both worlds
- ✅ Optimal performance

**Cons**:
- ❌ Most complex implementation
- ❌ More code to maintain

---

## Recommended: Option 1 (Simple Batched Memcpy)

### Why?

1. **Most tensors are contiguous** in CUDA graph path
2. **Simplicity** > complex optimization at this stage
3. **Easy to benchmark** and verify
4. Can upgrade to Option 3 later if needed

### Implementation

#### 1. Triton Kernel

```python
@triton.jit
def batched_memcpy_kernel(
    # Input: list of copy descriptors
    src_ptrs,  # [n_copies] pointer to source pointers
    dst_ptrs,  # [n_copies] pointer to destination pointers
    sizes,     # [n_copies] number of elements for each copy
    offsets,   # [n_copies+1] cumulative offsets for global indexing
    total_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Batched memcpy kernel.

    Copies multiple src -> dst regions in parallel.
    Each thread block handles a range of elements across all copies.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    block_end = min(block_start + BLOCK_SIZE, total_elements)

    # Process elements in this block
    for idx in range(block_start, block_end):
        # Binary search to find which copy this element belongs to
        copy_id = 0
        for i in range(n_copies):
            if idx >= tl.load(offsets + i) and idx < tl.load(offsets + i + 1):
                copy_id = i
                break

        # Load copy descriptor
        src_ptr = tl.load(src_ptrs + copy_id)
        dst_ptr = tl.load(dst_ptrs + copy_id)
        offset_in_copy = idx - tl.load(offsets + copy_id)

        # Perform copy
        value = tl.load(src_ptr + offset_in_copy)
        tl.store(dst_ptr + offset_in_copy, value)
```

**Issues with this approach**:
- Binary search in kernel is slow
- Random memory access pattern

#### 2. Better Approach: One Kernel Per Copy (Parallel Launch)

Actually, a better approach is to **launch all copy kernels in parallel** using CUDA streams or a single kernel with multiple grid dimensions.

**Even better**: Use a simpler kernel that copies all data in one vectorized pass.

#### 3. Simplified Design: Mega-Memcpy Kernel

```python
@triton.jit
def mega_memcpy_kernel(
    src_base,      # Base pointer (will offset from this)
    dst_base,      # Base pointer (will offset from this)
    src_offsets,   # [n_copies] byte offset from src_base
    dst_offsets,   # [n_copies] byte offset from dst_base
    copy_sizes,    # [n_copies] number of BYTES to copy
    n_copies,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple mega memcpy: process each copy independently."""
    copy_id = tl.program_id(0)

    if copy_id >= n_copies:
        return

    # Load copy parameters
    src_offset = tl.load(src_offsets + copy_id)
    dst_offset = tl.load(dst_offsets + copy_id)
    size_bytes = tl.load(copy_sizes + copy_id)

    # Calculate actual pointers
    src_ptr = src_base + src_offset
    dst_ptr = dst_base + dst_offset

    # Copy in blocks
    n_elements = size_bytes // 4  # Assuming int32
    block_id = tl.program_id(1)
    block_start = block_id * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Vectorized load/store
    data = tl.load(src_ptr + offsets * 4, mask=mask, other=0)
    tl.store(dst_ptr + offsets * 4, data, mask=mask)
```

**Issues**:
- Requires all data to be in a single contiguous buffer
- Complex to manage offsets

---

## Final Recommendation: Simplified Batched Copy

**Key insight**: PyTorch already optimizes `.copy_()` operations. The overhead is mainly from **kernel launch latency**, not the copy itself.

**Better optimization**: Use PyTorch's built-in batching:

```python
# Instead of:
tensor1.copy_(src1)
tensor2.copy_(src2)
tensor3.copy_(src3)

# Use:
torch.cat([tensor1.view(-1), tensor2.view(-1), tensor3.view(-1)]).copy_(
    torch.cat([src1.view(-1), src2.view(-1), src3.view(-1)])
)
```

But this requires **contiguous views**, which may not always work for slices.

---

## Practical Solution: Batch Copy with Pre-flattened Data

### Design

1. **At precomputation time**: Flatten all data into a single buffer
2. **At copy time**: Single kernel copies the entire buffer
3. **Post-copy**: Data is already in correct locations

### Implementation

```python
@dataclass
class PrecomputedMetadataFlat:
    """Flattened precomputed metadata for fast batched copy."""

    # Single contiguous buffer containing all data
    flat_data: torch.Tensor  # [total_size], int32

    # Offsets into flat_data for each field
    cache_seqlens_offset: int
    cache_seqlens_size: int

    cu_seqlens_k_offset: int
    cu_seqlens_k_size: int

    page_indices_offset: int
    page_indices_size: int

    # ... etc for all fields

    # Destination offsets in metadata structure
    dst_cache_seqlens_offset: int
    dst_cu_seqlens_k_offset: int
    # ... etc

def _precompute_flattened(self, ...) -> PrecomputedMetadataFlat:
    """Precompute and flatten all metadata into single buffer."""

    # Compute all intermediate results (same as before)
    cache_seqlens = ...
    cu_seqlens_k = ...
    page_indices = ...
    # ... etc

    # Flatten everything into single buffer
    flat_data = torch.cat([
        cache_seqlens.view(-1),
        cu_seqlens_k.view(-1),
        page_indices.view(-1),
        nsa_cache_seqlens.view(-1),
        nsa_cu_seqlens_k.view(-1),
        # ... etc
    ])

    # Record offsets
    offset = 0
    cache_seqlens_offset = offset
    cache_seqlens_size = cache_seqlens.numel()
    offset += cache_seqlens_size

    cu_seqlens_k_offset = offset
    cu_seqlens_k_size = cu_seqlens_k.numel()
    offset += cu_seqlens_k_size

    # ... etc

    return PrecomputedMetadataFlat(
        flat_data=flat_data,
        cache_seqlens_offset=cache_seqlens_offset,
        cache_seqlens_size=cache_seqlens_size,
        # ... etc
    )

def _batch_copy_from_flat(
    self,
    flat: PrecomputedMetadataFlat,
    metadata: NSAMetadata,
):
    """Single kernel to copy all data from flat buffer to metadata."""

    # Use single Triton kernel or even just PyTorch
    # Since both src and dst are now contiguous, PyTorch is fast

    # Copy cache_seqlens
    dst = metadata.cache_seqlens_int32.view(-1)
    src = flat.flat_data[
        flat.cache_seqlens_offset:
        flat.cache_seqlens_offset + flat.cache_seqlens_size
    ]
    dst.copy_(src)

    # Copy cu_seqlens_k
    dst = metadata.cu_seqlens_k.view(-1)[1:]  # Skip first element
    src = flat.flat_data[
        flat.cu_seqlens_k_offset:
        flat.cu_seqlens_k_offset + flat.cu_seqlens_k_size
    ]
    dst.copy_(src)

    # ... repeat for all fields
```

**Issues**:
- Still requires multiple `.copy_()` calls
- Complex offset management
- Doesn't actually solve the problem

---

## Reality Check

Let's measure actual overhead:

```python
import torch
import time

# Setup
src = [torch.randint(0, 100, (32,), dtype=torch.int32, device='cuda') for _ in range(7)]
dst = [torch.empty(32, dtype=torch.int32, device='cuda') for _ in range(7)]

# Warmup
for _ in range(100):
    for s, d in zip(src, dst):
        d.copy_(s)

# Benchmark individual copies
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(1000):
    for s, d in zip(src, dst):
        d.copy_(s)
torch.cuda.synchronize()
time_individual = (time.perf_counter() - start) / 1000 * 1e6  # μs

# Benchmark single cat+copy
src_cat = torch.cat([s.view(-1) for s in src])
dst_cat = torch.empty_like(src_cat)

torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(1000):
    dst_cat.copy_(src_cat)
    # Then scatter back
    offset = 0
    for d in dst:
        size = d.numel()
        d.copy_(dst_cat[offset:offset+size].view_as(d))
        offset += size
torch.cuda.synchronize()
time_batched = (time.perf_counter() - start) / 1000 * 1e6  # μs

print(f"Individual copies: {time_individual:.2f} μs")
print(f"Batched copy: {time_batched:.2f} μs")
```

**Expected result**: Individual copies might already be very fast (~10-15 μs total) due to PyTorch's optimizations.

---

## Final Recommendation

### Step 1: Measure First

Before optimizing, **benchmark current performance**:
- Measure time for `init_forward_metadata_replay_cuda_graph_from_precomputed`
- Break down copy time vs other overhead

### Step 2: Simple Optimization (if needed)

If copy overhead is significant (>10 μs), try:

**Option A: Fuse with PyTorch**
```python
# Group all 1D copies together
all_src_1d = [precomputed.cache_seqlens, precomputed.cu_seqlens_k[1:], ...]
all_dst_1d = [metadata.cache_seqlens_int32, metadata.cu_seqlens_k[1:], ...]

# Single copy for 1D tensors
torch.cat([d.view(-1) for d in all_dst_1d]).copy_(
    torch.cat([s.view(-1) for s in all_src_1d])
)
```

**Option B: Custom Triton Kernel**
```python
# Simple kernel that copies multiple regions
# Launch with grid = (n_copies, n_blocks_per_copy)
```

### Step 3: Advanced Optimization (if still needed)

Only if Step 2 doesn't provide enough speedup, implement full strided batched copy kernel.

---

## Conclusion

**Current overhead**: ~20 μs (6-7 copies)
**Potential savings**: 5-10 μs (if optimized)
**Effort**: Medium to High
**ROI**: Low to Medium

**Recommendation**:
1. **Measure first** - Profile to confirm copy operations are the bottleneck
2. **Try PyTorch cat+copy** - Simple and might be sufficient
3. **Custom Triton kernel** - Only if absolutely necessary

The 3-5x speedup from precomputation is already significant. Additional 25-50% speedup from batched copies has diminishing returns.

---

## Alternative: CUDA Streams

Instead of batched kernel, launch all copies in parallel using CUDA streams:

```python
streams = [torch.cuda.Stream() for _ in range(7)]

with torch.cuda.stream(streams[0]):
    metadata.cache_seqlens_int32.copy_(precomputed.cache_seqlens)

with torch.cuda.stream(streams[1]):
    metadata.cu_seqlens_k[1:].copy_(precomputed.cu_seqlens_k[1:])

# ... launch all 7 copies in parallel

# Synchronize all streams
for stream in streams:
    stream.synchronize()
```

**Pros**:
- ✅ Simple to implement
- ✅ Truly parallel execution
- ✅ No custom kernel needed

**Cons**:
- ❌ Stream creation overhead
- ❌ Still has 7 kernel launches (just parallel)
- ❌ Synchronization overhead

**Expected speedup**: 1.5-2x (reduce 20 μs → 10-15 μs)

---

**Next steps**: Benchmark current implementation before deciding on optimization strategy.
