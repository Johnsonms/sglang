# EAGLE Speculative Decoding: Async Copy Optimization

## Problem Analysis

**Location**: `/sgl-workspace/sglang/python/sglang/srt/speculative/eagle_info.py:365-366`

```python
accept_index_cpu = accept_index.tolist()  # ⚠️ 28ms synchronous GPU->CPU copy!
predict_cpu = predict.tolist()
```

### Root Cause

1. **`accept_index`** and **`predict`** are GPU tensors from `verify_tree_greedy_func` (sgl_kernel)
2. **`.tolist()`** causes **synchronous** GPU→CPU transfer
3. Transfer goes through **Device → Pageable memory** (slowest path)
4. **Blocks CPU thread** until transfer completes

**Current latency**: ~28ms (unacceptable!)

### Why It's Slow

`.tolist()` performs:
1. **GPU sync**: Wait for all GPU work to complete
2. **D2H copy**: Copy from GPU memory to CPU pageable memory (slow!)
3. **Convert to Python list**: Create Python objects (additional overhead)

---

## Where The Data Is Used

Lines 371-401: CPU loop that processes accepted tokens:

```python
for i, (req, accept_index_row) in enumerate(zip(batch.reqs, accept_index_cpu)):
    for j, idx in enumerate(accept_index_row):
        if idx == -1:
            break
        id = predict_cpu[idx]  # CPU access
        req.output_ids.append(id)  # CPU operation
        req.check_finished()  # CPU operation
        if req.finished():  # CPU check
            has_finished = True
            accept_index[i, j + 1 :] = -1  # GPU write
            break
        else:
            if req.grammar is not None:
                req.grammar.accept_token(id)  # CPU operation
```

**Requirements**:
- Need CPU data for control flow (checking -1, checking finished)
- Need to update CPU state (req.output_ids, req.grammar)
- Need to write back to GPU tensor (accept_index[i, j+1:] = -1)

---

## Solution Options

### Option 1: Async Copy with Pinned Memory (Recommended) ⭐

Use CUDA pinned memory and async transfers to overlap GPU work with data transfer.

#### Implementation

```python
class AsyncCopyHelper:
    """Helper class to manage async GPU->CPU transfers with pinned memory."""

    def __init__(self, device='cuda'):
        self.device = device
        self.stream = torch.cuda.Stream()
        # Pre-allocate pinned memory buffers (reused across calls)
        self.pinned_accept_index = None
        self.pinned_predict = None

    def async_copy_to_cpu(
        self,
        accept_index: torch.Tensor,  # [bs, spec_steps+1]
        predict: torch.Tensor,  # [total_tokens]
    ):
        """Start async copy from GPU to pinned CPU memory.

        Returns:
            event: CUDA event to wait on before accessing CPU data
            pinned_accept_index: Pinned memory tensor (don't access until event completes!)
            pinned_predict: Pinned memory tensor (don't access until event completes!)
        """
        bs, spec_steps_plus_1 = accept_index.shape
        predict_size = predict.shape[0]

        # Allocate or reuse pinned memory buffers
        if (self.pinned_accept_index is None or
            self.pinned_accept_index.shape != accept_index.shape):
            # pin_memory=True creates pinned (page-locked) CPU memory
            self.pinned_accept_index = torch.empty(
                accept_index.shape,
                dtype=accept_index.dtype,
                pin_memory=True  # ⚡ Key optimization
            )

        if (self.pinned_predict is None or
            self.pinned_predict.shape[0] < predict_size):
            self.pinned_predict = torch.empty(
                predict_size,
                dtype=predict.dtype,
                pin_memory=True  # ⚡ Key optimization
            )

        # Start async copy on separate stream
        with torch.cuda.stream(self.stream):
            # These are non-blocking copies
            self.pinned_accept_index.copy_(accept_index, non_blocking=True)
            self.pinned_predict[:predict_size].copy_(predict, non_blocking=True)

            # Record event when copies complete
            event = torch.cuda.Event()
            event.record(self.stream)

        return event, self.pinned_accept_index, self.pinned_predict[:predict_size]
```

#### Updated verify() method

```python
def verify(self, batch, logits_output, token_to_kv_pool_allocator, page_size, vocab_mask=None):
    # ... existing code up to line 362 ...

    # Initialize async copy helper (reuse across calls)
    if not hasattr(self, '_async_copy_helper'):
        self._async_copy_helper = AsyncCopyHelper(device=batch.device)

    # START ASYNC COPY EARLY (while GPU might still be working)
    # This overlaps GPU->CPU transfer with any remaining GPU operations
    event, pinned_accept_index, pinned_predict = self._async_copy_helper.async_copy_to_cpu(
        accept_index, predict
    )

    # ... do other GPU work here if any ...

    # WAIT for async copy to complete before accessing CPU data
    event.synchronize()  # Only blocks here, not at .tolist()

    # Now safe to access CPU data (already in pinned memory)
    # Convert to lists (fast because already on CPU)
    accept_index_cpu = pinned_accept_index.tolist()
    predict_cpu = pinned_predict.tolist()

    # ... rest of the function unchanged ...
```

**Benefits**:
- ✅ **Non-blocking transfer**: D2H copy happens asynchronously
- ✅ **Pinned memory**: 2-3x faster than pageable memory
- ✅ **Reuse buffers**: No repeated allocation overhead
- ✅ **Overlap opportunities**: Can do other work while copying

**Expected improvement**: ~28ms → ~8-12ms (2-3x faster)

---

### Option 2: Pre-start Transfer Earlier

Start the async copy **immediately after verify_tree_greedy_func** returns, before other operations.

```python
# Line 288-298: After verify_tree_greedy_func
predict, accept_index, accept_length = verify_tree_greedy_func(...)

# 🚀 START ASYNC COPY IMMEDIATELY
event, pinned_accept_index, pinned_predict = self._async_copy_helper.async_copy_to_cpu(
    accept_index, predict
)

# Do other GPU work here (simulations, etc.)
if SIMULATE_ACC_LEN > 0.0:
    accept_index = generate_simulated_accept_index(...)

# ... more GPU work ...

# WAIT for copy to complete only when we need CPU data (line 365)
event.synchronize()
accept_index_cpu = pinned_accept_index.tolist()
predict_cpu = pinned_predict.tolist()
```

**Benefits**:
- ✅ Maximum overlap with GPU work
- ✅ Transfer happens while GPU is still busy
- ✅ Zero-cost if GPU work takes longer than transfer

**Expected improvement**: ~28ms → ~5-8ms (3-5x faster if overlapped well)

---

### Option 3: Keep More Logic on GPU (Advanced)

Move the finish-checking logic to a custom CUDA/Triton kernel.

**Challenges**:
- `req.output_ids` is a Python list (CPU-side)
- `req.check_finished()` is complex Python logic
- Grammar state is on CPU

**Partial solution**: Create GPU kernel for finish detection, then only copy finished indices:

```python
# GPU kernel to find first finished position per sequence
finished_positions = find_finished_positions_kernel(
    accept_index, predict, finish_token_ids
)

# Only copy finished_positions (small) to CPU
finished_positions_cpu = finished_positions.cpu()  # Much smaller transfer!

# Then update req state on CPU as before
```

**Expected improvement**: ~28ms → ~10-15ms (1.8-2.8x faster)

**Complexity**: High (requires restructuring request state management)

---

### Option 4: CPU-side Tensor (Avoid .tolist())

Keep as tensor on CPU instead of converting to Python list:

```python
# Use .cpu() instead of .tolist()
accept_index_cpu = accept_index.cpu()  # Returns CPU tensor
predict_cpu = predict.cpu()  # Returns CPU tensor

# Access via indexing instead of list iteration
for i, req in enumerate(batch.reqs):
    for j in range(accept_index_cpu.shape[1]):
        idx = accept_index_cpu[i, j].item()  # Faster than list access
        if idx == -1:
            break
        id = predict_cpu[idx].item()  # Faster than list access
        req.output_ids.append(id)
        # ... rest unchanged ...
```

**Benefits**:
- ✅ Simpler code change
- ✅ Avoids Python list creation overhead
- ✅ Can use async copy: `.cpu(non_blocking=True)` with pinned memory

**Expected improvement**: ~28ms → ~15-20ms (1.4-1.8x faster)

---

## Recommended Implementation Plan

### Phase 1: Quick Win (Option 4)

Replace `.tolist()` with `.cpu()` + tensor indexing:

```python
# Line 365-366: Replace this
accept_index_cpu = accept_index.tolist()
predict_cpu = predict.tolist()

# With this
accept_index_cpu = accept_index.cpu()  # CPU tensor, not list
predict_cpu = predict.cpu()  # CPU tensor, not list

# Line 371-401: Update iteration
for i, req in enumerate(batch.reqs):
    for j in range(accept_index_cpu.shape[1]):
        idx = accept_index_cpu[i, j].item()
        if idx == -1:
            break
        id = predict_cpu[idx].item()
        req.output_ids.append(id)
        # ... rest unchanged ...
```

**Effort**: Low (5-10 minutes)
**Expected**: 1.4-1.8x faster (~28ms → ~15-20ms)

### Phase 2: Pinned Memory + Async (Option 1)

Add `AsyncCopyHelper` class and use pinned memory:

```python
# Add AsyncCopyHelper class to eagle_info.py

# In __init__ or as class variable:
self._async_copy_helper = AsyncCopyHelper(device=batch.device)

# Line 298: Start async copy immediately after kernel
event, pinned_accept_index, pinned_predict = self._async_copy_helper.async_copy_to_cpu(
    accept_index, predict
)

# Do other work here...

# Line 365: Wait and convert
event.synchronize()
accept_index_cpu = pinned_accept_index.cpu()  # Already on CPU, just reference
predict_cpu = pinned_predict.cpu()
```

**Effort**: Medium (30-60 minutes)
**Expected**: 2.5-3.5x faster (~28ms → ~8-10ms)

### Phase 3: Early Start + Overlap (Option 2)

Move async copy to line 298 (right after verify_tree_greedy_func):

**Effort**: Low (additional 10 minutes on top of Phase 2)
**Expected**: 3-5x faster (~28ms → ~5-8ms)

---

## Performance Projections

| Method | Current | Phase 1 | Phase 2 | Phase 3 |
|--------|---------|---------|---------|---------|
| **Latency** | 28 ms | ~15-20 ms | ~8-10 ms | ~5-8 ms |
| **Speedup** | 1.0x | 1.4-1.8x | 2.5-3.5x | 3.5-5.6x |
| **Effort** | - | Low | Medium | Low |

---

## Implementation Details

### Pinned Memory Benefits

Normal memory (pageable):
- GPU → CPU: ~10-15 GB/s (slow)
- Requires 2 copies: GPU → staging buffer → CPU

Pinned memory (page-locked):
- GPU → CPU: ~25-30 GB/s (fast)
- Direct DMA: GPU → CPU (single copy)
- Can use async transfers

### Async Transfer Mechanics

```python
# Without async (current):
data = gpu_tensor.tolist()  # Blocks for ~28ms
process(data)

# With async:
event, pinned_cpu = start_async_copy(gpu_tensor)  # Returns immediately
do_other_work()  # Overlaps with copy
event.synchronize()  # Blocks only if copy not done
process(pinned_cpu)  # Fast, already on CPU
```

---

## Testing Plan

1. **Correctness**: Verify output_ids are identical
2. **Performance**: Measure latency with different phases
3. **Memory**: Check pinned memory usage (should be small)
4. **Concurrency**: Test with multiple concurrent requests

---

## Code Location

**File**: `/sgl-workspace/sglang/python/sglang/srt/speculative/eagle_info.py`
**Critical lines**:
- Line 288-298: verify_tree_greedy_func returns
- Line 365-366: `.tolist()` (28ms bottleneck)
- Line 371-401: CPU loop using the data

---

## Recommendation

**Start with Phase 1** (Quick win):
- Replace `.tolist()` with `.cpu()` + tensor indexing
- Expected: 1.4-1.8x faster with minimal code change
- Test and measure actual improvement

**Then implement Phase 2+3** if needed:
- Add pinned memory + async copy
- Expected: 3-5x faster total
- More complex but much better performance

---

**Priority**: HIGH - 28ms is a significant bottleneck in speculative decoding
**Risk**: LOW - Changes are localized and testable
**Impact**: HIGH - 3-5x faster verification step improves overall throughput significantly
