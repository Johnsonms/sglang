"""Triton batched copy kernel for NSA metadata copying.

This kernel merges multiple .copy_() operations into a single kernel launch,
reducing kernel launch overhead from ~12-21μs to ~5-8μs.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def batched_copy_kernel_single(
    # Single copy operation
    src_ptr,  # Source pointer
    dst_ptr,  # Destination pointer
    copy_size,  # Number of elements (int32) to copy
    BLOCK_SIZE: tl.constexpr,
):
    """Simple copy kernel for a single copy operation.

    This kernel is launched multiple times (once per copy) to avoid
    pointer-to-pointer indirection issues in Triton.

    Grid: (n_blocks,)
    """
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < copy_size

    # Vectorized load and store
    data = tl.load(src_ptr + offsets, mask=mask, other=0)
    tl.store(dst_ptr + offsets, data, mask=mask)


@triton.jit
def batched_copy_kernel_unified(
    # All copy operations as direct parameters (max 8 copies)
    src0, dst0, size0,
    src1, dst1, size1,
    src2, dst2, size2,
    src3, dst3, size3,
    src4, dst4, size4,
    src5, dst5, size5,
    src6, dst6, size6,
    src7, dst7, size7,
    n_copies: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Unified batched copy kernel - single launch for all copies.

    This kernel uses a 2D grid:
    - Grid dim 0 (copy_id): Which copy operation (0 to n_copies-1)
    - Grid dim 1 (block_id): Which block within that copy

    By launching once with all copies, we minimize CPU launch overhead.

    Grid: (n_copies, max_blocks_needed)
    """
    copy_id = tl.program_id(0)
    block_id = tl.program_id(1)

    # Early exit for invalid copy_id
    if copy_id >= n_copies:
        return

    # Select the appropriate src/dst/size based on copy_id
    # Initialize to first copy, then override based on copy_id
    src_ptr = src0
    dst_ptr = dst0
    copy_size = size0

    if copy_id == 1:
        src_ptr = src1
        dst_ptr = dst1
        copy_size = size1
    if copy_id == 2:
        src_ptr = src2
        dst_ptr = dst2
        copy_size = size2
    if copy_id == 3:
        src_ptr = src3
        dst_ptr = dst3
        copy_size = size3
    if copy_id == 4:
        src_ptr = src4
        dst_ptr = dst4
        copy_size = size4
    if copy_id == 5:
        src_ptr = src5
        dst_ptr = dst5
        copy_size = size5
    if copy_id == 6:
        src_ptr = src6
        dst_ptr = dst6
        copy_size = size6
    if copy_id == 7:
        src_ptr = src7
        dst_ptr = dst7
        copy_size = size7

    # Calculate offsets for this block
    block_start = block_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < copy_size

    # Vectorized load and store
    data = tl.load(src_ptr + offsets, mask=mask, other=0)
    tl.store(dst_ptr + offsets, data, mask=mask)


@triton.jit
def batched_copy_kernel_simple(
    # Arrays of copy descriptors (passed as values, not pointers)
    src_base,  # Base address for all sources
    dst_base,  # Base address for all destinations
    src_offsets,  # [n_copies] byte offsets from src_base
    dst_offsets,  # [n_copies] byte offsets from dst_base
    copy_sizes,  # [n_copies] number of elements (int32)
    n_copies: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Simplified batched copy kernel using offset-based addressing.

    This version assumes all source and destination tensors are part of
    contiguous buffers, which simplifies pointer management.

    Grid: (n_copies, n_blocks_per_copy)
    """
    copy_id = tl.program_id(0)
    block_id = tl.program_id(1)

    if copy_id >= n_copies:
        return

    # Load copy parameters
    src_offset = tl.load(src_offsets + copy_id)
    dst_offset = tl.load(dst_offsets + copy_id)
    copy_size = tl.load(copy_sizes + copy_id)

    # Calculate pointers for this copy
    src_ptr = src_base + src_offset
    dst_ptr = dst_base + dst_offset

    # Process block
    block_start = block_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < copy_size

    # Load and store (assuming int32)
    src = src_ptr + offsets * 4
    dst = dst_ptr + offsets * 4
    data = tl.load(src, mask=mask, other=0)
    tl.store(dst, data, mask=mask)


class BatchedCopyDescriptor:
    """Descriptor for a single copy operation."""

    def __init__(self, src: torch.Tensor, dst: torch.Tensor):
        """Create a copy descriptor.

        Args:
            src: Source tensor (must be contiguous)
            dst: Destination tensor (must be contiguous, same shape as src)
        """
        # Verify tensors are contiguous
        if not src.is_contiguous():
            raise ValueError(f"Source tensor must be contiguous. Shape: {src.shape}, Strides: {src.stride()}")
        if not dst.is_contiguous():
            raise ValueError(f"Destination tensor must be contiguous. Shape: {dst.shape}, Strides: {dst.stride()}")

        assert src.shape == dst.shape, f"Shape mismatch: {src.shape} vs {dst.shape}"
        assert src.dtype == dst.dtype, f"Dtype mismatch: {src.dtype} vs {dst.dtype}"

        self.src = src.view(-1)
        self.dst = dst.view(-1)
        self.size = self.src.numel()
        self.src_ptr = src.data_ptr()
        self.dst_ptr = dst.data_ptr()


def batched_copy(descriptors: list[BatchedCopyDescriptor], block_size: int = 1024):
    """Perform multiple copy operations using Triton kernels.

    This launches one kernel per copy operation, but all launches are
    asynchronous and can execute in parallel on the GPU. This is faster
    than individual PyTorch .copy_() calls due to reduced overhead.

    Args:
        descriptors: List of copy descriptors
        block_size: Number of elements to process per block

    Returns:
        None (copies are performed in-place)
    """
    if not descriptors:
        return

    # Launch one kernel per copy (all async, can run in parallel)
    for desc in descriptors:
        n_blocks = (desc.size + block_size - 1) // block_size
        grid = (n_blocks,)

        batched_copy_kernel_single[grid](
            desc.src,
            desc.dst,
            desc.size,
            BLOCK_SIZE=block_size,
        )


def batched_copy_unified(descriptors: list[BatchedCopyDescriptor], block_size: int = 1024):
    """Perform multiple copy operations using a SINGLE Triton kernel launch.

    This reduces CPU launch overhead by batching all copies into one kernel.
    Uses a 2D grid where dim 0 = copy_id, dim 1 = block_id.

    Args:
        descriptors: List of copy descriptors (max 8)
        block_size: Number of elements to process per block

    Returns:
        None (copies are performed in-place)
    """
    if not descriptors:
        return

    if len(descriptors) > 8:
        raise ValueError(f"batched_copy_unified supports max 8 copies, got {len(descriptors)}")

    n_copies = len(descriptors)
    device = descriptors[0].src.device

    # Pad descriptors to 8 (use dummy tensors for unused slots)
    dummy_tensor = torch.zeros(1, dtype=torch.int32, device=device)
    padded = descriptors + [BatchedCopyDescriptor(dummy_tensor, dummy_tensor)] * (8 - n_copies)

    # Calculate max blocks needed
    max_size = max(d.size for d in descriptors)
    max_blocks = (max_size + block_size - 1) // block_size

    # Single kernel launch with 2D grid
    grid = (n_copies, max_blocks)

    batched_copy_kernel_unified[grid](
        # Copy 0
        padded[0].src, padded[0].dst, padded[0].size,
        # Copy 1
        padded[1].src, padded[1].dst, padded[1].size,
        # Copy 2
        padded[2].src, padded[2].dst, padded[2].size,
        # Copy 3
        padded[3].src, padded[3].dst, padded[3].size,
        # Copy 4
        padded[4].src, padded[4].dst, padded[4].size,
        # Copy 5
        padded[5].src, padded[5].dst, padded[5].size,
        # Copy 6
        padded[6].src, padded[6].dst, padded[6].size,
        # Copy 7
        padded[7].src, padded[7].dst, padded[7].size,
        n_copies=n_copies,
        BLOCK_SIZE=block_size,
    )


def batched_copy_metadata(
    metadata,  # Destination metadata
    precomputed,  # Source precomputed metadata
    forward_mode,  # ForwardMode enum
    max_len: int = None,  # For decode mode
    rows: int = None,  # For draft extend mode
    cols: int = None,  # For draft extend mode
    size: int = None,  # NSA seqlens expanded size
):
    """Batch copy all metadata using a single Triton kernel.

    This replaces 6-8 individual .copy_() operations with a single kernel launch,
    reducing CPU launch overhead. Non-contiguous tensors fall back to PyTorch .copy_().

    Args:
        metadata: Destination metadata object
        precomputed: Source PrecomputedMetadata object
        forward_mode: Forward mode (decode, target_verify, draft_extend)
        max_len: Maximum length (for decode mode page_table)
        rows: Number of rows (for draft extend mode)
        cols: Number of columns (for draft extend mode)
        size: NSA seqlens expanded size

    Returns:
        None (copies performed in-place)
    """
    descriptors = []
    fallback_copies = []  # List of (src, dst) for non-contiguous tensors

    def add_copy(src, dst):
        """Add copy to batched kernel or fallback list based on contiguity."""
        # Check if these are tensors (not custom objects like NSAFlashMLAMetadata)
        if not isinstance(src, torch.Tensor) or not isinstance(dst, torch.Tensor):
            fallback_copies.append((src, dst))
            return

        # Check contiguity for tensors
        if src.is_contiguous() and dst.is_contiguous():
            descriptors.append(BatchedCopyDescriptor(src=src, dst=dst))
        else:
            fallback_copies.append((src, dst))

    # 1. Copy basic seqlens (always)
    add_copy(precomputed.cache_seqlens, metadata.cache_seqlens_int32)

    # 2. Copy cu_seqlens_k (always, skip first element)
    add_copy(precomputed.cu_seqlens_k[1:], metadata.cu_seqlens_k[1:])

    # 3. Mode-specific copies
    if forward_mode.is_decode_or_idle():
        # Decode mode
        if max_len is not None:
            add_copy(precomputed.page_indices, metadata.page_table_1[:, :max_len])

        add_copy(precomputed.nsa_cache_seqlens, metadata.nsa_cache_seqlens_int32)

    elif forward_mode.is_target_verify():
        # Target verify mode
        add_copy(precomputed.page_indices, metadata.page_table_1[:, :precomputed.max_seqlen_k])
        add_copy(precomputed.seqlens_expanded, metadata.nsa_seqlens_expanded)
        add_copy(precomputed.nsa_cache_seqlens, metadata.nsa_cache_seqlens_int32)

    elif forward_mode.is_draft_extend():
        # Draft extend mode
        if rows is not None and cols is not None:
            add_copy(precomputed.page_indices, metadata.page_table_1[:rows, :cols])

        if size is not None:
            add_copy(precomputed.seqlens_expanded, metadata.nsa_seqlens_expanded[:size])
            add_copy(precomputed.nsa_cache_seqlens, metadata.nsa_cache_seqlens_int32[:size])

    # 4. Copy NSA cu_seqlens (always)
    if size is not None:
        add_copy(precomputed.nsa_cu_seqlens_k[1:1+size], metadata.nsa_cu_seqlens_k[1:1+size])

    # 5. Copy real page table (optional)
    if precomputed.real_page_table is not None:
        rows_real, cols_real = precomputed.real_page_table.shape
        add_copy(precomputed.real_page_table, metadata.real_page_table[:rows_real, :cols_real])

    # 6. Copy FlashMLA metadata (optional)
    if precomputed.flashmla_metadata is not None and size is not None:
        flashmla_metadata = metadata.flashmla_metadata.slice(slice(0, size + 1))
        add_copy(precomputed.flashmla_metadata, flashmla_metadata)

    # Perform batched copy using unified kernel (single launch for all contiguous copies)
    if descriptors:
        batched_copy_unified(descriptors, block_size=1024)

    # Fallback to PyTorch .copy_() for non-contiguous tensors
    # This is rare (typically only 2D slices like page_table_1[:rows, :cols])
    for src, dst in fallback_copies:
        dst.copy_(src)


# Benchmark function
def benchmark_batched_copy():
    """Benchmark batched copy vs individual copies."""
    import time

    device = torch.device("cuda")

    # Create test data (similar to real metadata)
    n_copies = 7
    sizes = [32, 64, 128, 256, 512, 1024, 2048]

    src_tensors = [
        torch.randint(0, 100, (size,), dtype=torch.int32, device=device)
        for size in sizes
    ]
    dst_tensors = [
        torch.empty(size, dtype=torch.int32, device=device)
        for size in sizes
    ]

    # Warmup
    for _ in range(100):
        for s, d in zip(src_tensors, dst_tensors):
            d.copy_(s)

    # Benchmark individual copies
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(1000):
        for s, d in zip(src_tensors, dst_tensors):
            d.copy_(s)
    torch.cuda.synchronize()
    time_individual = (time.perf_counter() - start) / 1000 * 1e6  # μs

    # Create descriptors
    descriptors = [
        BatchedCopyDescriptor(src=s, dst=d)
        for s, d in zip(src_tensors, dst_tensors)
    ]

    # Warmup batched (multiple launches)
    for _ in range(100):
        batched_copy(descriptors)

    # Benchmark batched copy (multiple launches)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(1000):
        batched_copy(descriptors)
    torch.cuda.synchronize()
    time_batched = (time.perf_counter() - start) / 1000 * 1e6  # μs

    # Warmup unified (single launch)
    for _ in range(100):
        batched_copy_unified(descriptors)

    # Benchmark unified batched copy (single launch)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(1000):
        batched_copy_unified(descriptors)
    torch.cuda.synchronize()
    time_unified = (time.perf_counter() - start) / 1000 * 1e6  # μs

    print(f"Individual copies (PyTorch):  {time_individual:.2f} μs")
    print(f"Batched copy (7 launches):    {time_batched:.2f} μs")
    print(f"Unified copy (1 launch):      {time_unified:.2f} μs")
    print(f"")
    print(f"Speedup (unified vs PyTorch): {time_individual / time_unified:.2f}x")
    print(f"Saved (unified vs PyTorch):   {time_individual - time_unified:.2f} μs")
    print(f"")
    print(f"CPU launch overhead saved:    ~{(7 - 1) * 2:.0f} μs (6 fewer launches @ ~2μs each)")


# Module-level cache for dummy tensors (one per device)
# Avoids recreating the same dummy tensor on every function call
_DUMMY_TENSOR_CACHE = {}


def _get_dummy_tuple(device):
    """Get or create a cached dummy tuple for padding descriptors.

    This avoids allocating a new tensor on every function call.
    Thread-safe because we're only reading after first write.
    """
    if device not in _DUMMY_TENSOR_CACHE:
        dummy = torch.zeros(1, dtype=torch.int32, device=device)
        _DUMMY_TENSOR_CACHE[device] = (dummy, dummy, dummy, dummy)
    return _DUMMY_TENSOR_CACHE[device]


@triton.jit
def batched_copy_to_3_backends_kernel(
    # Source tensors (single precomputed metadata)
    src0, src1, src2, src3, src4, src5, src6, src7,
    # Sizes for each source
    size0, size1, size2, size3, size4, size5, size6, size7,
    # Destination tensors (3 backends × 8 max copies)
    # Backend 0
    dst0_b0, dst1_b0, dst2_b0, dst3_b0, dst4_b0, dst5_b0, dst6_b0, dst7_b0,
    # Backend 1
    dst0_b1, dst1_b1, dst2_b1, dst3_b1, dst4_b1, dst5_b1, dst6_b1, dst7_b1,
    # Backend 2
    dst0_b2, dst1_b2, dst2_b2, dst3_b2, dst4_b2, dst5_b2, dst6_b2, dst7_b2,
    n_copies: tl.constexpr,  # Number of copy operations per backend
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for copying from 1 source to 3 backends.

    Grid: (3 * n_copies, max_blocks)
    - First dimension: backend_id * n_copies + copy_id
    - Second dimension: block_id within the copy
    """
    # Decode which backend and which copy operation
    combined_id = tl.program_id(0)
    backend_id = combined_id // n_copies
    copy_id = combined_id % n_copies
    block_id = tl.program_id(1)

    # Early exit for invalid IDs
    if backend_id >= 3 or copy_id >= n_copies:
        return

    # Select source based on copy_id
    src_ptr = src0
    copy_size = size0
    if copy_id == 1:
        src_ptr = src1
        copy_size = size1
    if copy_id == 2:
        src_ptr = src2
        copy_size = size2
    if copy_id == 3:
        src_ptr = src3
        copy_size = size3
    if copy_id == 4:
        src_ptr = src4
        copy_size = size4
    if copy_id == 5:
        src_ptr = src5
        copy_size = size5
    if copy_id == 6:
        src_ptr = src6
        copy_size = size6
    if copy_id == 7:
        src_ptr = src7
        copy_size = size7

    # Select destination based on backend_id and copy_id
    dst_ptr = dst0_b0
    if backend_id == 0:
        if copy_id == 0:
            dst_ptr = dst0_b0
        if copy_id == 1:
            dst_ptr = dst1_b0
        if copy_id == 2:
            dst_ptr = dst2_b0
        if copy_id == 3:
            dst_ptr = dst3_b0
        if copy_id == 4:
            dst_ptr = dst4_b0
        if copy_id == 5:
            dst_ptr = dst5_b0
        if copy_id == 6:
            dst_ptr = dst6_b0
        if copy_id == 7:
            dst_ptr = dst7_b0
    elif backend_id == 1:
        if copy_id == 0:
            dst_ptr = dst0_b1
        if copy_id == 1:
            dst_ptr = dst1_b1
        if copy_id == 2:
            dst_ptr = dst2_b1
        if copy_id == 3:
            dst_ptr = dst3_b1
        if copy_id == 4:
            dst_ptr = dst4_b1
        if copy_id == 5:
            dst_ptr = dst5_b1
        if copy_id == 6:
            dst_ptr = dst6_b1
        if copy_id == 7:
            dst_ptr = dst7_b1
    elif backend_id == 2:
        if copy_id == 0:
            dst_ptr = dst0_b2
        if copy_id == 1:
            dst_ptr = dst1_b2
        if copy_id == 2:
            dst_ptr = dst2_b2
        if copy_id == 3:
            dst_ptr = dst3_b2
        if copy_id == 4:
            dst_ptr = dst4_b2
        if copy_id == 5:
            dst_ptr = dst5_b2
        if copy_id == 6:
            dst_ptr = dst6_b2
        if copy_id == 7:
            dst_ptr = dst7_b2

    # Perform the copy
    block_start = block_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < copy_size

    data = tl.load(src_ptr + offsets, mask=mask, other=0)
    tl.store(dst_ptr + offsets, data, mask=mask)


def batched_copy_decode_to_3_backends(
    precomputed,
    metadata_b0,
    metadata_b1,
    metadata_b2,
    block_size: int = 1024,
):
    """Copy decode mode metadata from 1 source to 3 backends in a single kernel.

    Decode mode copies:
    1. cache_seqlens → cache_seqlens_int32
    2. cu_seqlens_k → cu_seqlens_k[1:]
    3. page_indices → page_table_1[:, :max_len]
    4. nsa_cache_seqlens → nsa_cache_seqlens_int32
    5. nsa_cu_seqlens_k[:size] → nsa_cu_seqlens_k[1:1+size]
    6. real_page_table (optional)
    7. flashmla_metadata (optional)

    Args:
        precomputed: Source PrecomputedMetadata
        metadata_b0, metadata_b1, metadata_b2: Destination metadata for 3 backends
        block_size: Block size for Triton kernel
    """
    # CPU-optimized: Pre-compute all slices once to avoid repeated view creation
    descriptors = []
    fallback_copies = []

    # Pre-compute common values and ALL tensor slices upfront
    size = precomputed.seqlens_expanded_size
    max_len = precomputed.max_len

    # Pre-slice all destination tensors (avoids creating views 3x per copy)
    cu_seqlens_k_b0 = metadata_b0.cu_seqlens_k[1:]
    cu_seqlens_k_b1 = metadata_b1.cu_seqlens_k[1:]
    cu_seqlens_k_b2 = metadata_b2.cu_seqlens_k[1:]

    page_table_b0 = metadata_b0.page_table_1[:, :max_len]
    page_table_b1 = metadata_b1.page_table_1[:, :max_len]
    page_table_b2 = metadata_b2.page_table_1[:, :max_len]

    nsa_cu_seqlens_b0 = metadata_b0.nsa_cu_seqlens_k[1:1+size]
    nsa_cu_seqlens_b1 = metadata_b1.nsa_cu_seqlens_k[1:1+size]
    nsa_cu_seqlens_b2 = metadata_b2.nsa_cu_seqlens_k[1:1+size]

    src_nsa_cu_seqlens = precomputed.nsa_cu_seqlens_k[:size]

    # Copy 1: cache_seqlens
    src = precomputed.cache_seqlens
    dst_b0 = metadata_b0.cache_seqlens_int32
    dst_b1 = metadata_b1.cache_seqlens_int32
    dst_b2 = metadata_b2.cache_seqlens_int32
    if src.is_contiguous() and dst_b0.is_contiguous() and dst_b1.is_contiguous() and dst_b2.is_contiguous():
        descriptors.append((src.view(-1), dst_b0.view(-1), dst_b1.view(-1), dst_b2.view(-1)))
    else:
        fallback_copies.extend([(src, dst_b0), (src, dst_b1), (src, dst_b2)])

    # Copy 2: cu_seqlens_k[1:] (pre-sliced)
    src = precomputed.cu_seqlens_k
    if src.is_contiguous() and cu_seqlens_k_b0.is_contiguous() and cu_seqlens_k_b1.is_contiguous() and cu_seqlens_k_b2.is_contiguous():
        descriptors.append((src.view(-1), cu_seqlens_k_b0.view(-1), cu_seqlens_k_b1.view(-1), cu_seqlens_k_b2.view(-1)))
    else:
        fallback_copies.extend([(src, cu_seqlens_k_b0), (src, cu_seqlens_k_b1), (src, cu_seqlens_k_b2)])

    # Copy 3: page_table_1[:, :max_len] (pre-sliced, typically non-contiguous)
    src = precomputed.page_indices
    if src.is_contiguous() and page_table_b0.is_contiguous() and page_table_b1.is_contiguous() and page_table_b2.is_contiguous():
        descriptors.append((src.view(-1), page_table_b0.view(-1), page_table_b1.view(-1), page_table_b2.view(-1)))
    else:
        fallback_copies.extend([(src, page_table_b0), (src, page_table_b1), (src, page_table_b2)])

    # Copy 4: nsa_cache_seqlens
    src = precomputed.nsa_cache_seqlens
    dst_b0 = metadata_b0.nsa_cache_seqlens_int32
    dst_b1 = metadata_b1.nsa_cache_seqlens_int32
    dst_b2 = metadata_b2.nsa_cache_seqlens_int32
    if src.is_contiguous() and dst_b0.is_contiguous() and dst_b1.is_contiguous() and dst_b2.is_contiguous():
        descriptors.append((src.view(-1), dst_b0.view(-1), dst_b1.view(-1), dst_b2.view(-1)))
    else:
        fallback_copies.extend([(src, dst_b0), (src, dst_b1), (src, dst_b2)])

    # Copy 5: nsa_cu_seqlens_k (pre-sliced)
    if src_nsa_cu_seqlens.is_contiguous() and nsa_cu_seqlens_b0.is_contiguous() and nsa_cu_seqlens_b1.is_contiguous() and nsa_cu_seqlens_b2.is_contiguous():
        descriptors.append((src_nsa_cu_seqlens.view(-1), nsa_cu_seqlens_b0.view(-1), nsa_cu_seqlens_b1.view(-1), nsa_cu_seqlens_b2.view(-1)))
    else:
        fallback_copies.extend([(src_nsa_cu_seqlens, nsa_cu_seqlens_b0), (src_nsa_cu_seqlens, nsa_cu_seqlens_b1), (src_nsa_cu_seqlens, nsa_cu_seqlens_b2)])

    # Copy 6: real_page_table (optional)
    if precomputed.real_page_table is not None:
        rows, cols = precomputed.real_page_table.shape
        src = precomputed.real_page_table
        real_page_b0 = metadata_b0.real_page_table[:rows, :cols]
        real_page_b1 = metadata_b1.real_page_table[:rows, :cols]
        real_page_b2 = metadata_b2.real_page_table[:rows, :cols]
        if src.is_contiguous() and real_page_b0.is_contiguous() and real_page_b1.is_contiguous() and real_page_b2.is_contiguous():
            descriptors.append((src.view(-1), real_page_b0.view(-1), real_page_b1.view(-1), real_page_b2.view(-1)))
        else:
            fallback_copies.extend([(src, real_page_b0), (src, real_page_b1), (src, real_page_b2)])

    # Copy 7: flashmla_metadata (optional, custom type - always fallback)
    if precomputed.flashmla_metadata is not None:
        fallback_copies.extend([
            (precomputed.flashmla_metadata, metadata_b0.flashmla_metadata.slice(slice(0, size + 1))),
            (precomputed.flashmla_metadata, metadata_b1.flashmla_metadata.slice(slice(0, size + 1))),
            (precomputed.flashmla_metadata, metadata_b2.flashmla_metadata.slice(slice(0, size + 1))),
        ])

    # Launch unified kernel if we have contiguous copies
    if descriptors:
        n_copies = len(descriptors)
        if n_copies > 8:
            raise ValueError(f"Too many copies: {n_copies} (max 8)")

        # Pad to 8 copies using cached dummy tuple (avoids allocating new tensors)
        if n_copies < 8:
            device = descriptors[0][0].device
            dummy_tuple = _get_dummy_tuple(device)
            descriptors.extend([dummy_tuple] * (8 - n_copies))

        # Calculate grid
        max_size = max(d[0].numel() for d in descriptors[:n_copies])
        max_blocks = (max_size + block_size - 1) // block_size
        grid = (3 * n_copies, max_blocks)

        # Launch kernel (tuples indexed as: 0=src, 1=dst_b0, 2=dst_b1, 3=dst_b2)
        batched_copy_to_3_backends_kernel[grid](
            # Sources
            descriptors[0][0], descriptors[1][0], descriptors[2][0], descriptors[3][0],
            descriptors[4][0], descriptors[5][0], descriptors[6][0], descriptors[7][0],
            # Sizes
            descriptors[0][0].numel(), descriptors[1][0].numel(),
            descriptors[2][0].numel(), descriptors[3][0].numel(),
            descriptors[4][0].numel(), descriptors[5][0].numel(),
            descriptors[6][0].numel(), descriptors[7][0].numel(),
            # Backend 0 destinations
            descriptors[0][1], descriptors[1][1], descriptors[2][1], descriptors[3][1],
            descriptors[4][1], descriptors[5][1], descriptors[6][1], descriptors[7][1],
            # Backend 1 destinations
            descriptors[0][2], descriptors[1][2], descriptors[2][2], descriptors[3][2],
            descriptors[4][2], descriptors[5][2], descriptors[6][2], descriptors[7][2],
            # Backend 2 destinations
            descriptors[0][3], descriptors[1][3], descriptors[2][3], descriptors[3][3],
            descriptors[4][3], descriptors[5][3], descriptors[6][3], descriptors[7][3],
            n_copies=n_copies,
            BLOCK_SIZE=block_size,
        )

    # Fallback to PyTorch .copy_() for non-contiguous tensors
    for src, dst in fallback_copies:
        dst.copy_(src)


def batched_copy_target_verify_to_3_backends(
    precomputed,
    metadata_b0,
    metadata_b1,
    metadata_b2,
    block_size: int = 1024,
):
    """Copy target_verify mode metadata from 1 source to 3 backends.

    Target verify mode copies:
    1. cache_seqlens → cache_seqlens_int32
    2. cu_seqlens_k → cu_seqlens_k[1:]
    3. page_indices → page_table_1[:, :max_seqlen_k]
    4. seqlens_expanded → nsa_seqlens_expanded
    5. nsa_cache_seqlens → nsa_cache_seqlens_int32
    6. nsa_cu_seqlens_k[:size] → nsa_cu_seqlens_k[1:1+size]
    7. real_page_table (optional)
    8. flashmla_metadata (optional)
    """
    # CPU-optimized: Pre-compute all slices once to avoid repeated view creation
    descriptors = []
    fallback_copies = []

    # Pre-compute common values and ALL tensor slices upfront
    size = precomputed.seqlens_expanded_size
    max_seqlen_k = precomputed.max_seqlen_k

    # Pre-slice all destination tensors (avoids creating views 3x per copy)
    cu_seqlens_k_b0 = metadata_b0.cu_seqlens_k[1:]
    cu_seqlens_k_b1 = metadata_b1.cu_seqlens_k[1:]
    cu_seqlens_k_b2 = metadata_b2.cu_seqlens_k[1:]

    page_table_b0 = metadata_b0.page_table_1[:, :max_seqlen_k]
    page_table_b1 = metadata_b1.page_table_1[:, :max_seqlen_k]
    page_table_b2 = metadata_b2.page_table_1[:, :max_seqlen_k]

    nsa_cu_seqlens_b0 = metadata_b0.nsa_cu_seqlens_k[1:1+size]
    nsa_cu_seqlens_b1 = metadata_b1.nsa_cu_seqlens_k[1:1+size]
    nsa_cu_seqlens_b2 = metadata_b2.nsa_cu_seqlens_k[1:1+size]

    src_nsa_cu_seqlens = precomputed.nsa_cu_seqlens_k[:size]

    # Copy 1: cache_seqlens
    src = precomputed.cache_seqlens
    dst_b0 = metadata_b0.cache_seqlens_int32
    dst_b1 = metadata_b1.cache_seqlens_int32
    dst_b2 = metadata_b2.cache_seqlens_int32
    if src.is_contiguous() and dst_b0.is_contiguous() and dst_b1.is_contiguous() and dst_b2.is_contiguous():
        descriptors.append((src.view(-1), dst_b0.view(-1), dst_b1.view(-1), dst_b2.view(-1)))
    else:
        fallback_copies.extend([(src, dst_b0), (src, dst_b1), (src, dst_b2)])

    # Copy 2: cu_seqlens_k[1:] (pre-sliced)
    src = precomputed.cu_seqlens_k
    if src.is_contiguous() and cu_seqlens_k_b0.is_contiguous() and cu_seqlens_k_b1.is_contiguous() and cu_seqlens_k_b2.is_contiguous():
        descriptors.append((src.view(-1), cu_seqlens_k_b0.view(-1), cu_seqlens_k_b1.view(-1), cu_seqlens_k_b2.view(-1)))
    else:
        fallback_copies.extend([(src, cu_seqlens_k_b0), (src, cu_seqlens_k_b1), (src, cu_seqlens_k_b2)])

    # Copy 3: page_table_1[:, :max_seqlen_k] (pre-sliced)
    src = precomputed.page_indices
    if src.is_contiguous() and page_table_b0.is_contiguous() and page_table_b1.is_contiguous() and page_table_b2.is_contiguous():
        descriptors.append((src.view(-1), page_table_b0.view(-1), page_table_b1.view(-1), page_table_b2.view(-1)))
    else:
        fallback_copies.extend([(src, page_table_b0), (src, page_table_b1), (src, page_table_b2)])

    # Copy 4: seqlens_expanded
    src = precomputed.seqlens_expanded
    dst_b0 = metadata_b0.nsa_seqlens_expanded
    dst_b1 = metadata_b1.nsa_seqlens_expanded
    dst_b2 = metadata_b2.nsa_seqlens_expanded
    if src.is_contiguous() and dst_b0.is_contiguous() and dst_b1.is_contiguous() and dst_b2.is_contiguous():
        descriptors.append((src.view(-1), dst_b0.view(-1), dst_b1.view(-1), dst_b2.view(-1)))
    else:
        fallback_copies.extend([(src, dst_b0), (src, dst_b1), (src, dst_b2)])

    # Copy 5: nsa_cache_seqlens
    src = precomputed.nsa_cache_seqlens
    dst_b0 = metadata_b0.nsa_cache_seqlens_int32
    dst_b1 = metadata_b1.nsa_cache_seqlens_int32
    dst_b2 = metadata_b2.nsa_cache_seqlens_int32
    if src.is_contiguous() and dst_b0.is_contiguous() and dst_b1.is_contiguous() and dst_b2.is_contiguous():
        descriptors.append((src.view(-1), dst_b0.view(-1), dst_b1.view(-1), dst_b2.view(-1)))
    else:
        fallback_copies.extend([(src, dst_b0), (src, dst_b1), (src, dst_b2)])

    # Copy 6: nsa_cu_seqlens_k (pre-sliced)
    if src_nsa_cu_seqlens.is_contiguous() and nsa_cu_seqlens_b0.is_contiguous() and nsa_cu_seqlens_b1.is_contiguous() and nsa_cu_seqlens_b2.is_contiguous():
        descriptors.append((src_nsa_cu_seqlens.view(-1), nsa_cu_seqlens_b0.view(-1), nsa_cu_seqlens_b1.view(-1), nsa_cu_seqlens_b2.view(-1)))
    else:
        fallback_copies.extend([(src_nsa_cu_seqlens, nsa_cu_seqlens_b0), (src_nsa_cu_seqlens, nsa_cu_seqlens_b1), (src_nsa_cu_seqlens, nsa_cu_seqlens_b2)])

    # Copy 7: real_page_table (optional, pre-sliced)
    if precomputed.real_page_table is not None:
        rows, cols = precomputed.real_page_table.shape
        src = precomputed.real_page_table
        real_page_b0 = metadata_b0.real_page_table[:rows, :cols]
        real_page_b1 = metadata_b1.real_page_table[:rows, :cols]
        real_page_b2 = metadata_b2.real_page_table[:rows, :cols]
        if src.is_contiguous() and real_page_b0.is_contiguous() and real_page_b1.is_contiguous() and real_page_b2.is_contiguous():
            descriptors.append((src.view(-1), real_page_b0.view(-1), real_page_b1.view(-1), real_page_b2.view(-1)))
        else:
            fallback_copies.extend([(src, real_page_b0), (src, real_page_b1), (src, real_page_b2)])

    # Copy 8: flashmla_metadata (optional, custom type - always fallback)
    if precomputed.flashmla_metadata is not None:
        fallback_copies.extend([
            (precomputed.flashmla_metadata, metadata_b0.flashmla_metadata.slice(slice(0, size + 1))),
            (precomputed.flashmla_metadata, metadata_b1.flashmla_metadata.slice(slice(0, size + 1))),
            (precomputed.flashmla_metadata, metadata_b2.flashmla_metadata.slice(slice(0, size + 1))),
        ])

    # Launch unified kernel
    if descriptors:
        n_copies = len(descriptors)
        if n_copies > 8:
            raise ValueError(f"Too many copies: {n_copies} (max 8)")

        # Pad to 8 copies using cached dummy tuple (avoids allocating new tensors)
        if n_copies < 8:
            device = descriptors[0][0].device
            dummy_tuple = _get_dummy_tuple(device)
            descriptors.extend([dummy_tuple] * (8 - n_copies))

        max_size = max(d[0].numel() for d in descriptors[:n_copies])
        max_blocks = (max_size + block_size - 1) // block_size
        grid = (3 * n_copies, max_blocks)

        batched_copy_to_3_backends_kernel[grid](
            descriptors[0][0], descriptors[1][0], descriptors[2][0], descriptors[3][0],
            descriptors[4][0], descriptors[5][0], descriptors[6][0], descriptors[7][0],
            descriptors[0][0].numel(), descriptors[1][0].numel(),
            descriptors[2][0].numel(), descriptors[3][0].numel(),
            descriptors[4][0].numel(), descriptors[5][0].numel(),
            descriptors[6][0].numel(), descriptors[7][0].numel(),
            descriptors[0][1], descriptors[1][1], descriptors[2][1], descriptors[3][1],
            descriptors[4][1], descriptors[5][1], descriptors[6][1], descriptors[7][1],
            descriptors[0][2], descriptors[1][2], descriptors[2][2], descriptors[3][2],
            descriptors[4][2], descriptors[5][2], descriptors[6][2], descriptors[7][2],
            descriptors[0][3], descriptors[1][3], descriptors[2][3], descriptors[3][3],
            descriptors[4][3], descriptors[5][3], descriptors[6][3], descriptors[7][3],
            n_copies=n_copies,
            BLOCK_SIZE=block_size,
        )

    for src, dst in fallback_copies:
        dst.copy_(src)


def batched_copy_draft_extend_to_3_backends(
    precomputed,
    metadata_b0,
    metadata_b1,
    metadata_b2,
    block_size: int = 1024,
):
    """Copy draft_extend mode metadata from 1 source to 3 backends.

    Draft extend mode copies:
    1. cache_seqlens → cache_seqlens_int32
    2. cu_seqlens_k → cu_seqlens_k[1:]
    3. page_indices → page_table_1[:rows, :cols]
    4. seqlens_expanded → nsa_seqlens_expanded[:size]
    5. nsa_cache_seqlens → nsa_cache_seqlens_int32[:size]
    6. nsa_cu_seqlens_k[:size] → nsa_cu_seqlens_k[1:1+size]
    7. real_page_table (optional)
    8. flashmla_metadata (optional)
    """
    # CPU-optimized: Pre-compute all slices once to avoid repeated view creation
    descriptors = []
    fallback_copies = []

    # Pre-compute common values and ALL tensor slices upfront
    size = precomputed.seqlens_expanded_size
    rows = precomputed.page_indices.shape[0]
    cols = precomputed.max_seqlen_k

    # Pre-slice all destination tensors (avoids creating views 3x per copy)
    cu_seqlens_k_b0 = metadata_b0.cu_seqlens_k[1:]
    cu_seqlens_k_b1 = metadata_b1.cu_seqlens_k[1:]
    cu_seqlens_k_b2 = metadata_b2.cu_seqlens_k[1:]

    page_table_b0 = metadata_b0.page_table_1[:rows, :cols]
    page_table_b1 = metadata_b1.page_table_1[:rows, :cols]
    page_table_b2 = metadata_b2.page_table_1[:rows, :cols]

    nsa_seqlens_expanded_b0 = metadata_b0.nsa_seqlens_expanded[:size]
    nsa_seqlens_expanded_b1 = metadata_b1.nsa_seqlens_expanded[:size]
    nsa_seqlens_expanded_b2 = metadata_b2.nsa_seqlens_expanded[:size]

    nsa_cache_seqlens_b0 = metadata_b0.nsa_cache_seqlens_int32[:size]
    nsa_cache_seqlens_b1 = metadata_b1.nsa_cache_seqlens_int32[:size]
    nsa_cache_seqlens_b2 = metadata_b2.nsa_cache_seqlens_int32[:size]

    nsa_cu_seqlens_b0 = metadata_b0.nsa_cu_seqlens_k[1:1+size]
    nsa_cu_seqlens_b1 = metadata_b1.nsa_cu_seqlens_k[1:1+size]
    nsa_cu_seqlens_b2 = metadata_b2.nsa_cu_seqlens_k[1:1+size]

    src_nsa_cu_seqlens = precomputed.nsa_cu_seqlens_k[:size]

    # Copy 1: cache_seqlens
    src = precomputed.cache_seqlens
    dst_b0 = metadata_b0.cache_seqlens_int32
    dst_b1 = metadata_b1.cache_seqlens_int32
    dst_b2 = metadata_b2.cache_seqlens_int32
    if src.is_contiguous() and dst_b0.is_contiguous() and dst_b1.is_contiguous() and dst_b2.is_contiguous():
        descriptors.append((src.view(-1), dst_b0.view(-1), dst_b1.view(-1), dst_b2.view(-1)))
    else:
        fallback_copies.extend([(src, dst_b0), (src, dst_b1), (src, dst_b2)])

    # Copy 2: cu_seqlens_k[1:] (pre-sliced)
    src = precomputed.cu_seqlens_k
    if src.is_contiguous() and cu_seqlens_k_b0.is_contiguous() and cu_seqlens_k_b1.is_contiguous() and cu_seqlens_k_b2.is_contiguous():
        descriptors.append((src.view(-1), cu_seqlens_k_b0.view(-1), cu_seqlens_k_b1.view(-1), cu_seqlens_k_b2.view(-1)))
    else:
        fallback_copies.extend([(src, cu_seqlens_k_b0), (src, cu_seqlens_k_b1), (src, cu_seqlens_k_b2)])

    # Copy 3: page_table_1[:rows, :cols] (pre-sliced)
    src = precomputed.page_indices
    if src.is_contiguous() and page_table_b0.is_contiguous() and page_table_b1.is_contiguous() and page_table_b2.is_contiguous():
        descriptors.append((src.view(-1), page_table_b0.view(-1), page_table_b1.view(-1), page_table_b2.view(-1)))
    else:
        fallback_copies.extend([(src, page_table_b0), (src, page_table_b1), (src, page_table_b2)])

    # Copy 4: seqlens_expanded (source, pre-sliced destinations)
    src = precomputed.seqlens_expanded
    if src.is_contiguous() and nsa_seqlens_expanded_b0.is_contiguous() and nsa_seqlens_expanded_b1.is_contiguous() and nsa_seqlens_expanded_b2.is_contiguous():
        descriptors.append((src.view(-1), nsa_seqlens_expanded_b0.view(-1), nsa_seqlens_expanded_b1.view(-1), nsa_seqlens_expanded_b2.view(-1)))
    else:
        fallback_copies.extend([(src, nsa_seqlens_expanded_b0), (src, nsa_seqlens_expanded_b1), (src, nsa_seqlens_expanded_b2)])

    # Copy 5: nsa_cache_seqlens (source, pre-sliced destinations)
    src = precomputed.nsa_cache_seqlens
    if src.is_contiguous() and nsa_cache_seqlens_b0.is_contiguous() and nsa_cache_seqlens_b1.is_contiguous() and nsa_cache_seqlens_b2.is_contiguous():
        descriptors.append((src.view(-1), nsa_cache_seqlens_b0.view(-1), nsa_cache_seqlens_b1.view(-1), nsa_cache_seqlens_b2.view(-1)))
    else:
        fallback_copies.extend([(src, nsa_cache_seqlens_b0), (src, nsa_cache_seqlens_b1), (src, nsa_cache_seqlens_b2)])

    # Copy 6: nsa_cu_seqlens_k (pre-sliced)
    if src_nsa_cu_seqlens.is_contiguous() and nsa_cu_seqlens_b0.is_contiguous() and nsa_cu_seqlens_b1.is_contiguous() and nsa_cu_seqlens_b2.is_contiguous():
        descriptors.append((src_nsa_cu_seqlens.view(-1), nsa_cu_seqlens_b0.view(-1), nsa_cu_seqlens_b1.view(-1), nsa_cu_seqlens_b2.view(-1)))
    else:
        fallback_copies.extend([(src_nsa_cu_seqlens, nsa_cu_seqlens_b0), (src_nsa_cu_seqlens, nsa_cu_seqlens_b1), (src_nsa_cu_seqlens, nsa_cu_seqlens_b2)])

    # Copy 7: real_page_table (optional, pre-sliced)
    if precomputed.real_page_table is not None:
        rows_real, cols_real = precomputed.real_page_table.shape
        src = precomputed.real_page_table
        real_page_b0 = metadata_b0.real_page_table[:rows_real, :cols_real]
        real_page_b1 = metadata_b1.real_page_table[:rows_real, :cols_real]
        real_page_b2 = metadata_b2.real_page_table[:rows_real, :cols_real]
        if src.is_contiguous() and real_page_b0.is_contiguous() and real_page_b1.is_contiguous() and real_page_b2.is_contiguous():
            descriptors.append((src.view(-1), real_page_b0.view(-1), real_page_b1.view(-1), real_page_b2.view(-1)))
        else:
            fallback_copies.extend([(src, real_page_b0), (src, real_page_b1), (src, real_page_b2)])

    # Copy 8: flashmla_metadata (optional, custom type - always fallback)
    if precomputed.flashmla_metadata is not None:
        fallback_copies.extend([
            (precomputed.flashmla_metadata, metadata_b0.flashmla_metadata.slice(slice(0, size + 1))),
            (precomputed.flashmla_metadata, metadata_b1.flashmla_metadata.slice(slice(0, size + 1))),
            (precomputed.flashmla_metadata, metadata_b2.flashmla_metadata.slice(slice(0, size + 1))),
        ])

    # Launch unified kernel
    if descriptors:
        n_copies = len(descriptors)
        if n_copies > 8:
            raise ValueError(f"Too many copies: {n_copies} (max 8)")

        # Pad to 8 copies using cached dummy tuple (avoids allocating new tensors)
        if n_copies < 8:
            device = descriptors[0][0].device
            dummy_tuple = _get_dummy_tuple(device)
            descriptors.extend([dummy_tuple] * (8 - n_copies))

        max_size = max(d[0].numel() for d in descriptors[:n_copies])
        max_blocks = (max_size + block_size - 1) // block_size
        grid = (3 * n_copies, max_blocks)

        batched_copy_to_3_backends_kernel[grid](
            descriptors[0][0], descriptors[1][0], descriptors[2][0], descriptors[3][0],
            descriptors[4][0], descriptors[5][0], descriptors[6][0], descriptors[7][0],
            descriptors[0][0].numel(), descriptors[1][0].numel(),
            descriptors[2][0].numel(), descriptors[3][0].numel(),
            descriptors[4][0].numel(), descriptors[5][0].numel(),
            descriptors[6][0].numel(), descriptors[7][0].numel(),
            descriptors[0][1], descriptors[1][1], descriptors[2][1], descriptors[3][1],
            descriptors[4][1], descriptors[5][1], descriptors[6][1], descriptors[7][1],
            descriptors[0][2], descriptors[1][2], descriptors[2][2], descriptors[3][2],
            descriptors[4][2], descriptors[5][2], descriptors[6][2], descriptors[7][2],
            descriptors[0][3], descriptors[1][3], descriptors[2][3], descriptors[3][3],
            descriptors[4][3], descriptors[5][3], descriptors[6][3], descriptors[7][3],
            n_copies=n_copies,
            BLOCK_SIZE=block_size,
        )

    for src, dst in fallback_copies:
        dst.copy_(src)


if __name__ == "__main__":
    print("Running batched copy benchmark...")
    benchmark_batched_copy()
