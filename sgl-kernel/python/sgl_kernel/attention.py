from typing import List, Optional, Tuple

import torch


def merge_state(
    v_a: torch.Tensor,
    s_a: torch.Tensor,
    v_b: torch.Tensor,
    s_b: torch.Tensor,
    v_merged: Optional[torch.Tensor] = None,
    s_merged: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    s_a = s_a.to(torch.float32)
    s_b = s_b.to(torch.float32)
    # Avoid creating new tensors if they are already provided
    if v_merged is None:
        v_merged = torch.empty_like(v_a)
    if s_merged is None:
        s_merged = torch.empty_like(s_a)
    torch.ops.sgl_kernel.merge_state.default(v_a, s_a, v_b, s_b, v_merged, s_merged)
    return v_merged, s_merged


def merge_state_v2(
    v_a: torch.Tensor,
    s_a: torch.Tensor,
    v_b: torch.Tensor,
    s_b: torch.Tensor,
    v_merged: Optional[torch.Tensor] = None,
    s_merged: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    s_a = s_a.to(torch.float32)
    s_b = s_b.to(torch.float32)
    # TODO(DefTruth): Currently, the custom merge_attn_states kernel
    # does not support the FP8 data type and non - CUDA devices.
    # It may be necessary to fall back to using the Triton kernel.

    # Avoid creating new tensors if they are already provided
    if v_merged is None:
        v_merged = torch.empty_like(v_a)
    if s_merged is None:
        s_merged = torch.empty_like(s_a)
    torch.ops.sgl_kernel.merge_state_v2.default(v_a, s_a, v_b, s_b, v_merged, s_merged)
    return v_merged, s_merged


def cutlass_mla_decode(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kv_c_and_k_pe_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    workspace: torch.Tensor,
    sm_scale: float,
    num_kv_splits: int = 1,  # Set to 1 to avoid cuda_graph issue by default.
) -> torch.Tensor:
    assert q_nope.ndim == 3, f"q_nope must be a 3D tensor, but got {q_nope.ndim}"
    assert q_pe.ndim == 3, f"q_pe must be a 3D tensor, but got {q_pe.ndim}"
    assert (
        kv_c_and_k_pe_cache.ndim == 3
    ), f"kv_c_and_k_pe_cache must be a 3D tensor, but got {kv_c_and_k_pe_cache.ndim}"

    B_q, H, D_q_nope = q_nope.shape
    B_q_2, H_2, D_q_pe = q_pe.shape
    assert (B_q == B_q_2) and (H == H_2)

    _, PAGE_SIZE, D_ckv = kv_c_and_k_pe_cache.shape

    D_latent = 512
    D_rope = 64
    assert D_q_nope == D_latent
    assert D_q_pe == D_rope
    assert D_ckv == D_latent + D_rope

    MAX_HEADS = 128
    assert H <= MAX_HEADS, f"H must be <= {MAX_HEADS}, but got {H}"
    if H < MAX_HEADS:
        q_nope_padded = q_nope.new_empty((B_q, MAX_HEADS, D_q_nope))
        q_nope_padded[:, :H] = q_nope
        q_nope = q_nope_padded

        q_pe_padded = q_pe.new_empty((B_q, MAX_HEADS, D_q_pe))
        q_pe_padded[:, :H] = q_pe
        q_pe = q_pe_padded

    assert len(page_table.shape) == 2
    B_block_table, block_num = page_table.shape
    assert B_block_table == B_q
    assert block_num > 0, f"block num must be greater than 0, got {block_num}"
    assert block_num % (128 / PAGE_SIZE) == 0

    # TODO(kaixih@nvidia): support fp8
    assert q_nope.dtype in (
        torch.float16,
        torch.bfloat16,
    ), f"q_nope.dtype needs to be fp16 or bf16 but got {q_nope.dtype}."
    assert q_nope.dtype == q_pe.dtype == kv_c_and_k_pe_cache.dtype
    assert (
        seq_lens.dtype == torch.int32
    ), f"seq_lens.dtype needs to be int32 but got {seq_lens.dtype}."
    assert (
        page_table.dtype == torch.int32
    ), f"page_table.dtype needs to be int32 but got {page_table.dtype}."

    out = q_nope.new_empty((B_q, MAX_HEADS, D_latent))

    torch.ops.sgl_kernel.cutlass_mla_decode.default(
        out,
        q_nope,
        q_pe,
        kv_c_and_k_pe_cache,
        seq_lens,
        page_table,
        workspace,
        sm_scale,
        num_kv_splits,
    )
    return out[:, :H].contiguous()


def cutlass_mla_get_workspace_size(
    max_seq_len: int,
    num_batches: int,
    sm_count: int = 0,
    num_kv_splits: int = 1,  # Set to 1 to avoid cuda_graph issue by default.
) -> int:
    assert max_seq_len > 0, f"max_seq_len must be greater than 0, got {max_seq_len}"
    assert num_batches > 0, f"num_batches must be greater than 0, got {num_batches}"
    return torch.ops.sgl_kernel.cutlass_mla_get_workspace_size.default(
        max_seq_len, num_batches, sm_count, num_kv_splits
    )


def fused_metadata_copy_cuda(
    cache_seqlens_src: torch.Tensor,
    cu_seqlens_k_src: torch.Tensor,
    page_indices_src: torch.Tensor,
    nsa_cache_seqlens_src: torch.Tensor,
    seqlens_expanded_src: torch.Tensor,
    nsa_cu_seqlens_k_src: torch.Tensor,
    real_page_table_src: Optional[torch.Tensor],
    flashmla_num_splits_src: Optional[torch.Tensor],
    flashmla_metadata_src: Optional[torch.Tensor],
    cache_seqlens_dst: torch.Tensor,
    cu_seqlens_k_dst: torch.Tensor,
    page_table_1_dst: torch.Tensor,
    nsa_cache_seqlens_dst: torch.Tensor,
    seqlens_expanded_dst: torch.Tensor,
    nsa_cu_seqlens_k_dst: torch.Tensor,
    real_page_table_dst: Optional[torch.Tensor],
    flashmla_num_splits_dst: Optional[torch.Tensor],
    flashmla_metadata_dst: Optional[torch.Tensor],
    forward_mode: int,
    bs: int,
    max_len: int,
    max_seqlen_k: int,
    seqlens_expanded_size: int,
) -> None:
    """
    Fused metadata copy kernel for NSA backend CUDA graph replay.

    This function fuses multiple tensor copy operations into a single kernel launch,
    reducing kernel launch overhead and improving performance.

    Args:
        cache_seqlens_src: Source cache sequence lengths [bs]
        cu_seqlens_k_src: Source cumulative sequence lengths [bs+1]
        page_indices_src: Source page indices [rows, max_len]
        nsa_cache_seqlens_src: Source NSA cache sequence lengths [size]
        seqlens_expanded_src: Source expanded sequence lengths [size]
        nsa_cu_seqlens_k_src: Source NSA cumulative sequence lengths [size+1]
        real_page_table_src: Optional source real page table [rows, cols]
        flashmla_num_splits_src: Optional source FlashMLA num_splits [size+1]
        flashmla_metadata_src: Optional source FlashMLA metadata tensor
        cache_seqlens_dst: Destination cache sequence lengths [bs]
        cu_seqlens_k_dst: Destination cumulative sequence lengths [bs+1]
        page_table_1_dst: Destination page table [rows, stride]
        nsa_cache_seqlens_dst: Destination NSA cache sequence lengths [size]
        seqlens_expanded_dst: Destination expanded sequence lengths [size]
        nsa_cu_seqlens_k_dst: Destination NSA cumulative sequence lengths [size+1]
        real_page_table_dst: Optional destination real page table [rows, cols]
        flashmla_num_splits_dst: Optional destination FlashMLA num_splits [size+1]
        flashmla_metadata_dst: Optional destination FlashMLA metadata tensor
        forward_mode: Forward mode (0=DECODE, 1=TARGET_VERIFY, 2=DRAFT_EXTEND)
        bs: Batch size
        max_len: Maximum length for decode/draft_extend mode
        max_seqlen_k: Maximum sequence length for target_verify mode
        seqlens_expanded_size: Size of expanded sequence lengths
    """
    torch.ops.sgl_kernel.fused_metadata_copy_cuda.default(
        cache_seqlens_src,
        cu_seqlens_k_src,
        page_indices_src,
        nsa_cache_seqlens_src,
        seqlens_expanded_src,
        nsa_cu_seqlens_k_src,
        real_page_table_src,
        flashmla_num_splits_src,
        flashmla_metadata_src,
        cache_seqlens_dst,
        cu_seqlens_k_dst,
        page_table_1_dst,
        nsa_cache_seqlens_dst,
        seqlens_expanded_dst,
        nsa_cu_seqlens_k_dst,
        real_page_table_dst,
        flashmla_num_splits_dst,
        flashmla_metadata_dst,
        forward_mode,
        bs,
        max_len,
        max_seqlen_k,
        seqlens_expanded_size,
    )


def precompute_decode_metadata_cuda(
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_indices: torch.Tensor,
    nsa_cache_seqlens: torch.Tensor,
    nsa_cu_seqlens_k: torch.Tensor,
    real_page_table: Optional[torch.Tensor],
    max_len: int,
    nsa_index_topk: int,
    real_page_size: int,
) -> None:
    """
    Precompute decode mode metadata for NSA backend.

    This function fuses multiple operations into a single kernel launch:
    1. dtype conversion (seq_lens -> cache_seqlens)
    2. cumulative sum with padding (cache_seqlens -> cu_seqlens_k)
    3. NSA seqlens computation (clamp)
    4. cumulative sum for NSA (nsa_cache_seqlens -> nsa_cu_seqlens_k)
    5. page table gathering from req_to_token
    6. page table transformation (if real_page_size > 1)

    Args:
        seq_lens: Input sequence lengths [bs], dtype can be int32 or int64
        req_pool_indices: Request pool indices [bs], can be int32 or int64 (auto-converted in kernel)
        req_to_token: Request to token mapping [total_requests, req_to_token_stride], can be int32 or int64 (auto-converted in kernel)
        cache_seqlens: Output cache sequence lengths [bs], int32
        cu_seqlens_k: Output cumulative sequence lengths [bs+1], int32
        page_indices: Output page indices [bs, max_len], int32
        nsa_cache_seqlens: Output NSA cache sequence lengths [bs], int32
        nsa_cu_seqlens_k: Output NSA cumulative sequence lengths [bs+1], int32
        real_page_table: Optional output real page table [bs, real_page_table_cols], int32
        max_len: Maximum length
        nsa_index_topk: NSA index top-k value
        real_page_size: Real page size
    """
    torch.ops.sgl_kernel.precompute_decode_metadata_cuda.default(
        seq_lens,
        req_pool_indices,
        req_to_token,
        cache_seqlens,
        cu_seqlens_k,
        page_indices,
        nsa_cache_seqlens,
        nsa_cu_seqlens_k,
        real_page_table,
        int(max_len),
        int(nsa_index_topk),
        int(real_page_size),
    )


def fused_metadata_precompute_and_broadcast_cuda(
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    backend_pointers: torch.Tensor,
    max_len: int,
    page_indices_dst_stride: int,
    nsa_index_topk: int,
    real_page_size: int,
    real_page_table_cols: int,
    real_page_table_dst_stride: int,
) -> None:
    """
    Fused kernel: precompute decode metadata + broadcast to N backends in one launch.

    This kernel combines two operations:
    1. Precompute decode metadata (dtype conversion, cumsum, NSA seqlens)
    2. Broadcast results to N backend destinations in parallel

    Benefits over separate calls:
    - Single kernel launch instead of 1 + N launches
    - Shared memory reuse: precomputed results directly broadcast
    - Reduced memory bandwidth: avoid intermediate writes
    - Better CUDA graph compatibility
    - Zero-copy: GPU pointers extracted in Python, no CPU round-trip

    Use case: Multi-step speculative decoding where N backends need identical metadata.

    Args:
        seq_lens: Input sequence lengths [bs], can be int32 or int64
        req_pool_indices: Request pool indices [bs], can be int32 or int64 (auto-converted)
        req_to_token: Request to token mapping [total_requests, stride], can be int32 or int64 (auto-converted)
        backend_pointers: Tensor [6, N] of int64 GPU pointers to destination buffers
            Row 0: cache_seqlens_int32 pointers
            Row 1: cu_seqlens_k pointers
            Row 2: page_table_1 pointers
            Row 3: nsa_cache_seqlens_int32 pointers
            Row 4: nsa_cu_seqlens_k pointers
            Row 5: real_page_table pointers (or 0 if nullptr)
        max_len: Maximum sequence length
        page_indices_dst_stride: Stride for page_indices destination tensors
        nsa_index_topk: NSA index top-k value for clamping
        real_page_size: Real page size for page table transformation
        real_page_table_cols: Number of columns in real_page_table
        real_page_table_dst_stride: Stride for real_page_table destination tensors

    Example:
        # Extract pointers on CPU to avoid multiple CPU->GPU copies
        # The function will automatically transfer to GPU in a single batched copy
        backend_pointers = torch.empty((6, N), dtype=torch.int64, device='cpu')
        for i, backend in enumerate(backends):
            backend_pointers[0, i] = backend.cache_seqlens.data_ptr()
            backend_pointers[1, i] = backend.cu_seqlens_k.data_ptr()
            # ... etc

        fused_metadata_precompute_and_broadcast_cuda(
            seq_lens, req_pool_indices, req_to_token, backend_pointers, ...)
    """
    # Transfer backend_pointers to GPU if it's on CPU (single batched transfer is much faster
    # than multiple individual CPU->GPU copies in the caller loop)
    if not backend_pointers.is_cuda:
        backend_pointers = backend_pointers.to(seq_lens.device)

    torch.ops.sgl_kernel.fused_metadata_precompute_and_broadcast_cuda.default(
        seq_lens,
        req_pool_indices,
        req_to_token,
        backend_pointers,
        int(max_len),
        int(page_indices_dst_stride),
        int(nsa_index_topk),
        int(real_page_size),
        int(real_page_table_cols),
        int(real_page_table_dst_stride),
    )


def unified_decode_metadata_cuda(
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    backend_pointers: torch.Tensor,
    max_len_allocated: int,
    nsa_index_topk: int,
    real_page_size: int,
    real_page_table_cols: int,
    real_page_table_dst_stride: int,
) -> None:
    """
    Unified DECODE mode metadata computation kernel.

    Performs ALL metadata operations in a single kernel launch:
    1. dtype conversion (seq_lens → cache_seqlens)
    2. max reduction (compute max_len)
    3. cumulative sum (cache_seqlens → cu_seqlens_k)
    4. NSA clamping (cache_seqlens → nsa_cache_seqlens)
    5. cumulative sum (nsa_cache_seqlens → nsa_cu_seqlens_k)
    6. seqlens_expanded = cache_seqlens (DECODE mode identity)
    7. page table gathering from req_to_token
    8. real page table transformation (if needed)
    9. broadcast all results to N backend destinations

    Benefits:
    - Single kernel launch for all metadata computation
    - All intermediate computation in shared memory
    - Zero CPU-GPU round-trips
    - Optimal for CUDA graph replay
    - 10-15x faster than sequential approach for multi-backend scenarios

    Args:
        seq_lens: Input sequence lengths [bs], int32 or int64
        req_pool_indices: Request pool indices [bs], int32 or int64 (auto-converted)
        req_to_token: Request to token mapping [total_requests, stride], int32 or int64 (auto-converted)
        backend_pointers: Tensor [7, N] of int64 GPU pointers to destination buffers
            Row 0: cache_seqlens_int32 pointers
            Row 1: cu_seqlens_k pointers
            Row 2: page_table_1 pointers
            Row 3: nsa_cache_seqlens_int32 pointers
            Row 4: nsa_cu_seqlens_k pointers
            Row 5: real_page_table pointers (or 0 if nullptr)
            Row 6: nsa_seqlens_expanded pointers
        max_len_allocated: Pre-allocated page table size (from metadata.page_table_1.size(1))
        nsa_index_topk: NSA index top-k value for clamping
        real_page_size: Real page size for page table transformation
        real_page_table_cols: Number of columns in real_page_table
        real_page_table_dst_stride: Stride for real_page_table destination tensors

    Example:
        # Extract pointers on CPU to avoid multiple CPU->GPU copies
        # The function will automatically transfer to GPU in a single batched copy
        backend_pointers = torch.empty((7, N), dtype=torch.int64, device='cpu')
        for i, backend in enumerate(backends):
            metadata = backend.decode_cuda_graph_metadata[bs]
            backend_pointers[0, i] = metadata.cache_seqlens_int32.data_ptr()
            backend_pointers[1, i] = metadata.cu_seqlens_k.data_ptr()
            backend_pointers[2, i] = metadata.page_table_1.data_ptr()
            backend_pointers[3, i] = metadata.nsa_cache_seqlens_int32.data_ptr()
            backend_pointers[4, i] = metadata.nsa_cu_seqlens_k.data_ptr()
            backend_pointers[5, i] = metadata.real_page_table.data_ptr()
            backend_pointers[6, i] = metadata.nsa_seqlens_expanded.data_ptr()

        unified_decode_metadata_cuda(
            seq_lens, req_pool_indices, req_to_token,
            backend_pointers, max_len_allocated, ...)
    """
    # Transfer backend_pointers to GPU if it's on CPU (single batched transfer is much faster
    # than multiple individual CPU->GPU copies in the caller loop)
    if not backend_pointers.is_cuda:
        backend_pointers = backend_pointers.to(seq_lens.device)

    torch.ops.sgl_kernel.unified_decode_metadata_cuda.default(
        seq_lens,
        req_pool_indices,
        req_to_token,
        backend_pointers,
        int(max_len_allocated),
        int(nsa_index_topk),
        int(real_page_size),
        int(real_page_table_cols),
        int(real_page_table_dst_stride),
    )


def unified_decode_metadata_cuda_direct(
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    backend_metadata_list: List,  # List of metadata objects
    nsa_index_topk: int,
    real_page_size: int,
) -> None:
    """
    Optimized variant for small number of backends (≤3).
    Passes pointers directly as kernel arguments - NO tensor allocation or GPU transfer!

    This eliminates:
    - CPU tensor allocation for backend_pointers
    - CPU->GPU transfer of pointer array
    - Any memory allocation overhead

    Benefits:
    - Zero tensor operations for pointer passing
    - Lower latency (especially for CUDA graph replay)
    - Cleaner code in caller

    Args:
        seq_lens: Input sequence lengths [bs], int32 or int64
        req_pool_indices: Request pool indices [bs]
        req_to_token: Request to token mapping [total_requests, stride]
        backend_metadata_list: List of backend decode metadata objects (1-3 items)
        nsa_index_topk: NSA index top-k value
        real_page_size: Real page size for page table transformation

    Example:
        # No tensor allocation needed - just pass the metadata objects directly!
        metadata_list = [backend.decode_cuda_graph_metadata[bs] for backend in backends]

        unified_decode_metadata_cuda_direct(
            seq_lens, req_pool_indices, req_to_token,
            metadata_list, nsa_index_topk, real_page_size)
    """
    num_backends = len(backend_metadata_list)
    assert 1 <= num_backends <= 3, f"Direct variant supports 1-3 backends, got {num_backends}"

    # Extract parameters from first backend
    first_metadata = backend_metadata_list[0]
    max_len_allocated = first_metadata.page_table_1.size(1)
    real_page_table_cols = first_metadata.real_page_table.size(1) if real_page_size > 1 else 0
    real_page_table_dst_stride = first_metadata.real_page_table.stride(0) if real_page_size > 1 else 0

    # Extract pointers for each backend (pad with 0 for unused backends)
    def get_ptrs(idx):
        if idx < num_backends:
            metadata = backend_metadata_list[idx]
            return [
                metadata.cache_seqlens_int32.data_ptr(),
                metadata.cu_seqlens_k.data_ptr(),
                metadata.page_table_1.data_ptr(),
                metadata.nsa_cache_seqlens_int32.data_ptr(),
                metadata.nsa_cu_seqlens_k.data_ptr(),
                metadata.real_page_table.data_ptr() if real_page_size > 1 else 0,
                metadata.nsa_seqlens_expanded.data_ptr(),
            ]
        else:
            return [0, 0, 0, 0, 0, 0, 0]  # Unused backend

    ptrs0 = get_ptrs(0)
    ptrs1 = get_ptrs(1)
    ptrs2 = get_ptrs(2)

    torch.ops.sgl_kernel.unified_decode_metadata_cuda_direct.default(
        seq_lens,
        req_pool_indices,
        req_to_token,
        # Backend 0
        ptrs0[0], ptrs0[1], ptrs0[2], ptrs0[3], ptrs0[4], ptrs0[5], ptrs0[6],
        # Backend 1
        ptrs1[0], ptrs1[1], ptrs1[2], ptrs1[3], ptrs1[4], ptrs1[5], ptrs1[6],
        # Backend 2
        ptrs2[0], ptrs2[1], ptrs2[2], ptrs2[3], ptrs2[4], ptrs2[5], ptrs2[6],
        # Parameters
        num_backends,
        max_len_allocated,
        nsa_index_topk,
        real_page_size,
        real_page_table_cols,
        real_page_table_dst_stride,
    )
