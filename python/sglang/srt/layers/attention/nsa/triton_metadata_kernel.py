"""
Fused Triton kernel for draft_extend metadata computation.
Replaces CPU-side loops and .tolist() calls with GPU-native computation.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def fill_draft_extend_metadata_kernel(
    # Input pointers
    extend_seq_lens_ptr,  # [bs]
    seq_lens_ptr,  # [bs]
    extend_offsets_ptr,  # [bs+1] - prefix sum of extend_seq_lens
    nsa_index_topk,  # scalar
    bs,  # scalar
    # Output pointers
    seqlens_expanded_ptr,  # [total_tokens]
    nsa_cache_seqlens_ptr,  # [total_tokens]
    # Strides
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel to compute seqlens_expanded and nsa_cache_seqlens.

    Logic:
    For each token in each batch:
        seqlens_expanded[token_id] = kv_len - extend_len + 1 + local_token_id
        nsa_cache_seqlens[token_id] = min(seqlens_expanded[token_id], nsa_index_topk)
    """
    # Global token ID
    token_id = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Load total tokens from extend_offsets[bs]
    total_tokens = tl.load(extend_offsets_ptr + bs)

    # Mask for valid tokens
    mask = token_id < total_tokens

    # Binary search to find which batch this token belongs to
    # For simplicity, we'll use a linear search (acceptable for small bs)
    batch_id = tl.zeros_like(token_id)

    for b in range(bs):
        offset_start = tl.load(extend_offsets_ptr + b)
        offset_end = tl.load(extend_offsets_ptr + b + 1)
        # If token_id is in [offset_start, offset_end), set batch_id = b
        in_range = (token_id >= offset_start) & (token_id < offset_end)
        batch_id = tl.where(in_range, b, batch_id)

    # Load batch-specific values
    extend_len = tl.load(extend_seq_lens_ptr + batch_id, mask=mask, other=0)
    kv_len = tl.load(seq_lens_ptr + batch_id, mask=mask, other=0)
    offset_start = tl.load(extend_offsets_ptr + batch_id, mask=mask, other=0)

    # Compute local token ID within batch
    local_token_id = token_id - offset_start

    # Compute seqlens_expanded[token_id] = kv_len - extend_len + 1 + local_token_id
    seq_val = kv_len - extend_len + 1 + local_token_id

    # Compute nsa_cache_seqlens[token_id] = min(seq_val, nsa_index_topk)
    nsa_seq_val = tl.minimum(seq_val, nsa_index_topk)

    # Store results
    tl.store(seqlens_expanded_ptr + token_id, seq_val, mask=mask)
    tl.store(nsa_cache_seqlens_ptr + token_id, nsa_seq_val, mask=mask)


@triton.jit
def copy_page_table_kernel(
    # Input pointers
    page_indices_ptr,  # [total_tokens, max_seqlen_k]
    max_seqlen_k,  # scalar
    total_tokens,  # scalar
    # Output pointer
    out_page_table_ptr,  # [total_tokens, max_seqlen_k]
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """Copy page_table with coalesced memory access."""
    token_id = tl.program_id(0)

    if token_id >= total_tokens:
        return

    # Copy one row at a time
    col_idx = tl.arange(0, BLOCK_SIZE)

    for col_start in range(0, max_seqlen_k, BLOCK_SIZE):
        col = col_start + col_idx
        mask = col < max_seqlen_k

        # Load from input
        in_offset = token_id * max_seqlen_k + col
        val = tl.load(page_indices_ptr + in_offset, mask=mask, other=0)

        # Store to output
        out_offset = token_id * max_seqlen_k + col
        tl.store(out_page_table_ptr + out_offset, val, mask=mask)


def fill_draft_extend_metadata_fused(
    extend_seq_lens: torch.Tensor,  # [bs], int32, GPU
    seq_lens: torch.Tensor,  # [bs], int32, GPU
    cache_seqlens: torch.Tensor,  # [bs], int32, GPU
    cumulate_cache_seqlens: torch.Tensor,  # [bs], int32, GPU
    page_indices: torch.Tensor,  # [total_tokens, max_seqlen_k], int32, GPU
    nsa_index_topk: int,
    metadata,  # NSAMetadata object
):
    """
    Fused function to replace lines 1007-1031 in nsa_backend.py.

    Eliminates:
    - 2x .tolist() GPU→CPU sync
    - Python for loop
    - torch.cat dynamic allocation
    - Multiple separate copy operations

    Args:
        extend_seq_lens: [bs] - number of extend tokens per batch
        seq_lens: [bs] - sequence length (kv_len) per batch
        cache_seqlens: [bs] - cache sequence lengths
        cumulate_cache_seqlens: [bs] - cumulative sum of cache_seqlens
        page_indices: [total_tokens, max_seqlen_k] - page table indices
        nsa_index_topk: scalar - NSA topk parameter
        metadata: NSAMetadata object to fill
    """
    bs = extend_seq_lens.shape[0]
    device = extend_seq_lens.device

    # Compute prefix sum of extend_seq_lens to get offsets
    # This is O(bs) and happens on GPU
    extend_offsets = torch.cumsum(
        torch.cat([
            torch.zeros(1, dtype=torch.int32, device=device),
            extend_seq_lens
        ]),
        dim=0
    )

    # Get total number of tokens (single CPU sync, unavoidable for allocation)
    total_tokens = extend_offsets[-1].item()

    if total_tokens == 0:
        # Edge case: no tokens to process
        return

    # Allocate output buffers (or reuse from metadata)
    max_seqlen_k = page_indices.shape[1]

    # Launch kernel to compute seqlens_expanded and nsa_cache_seqlens
    BLOCK_SIZE = 256
    grid = (triton.cdiv(total_tokens, BLOCK_SIZE),)

    fill_draft_extend_metadata_kernel[grid](
        extend_seq_lens,
        seq_lens,
        extend_offsets,
        nsa_index_topk,
        bs,
        metadata.nsa_seqlens_expanded,
        metadata.nsa_cache_seqlens_int32,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Copy scalar metadata (very fast, negligible cost)
    metadata.cache_seqlens_int32.copy_(cache_seqlens)
    metadata.cu_seqlens_k[1:].copy_(cumulate_cache_seqlens)

    # Copy page_table using Triton kernel for better performance
    if total_tokens > 0:
        grid_page = (total_tokens,)
        BLOCK_SIZE_PAGE = 64  # Tune based on max_seqlen_k

        copy_page_table_kernel[grid_page](
            page_indices,
            max_seqlen_k,
            total_tokens,
            metadata.page_table_1,
            BLOCK_SIZE=BLOCK_SIZE_PAGE,
        )


def fill_draft_extend_metadata_fused_simple(
    extend_seq_lens: torch.Tensor,  # [bs], int32, GPU
    seq_lens: torch.Tensor,  # [bs], int32, GPU
    nsa_index_topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Simplified version that only computes seqlens_expanded and nsa_cache_seqlens.
    Can be used to replace just lines 1007-1023.

    Returns:
        seqlens_expanded: [total_tokens]
        nsa_cache_seqlens: [total_tokens]
    """
    bs = extend_seq_lens.shape[0]
    device = extend_seq_lens.device

    # Compute prefix sum
    extend_offsets = torch.cumsum(
        torch.cat([
            torch.zeros(1, dtype=torch.int32, device=device),
            extend_seq_lens
        ]),
        dim=0
    )

    total_tokens = extend_offsets[-1].item()

    if total_tokens == 0:
        return (
            torch.empty(0, dtype=torch.int32, device=device),
            torch.empty(0, dtype=torch.int32, device=device)
        )

    # Allocate outputs
    seqlens_expanded = torch.empty(total_tokens, dtype=torch.int32, device=device)
    nsa_cache_seqlens = torch.empty(total_tokens, dtype=torch.int32, device=device)

    # Launch kernel
    BLOCK_SIZE = 256
    grid = (triton.cdiv(total_tokens, BLOCK_SIZE),)

    fill_draft_extend_metadata_kernel[grid](
        extend_seq_lens,
        seq_lens,
        extend_offsets,
        nsa_index_topk,
        bs,
        seqlens_expanded,
        nsa_cache_seqlens,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return seqlens_expanded, nsa_cache_seqlens


def fill_draft_extend_metadata_inplace(
    extend_seq_lens: torch.Tensor,  # [bs], int32, GPU
    seq_lens: torch.Tensor,  # [bs], int32, GPU
    nsa_index_topk: int,
    out_seqlens_expanded: torch.Tensor,  # [max_tokens], pre-allocated output buffer
    out_nsa_cache_seqlens: torch.Tensor,  # [max_tokens], pre-allocated output buffer
) -> int:
    """
    In-place version that writes directly to pre-allocated metadata buffers.
    Eliminates the need for .copy_() operations.

    This version is more efficient when metadata buffers are already allocated
    (e.g., in CUDA graph replay scenarios).

    Args:
        extend_seq_lens: [bs] - number of extend tokens per batch
        seq_lens: [bs] - sequence length (kv_len) per batch
        nsa_index_topk: scalar - NSA topk parameter
        out_seqlens_expanded: [max_tokens] - pre-allocated output buffer
        out_nsa_cache_seqlens: [max_tokens] - pre-allocated output buffer

    Returns:
        total_tokens: int - actual number of tokens written
    """
    bs = extend_seq_lens.shape[0]
    device = extend_seq_lens.device

    # Compute prefix sum
    extend_offsets = torch.cumsum(
        torch.cat([
            torch.zeros(1, dtype=torch.int32, device=device),
            extend_seq_lens
        ]),
        dim=0
    )

    total_tokens = extend_offsets[-1].item()

    if total_tokens == 0:
        return 0

    # Verify output buffers are large enough
    assert out_seqlens_expanded.shape[0] >= total_tokens, (
        f"out_seqlens_expanded buffer too small: {out_seqlens_expanded.shape[0]} < {total_tokens}"
    )
    assert out_nsa_cache_seqlens.shape[0] >= total_tokens, (
        f"out_nsa_cache_seqlens buffer too small: {out_nsa_cache_seqlens.shape[0]} < {total_tokens}"
    )

    # Launch kernel - writes directly to output buffers
    BLOCK_SIZE = 256
    grid = (triton.cdiv(total_tokens, BLOCK_SIZE),)

    fill_draft_extend_metadata_kernel[grid](
        extend_seq_lens,
        seq_lens,
        extend_offsets,
        nsa_index_topk,
        bs,
        out_seqlens_expanded,  # Write directly to metadata buffer
        out_nsa_cache_seqlens,  # Write directly to metadata buffer
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return total_tokens
