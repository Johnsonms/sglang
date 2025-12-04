"""
Precomputation methods for NSA backend optimization.

This file contains the implementation of precomputation optimization
for multi-step speculative decoding. These methods should be added to
the NativeSparseAttnBackend class.

To integrate: Copy the methods below into NativeSparseAttnBackend class.
"""

from typing import Optional
import torch
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.layers.attention.nsa.utils import compute_nsa_seqlens


def _precompute_replay_metadata(
    self,
    bs: int,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    forward_mode: ForwardMode,
    spec_info: Optional["SpecInput"],
) -> "PrecomputedMetadata":
    """Precompute all shared metadata for multi-step backends.

    This function extracts and computes all operations that are
    identical across different backend instances in multi-step
    speculative decoding.

    Performance: Saves ~(N-1) × 175μs for N backend instances.

    Args:
        bs: Batch size
        req_pool_indices: Request pool indices [bs]
        seq_lens: Sequence lengths [bs]
        seq_lens_cpu: Sequence lengths on CPU [bs]
        forward_mode: Forward mode (decode/target_verify/draft_extend)
        spec_info: Speculative decoding info (for draft_extend mode)

    Returns:
        PrecomputedMetadata containing all shared intermediate results
    """
    # Import here to avoid circular dependency
    from sglang.srt.layers.attention.nsa_backend import PrecomputedMetadata

    # Slice inputs to batch size
    seq_lens = seq_lens[:bs]
    seq_lens_cpu = seq_lens_cpu[:bs]
    req_pool_indices = req_pool_indices[:bs]

    # Dispatch to mode-specific precomputation
    if forward_mode.is_decode_or_idle():
        return self._precompute_decode_mode(
            bs, req_pool_indices, seq_lens, seq_lens_cpu
        )
    elif forward_mode.is_target_verify():
        return self._precompute_target_verify_mode(
            bs, req_pool_indices, seq_lens, seq_lens_cpu
        )
    elif forward_mode.is_draft_extend():
        return self._precompute_draft_extend_mode(
            bs, req_pool_indices, seq_lens, seq_lens_cpu, spec_info
        )
    else:
        raise ValueError(f"Unsupported forward mode: {forward_mode}")


def _precompute_decode_mode(
    self,
    bs: int,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
) -> "PrecomputedMetadata":
    """Precompute metadata for normal decode mode."""
    from sglang.srt.layers.attention.nsa_backend import PrecomputedMetadata

    max_len = int(seq_lens_cpu.max().item())

    # Convert to int32 and compute cumsum
    cache_seqlens = seq_lens.to(torch.int32)
    cu_seqlens_k = torch.nn.functional.pad(
        torch.cumsum(cache_seqlens, dim=0, dtype=torch.int32), (1, 0)
    )

    # Get page indices from cache
    page_indices = self.req_to_token[req_pool_indices, :max_len]

    # Compute NSA seqlens
    nsa_cache_seqlens = compute_nsa_seqlens(
        cache_seqlens, nsa_index_topk=self.nsa_index_topk
    )
    seqlens_expanded = cache_seqlens
    seqlens_expanded_size = seqlens_expanded.shape[0]

    # Compute NSA cumsum
    nsa_cu_seqlens_k = torch.nn.functional.pad(
        torch.cumsum(nsa_cache_seqlens, dim=0, dtype=torch.int32), (1, 0)
    )

    # Transform page table if needed
    if self.real_page_size > 1:
        real_page_table = self._transform_table_1_to_real(page_indices)
    else:
        real_page_table = None  # Will use page_indices directly

    # Compute FlashMLA metadata if needed
    flashmla_metadata = None
    if self.nsa_decode_impl == "flashmla_kv":
        flashmla_metadata = self._compute_flashmla_metadata(
            cache_seqlens=nsa_cache_seqlens,
            seq_len_q=1,
        )

    return PrecomputedMetadata(
        cache_seqlens=cache_seqlens,
        cu_seqlens_k=cu_seqlens_k,
        page_indices=page_indices,
        real_page_table=real_page_table,
        seqlens_expanded=seqlens_expanded,
        nsa_cache_seqlens=nsa_cache_seqlens,
        nsa_cu_seqlens_k=nsa_cu_seqlens_k,
        seqlens_expanded_size=seqlens_expanded_size,
        max_len=max_len,
        max_seqlen_k=max_len,
        flashmla_metadata=flashmla_metadata,
    )


def _precompute_target_verify_mode(
    self,
    bs: int,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
) -> "PrecomputedMetadata":
    """Precompute metadata for target verify mode."""
    from sglang.srt.layers.attention.nsa_backend import PrecomputedMetadata

    max_seqlen_k = int(
        seq_lens_cpu.max().item() + self.speculative_num_draft_tokens
    )

    # Cache seqlens with draft tokens
    cache_seqlens = (seq_lens + self.speculative_num_draft_tokens).to(torch.int32)
    cu_seqlens_k = torch.nn.functional.pad(
        torch.cumsum(cache_seqlens, dim=0, dtype=torch.int32), (1, 0)
    )

    # Page indices (repeated for each draft token)
    page_indices = self.req_to_token[req_pool_indices, :max_seqlen_k]
    page_indices = torch.repeat_interleave(
        page_indices, repeats=self.speculative_num_draft_tokens, dim=0
    )

    # Generate expanded seqlens
    extend_seq_lens_cpu = [self.speculative_num_draft_tokens] * bs
    seqlens_int32_cpu = [
        self.speculative_num_draft_tokens + kv_len
        for kv_len in seq_lens_cpu.tolist()
    ]
    seqlens_expanded = torch.cat(
        [
            torch.arange(
                kv_len - qo_len + 1,
                kv_len + 1,
                dtype=torch.int32,
                device=self.device,
            )
            for qo_len, kv_len in zip(
                extend_seq_lens_cpu,
                seqlens_int32_cpu,
                strict=True,
            )
        ]
    )

    # Compute NSA seqlens
    nsa_cache_seqlens = compute_nsa_seqlens(
        seqlens_expanded, self.nsa_index_topk
    )
    seqlens_expanded_size = seqlens_expanded.shape[0]

    # NSA cumsum
    nsa_cu_seqlens_k = torch.nn.functional.pad(
        torch.cumsum(nsa_cache_seqlens, dim=0, dtype=torch.int32), (1, 0)
    )

    # Transform page table
    if self.real_page_size > 1:
        real_page_table = self._transform_table_1_to_real(page_indices)
    else:
        real_page_table = None

    # FlashMLA metadata
    flashmla_metadata = None
    if self.nsa_decode_impl == "flashmla_kv":
        flashmla_metadata = self._compute_flashmla_metadata(
            cache_seqlens=nsa_cache_seqlens,
            seq_len_q=1,
        )

    return PrecomputedMetadata(
        cache_seqlens=cache_seqlens,
        cu_seqlens_k=cu_seqlens_k,
        page_indices=page_indices,
        real_page_table=real_page_table,
        seqlens_expanded=seqlens_expanded,
        nsa_cache_seqlens=nsa_cache_seqlens,
        nsa_cu_seqlens_k=nsa_cu_seqlens_k,
        seqlens_expanded_size=seqlens_expanded_size,
        max_len=-1,  # Not used in this mode
        max_seqlen_k=max_seqlen_k,
        flashmla_metadata=flashmla_metadata,
    )


def _precompute_draft_extend_mode(
    self,
    bs: int,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    spec_info: "SpecInput",
) -> "PrecomputedMetadata":
    """Precompute metadata for draft extend mode."""
    from sglang.srt.layers.attention.nsa_backend import PrecomputedMetadata

    max_seqlen_k = int(seq_lens_cpu.max().item())

    # Cache seqlens
    cache_seqlens = seq_lens.to(torch.int32)
    cu_seqlens_k = torch.nn.functional.pad(
        torch.cumsum(cache_seqlens, dim=0, dtype=torch.int32), (1, 0)
    )

    # Extend seqlens from spec_info
    extend_seq_lens = spec_info.accept_length[:bs]
    extend_seq_lens_cpu = extend_seq_lens.tolist()

    # Page indices (repeated per accept length)
    page_indices = self.req_to_token[req_pool_indices, :max_seqlen_k]
    page_indices = torch.repeat_interleave(
        page_indices, repeats=extend_seq_lens, dim=0
    )

    # Generate expanded seqlens
    seqlens_expanded = torch.cat(
        [
            torch.arange(
                kv_len - qo_len + 1,
                kv_len + 1,
                dtype=torch.int32,
                device=self.device,
            )
            for qo_len, kv_len in zip(
                extend_seq_lens_cpu,
                seq_lens_cpu.tolist(),
                strict=True,
            )
        ]
    )

    # Compute NSA seqlens
    nsa_cache_seqlens = compute_nsa_seqlens(
        seqlens_expanded, self.nsa_index_topk
    )
    seqlens_expanded_size = seqlens_expanded.shape[0]

    # NSA cumsum
    nsa_cu_seqlens_k = torch.nn.functional.pad(
        torch.cumsum(nsa_cache_seqlens, dim=0, dtype=torch.int32), (1, 0)
    )

    # Transform page table
    if self.real_page_size > 1:
        real_page_table = self._transform_table_1_to_real(page_indices)
    else:
        real_page_table = None

    # FlashMLA metadata
    flashmla_metadata = None
    if self.nsa_decode_impl == "flashmla_kv":
        flashmla_metadata = self._compute_flashmla_metadata(
            cache_seqlens=nsa_cache_seqlens,
            seq_len_q=1,
        )

    return PrecomputedMetadata(
        cache_seqlens=cache_seqlens,
        cu_seqlens_k=cu_seqlens_k,
        page_indices=page_indices,
        real_page_table=real_page_table,
        seqlens_expanded=seqlens_expanded,
        nsa_cache_seqlens=nsa_cache_seqlens,
        nsa_cu_seqlens_k=nsa_cu_seqlens_k,
        seqlens_expanded_size=seqlens_expanded_size,
        max_len=max_seqlen_k,
        max_seqlen_k=max_seqlen_k,
        flashmla_metadata=flashmla_metadata,
    )


def init_forward_metadata_replay_cuda_graph_from_precomputed(
    self,
    bs: int,
    precomputed: "PrecomputedMetadata",
    forward_mode: ForwardMode,
):
    """Fast path: copy precomputed metadata to this backend's metadata.

    This function only performs copy operations, no computation.
    Performance: ~20μs (vs ~175μs for full computation).

    Args:
        bs: Batch size
        precomputed: Precomputed metadata to copy from
        forward_mode: Forward mode
    """
    self.set_nsa_prefill_impl(forward_batch=None)

    metadata = self.decode_cuda_graph_metadata[bs]

    # Copy basic seqlens
    metadata.cache_seqlens_int32.copy_(precomputed.cache_seqlens)
    metadata.cu_seqlens_k[1:].copy_(precomputed.cu_seqlens_k[1:])

    # Mode-specific copy logic
    if forward_mode.is_decode_or_idle():
        # Decode mode
        metadata.page_table_1[:, :precomputed.max_len].copy_(
            precomputed.page_indices
        )
        metadata.nsa_cache_seqlens_int32.copy_(precomputed.nsa_cache_seqlens)
        # seqlens_expanded is same as cache_seqlens (already copied)

    elif forward_mode.is_target_verify():
        # Target verify mode
        metadata.page_table_1[:, :precomputed.max_seqlen_k].copy_(
            precomputed.page_indices
        )
        metadata.nsa_seqlens_expanded.copy_(precomputed.seqlens_expanded)
        metadata.nsa_cache_seqlens_int32.copy_(precomputed.nsa_cache_seqlens)

    elif forward_mode.is_draft_extend():
        # Draft extend mode
        rows = precomputed.page_indices.shape[0]
        cols = precomputed.max_seqlen_k
        metadata.page_table_1[:rows, :cols].copy_(precomputed.page_indices)

        size = precomputed.seqlens_expanded_size
        metadata.nsa_seqlens_expanded[:size].copy_(precomputed.seqlens_expanded)
        metadata.nsa_cache_seqlens_int32[:size].copy_(precomputed.nsa_cache_seqlens)

    # Copy NSA cu_seqlens
    size = precomputed.seqlens_expanded_size
    metadata.nsa_cu_seqlens_k[1:1+size].copy_(precomputed.nsa_cu_seqlens_k[1:1+size])

    # Copy real page table
    if precomputed.real_page_table is not None:
        rows, cols = precomputed.real_page_table.shape
        metadata.real_page_table[:rows, :cols].copy_(precomputed.real_page_table)
    else:
        # real_page_table is same as page_table_1 (already copied)
        pass

    # Copy FlashMLA metadata
    if precomputed.flashmla_metadata is not None:
        flashmla_metadata = metadata.flashmla_metadata.slice(
            slice(0, size + 1)
        )
        flashmla_metadata.copy_(precomputed.flashmla_metadata)

    self.forward_metadata = metadata
