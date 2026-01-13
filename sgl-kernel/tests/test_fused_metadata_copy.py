"""
Test for fused metadata copy kernel.

This test verifies that the fused kernel produces identical results to
individual .copy_() operations across all forward modes and optional tensor combinations.
"""

import pytest
import torch
from sgl_kernel import fused_metadata_copy_cuda


def create_test_metadata(
    bs: int,
    max_len: int,
    max_seqlen_k: int,
    seqlens_expanded_size: int,
    has_real_page_table: bool = False,
    has_flashmla: bool = False,
    device: str = "cuda",
):
    """Create test metadata tensors matching NSA backend structure."""
    # Basic tensors (always present)
    cache_seqlens_src = torch.randint(
        1, max_len, (bs,), dtype=torch.int32, device=device
    )
    cu_seqlens_k_src = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    cu_seqlens_k_src[1:] = torch.cumsum(cache_seqlens_src, dim=0)

    page_indices_src = torch.randint(
        0, 1000, (bs, max_len), dtype=torch.int32, device=device
    )
    nsa_cache_seqlens_src = torch.randint(
        1, max_len, (seqlens_expanded_size,), dtype=torch.int32, device=device
    )
    seqlens_expanded_src = torch.randint(
        1, max_seqlen_k, (seqlens_expanded_size,), dtype=torch.int32, device=device
    )
    nsa_cu_seqlens_k_src = torch.zeros(
        seqlens_expanded_size + 1, dtype=torch.int32, device=device
    )
    nsa_cu_seqlens_k_src[1:] = torch.cumsum(nsa_cache_seqlens_src, dim=0)

    # Destination tensors
    cache_seqlens_dst = torch.zeros(bs, dtype=torch.int32, device=device)
    cu_seqlens_k_dst = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    page_table_1_dst = torch.zeros((bs, max_len + 16), dtype=torch.int32, device=device)
    nsa_cache_seqlens_dst = torch.zeros(
        seqlens_expanded_size, dtype=torch.int32, device=device
    )
    nsa_seqlens_expanded_dst = torch.zeros(
        seqlens_expanded_size, dtype=torch.int32, device=device
    )
    nsa_cu_seqlens_k_dst = torch.zeros(
        seqlens_expanded_size + 1, dtype=torch.int32, device=device
    )

    # Optional tensors
    real_page_table_src = None
    real_page_table_dst = None
    if has_real_page_table:
        real_page_table_cols = max_len // 2
        real_page_table_src = torch.randint(
            0, 1000, (bs, real_page_table_cols), dtype=torch.int32, device=device
        )
        real_page_table_dst = torch.zeros(
            (bs, real_page_table_cols + 8), dtype=torch.int32, device=device
        )

    flashmla_num_splits_src = None
    flashmla_num_splits_dst = None
    flashmla_metadata_src = None
    flashmla_metadata_dst = None
    if has_flashmla:
        flashmla_num_splits_src = torch.randint(
            1, 10, (seqlens_expanded_size + 1,), dtype=torch.int32, device=device
        )
        flashmla_num_splits_dst = torch.zeros(
            seqlens_expanded_size + 1, dtype=torch.int32, device=device
        )
        # FlashMLA metadata is typically (num_sm_parts, TileSchedulerMetaDataSize)
        # For testing, we use a simplified size
        flashmla_metadata_size = 128
        flashmla_metadata_src = torch.randint(
            0, 100, (flashmla_metadata_size,), dtype=torch.int32, device=device
        )
        flashmla_metadata_dst = torch.zeros(
            flashmla_metadata_size, dtype=torch.int32, device=device
        )

    return {
        "src": {
            "cache_seqlens": cache_seqlens_src,
            "cu_seqlens_k": cu_seqlens_k_src,
            "page_indices": page_indices_src,
            "nsa_cache_seqlens": nsa_cache_seqlens_src,
            "seqlens_expanded": seqlens_expanded_src,
            "nsa_cu_seqlens_k": nsa_cu_seqlens_k_src,
            "real_page_table": real_page_table_src,
            "flashmla_num_splits": flashmla_num_splits_src,
            "flashmla_metadata": flashmla_metadata_src,
        },
        "dst": {
            "cache_seqlens": cache_seqlens_dst,
            "cu_seqlens_k": cu_seqlens_k_dst,
            "page_table_1": page_table_1_dst,
            "nsa_cache_seqlens": nsa_cache_seqlens_dst,
            "nsa_seqlens_expanded": nsa_seqlens_expanded_dst,
            "nsa_cu_seqlens_k": nsa_cu_seqlens_k_dst,
            "real_page_table": real_page_table_dst,
            "flashmla_num_splits": flashmla_num_splits_dst,
            "flashmla_metadata": flashmla_metadata_dst,
        },
    }


def reference_copy_decode(src, dst, max_len):
    """Reference implementation: individual .copy_() for DECODE mode."""
    bs = src["cache_seqlens"].shape[0]
    dst["cache_seqlens"].copy_(src["cache_seqlens"])
    dst["cu_seqlens_k"][1:].copy_(src["cu_seqlens_k"][1:])
    dst["page_table_1"][:, :max_len].copy_(src["page_indices"])
    dst["nsa_cache_seqlens"].copy_(src["nsa_cache_seqlens"])
    dst["nsa_cu_seqlens_k"][1 : bs + 1].copy_(src["nsa_cu_seqlens_k"][1 : bs + 1])

    if src["real_page_table"] is not None:
        rows, cols = src["real_page_table"].shape
        dst["real_page_table"][:rows, :cols].copy_(src["real_page_table"])

    if src["flashmla_num_splits"] is not None:
        flashmla_size = bs + 1
        dst["flashmla_num_splits"][:flashmla_size].copy_(
            src["flashmla_num_splits"][:flashmla_size]
        )

    if src["flashmla_metadata"] is not None:
        dst["flashmla_metadata"].copy_(src["flashmla_metadata"])


def reference_copy_target_verify(src, dst, max_seqlen_k, seqlens_expanded_size):
    """Reference implementation: individual .copy_() for TARGET_VERIFY mode."""
    bs = src["cache_seqlens"].shape[0]
    dst["cache_seqlens"].copy_(src["cache_seqlens"])
    dst["cu_seqlens_k"][1:].copy_(src["cu_seqlens_k"][1:])

    rows = src["page_indices"].shape[0]
    dst["page_table_1"][:rows, :max_seqlen_k].copy_(src["page_indices"])
    dst["nsa_seqlens_expanded"][:seqlens_expanded_size].copy_(src["seqlens_expanded"])
    dst["nsa_cache_seqlens"][:seqlens_expanded_size].copy_(src["nsa_cache_seqlens"])
    dst["nsa_cu_seqlens_k"][1 : seqlens_expanded_size + 1].copy_(
        src["nsa_cu_seqlens_k"][1 : seqlens_expanded_size + 1]
    )

    if src["real_page_table"] is not None:
        rows, cols = src["real_page_table"].shape
        dst["real_page_table"][:rows, :cols].copy_(src["real_page_table"])

    if src["flashmla_num_splits"] is not None:
        flashmla_size = seqlens_expanded_size + 1
        dst["flashmla_num_splits"][:flashmla_size].copy_(
            src["flashmla_num_splits"][:flashmla_size]
        )

    if src["flashmla_metadata"] is not None:
        dst["flashmla_metadata"].copy_(src["flashmla_metadata"])


def reference_copy_draft_extend(src, dst, max_seqlen_k, seqlens_expanded_size):
    """Reference implementation: individual .copy_() for DRAFT_EXTEND mode."""
    bs = src["cache_seqlens"].shape[0]
    dst["cache_seqlens"].copy_(src["cache_seqlens"])
    dst["cu_seqlens_k"][1:].copy_(src["cu_seqlens_k"][1:])

    rows, cols = src["page_indices"].shape
    dst["page_table_1"][:rows, :cols].copy_(src["page_indices"])
    dst["nsa_seqlens_expanded"][:seqlens_expanded_size].copy_(src["seqlens_expanded"])
    dst["nsa_cache_seqlens"][:seqlens_expanded_size].copy_(src["nsa_cache_seqlens"])
    dst["nsa_cu_seqlens_k"][1 : seqlens_expanded_size + 1].copy_(
        src["nsa_cu_seqlens_k"][1 : seqlens_expanded_size + 1]
    )

    if src["real_page_table"] is not None:
        rows, cols = src["real_page_table"].shape
        dst["real_page_table"][:rows, :cols].copy_(src["real_page_table"])

    if src["flashmla_num_splits"] is not None:
        flashmla_size = seqlens_expanded_size + 1
        dst["flashmla_num_splits"][:flashmla_size].copy_(
            src["flashmla_num_splits"][:flashmla_size]
        )

    if src["flashmla_metadata"] is not None:
        dst["flashmla_metadata"].copy_(src["flashmla_metadata"])


@pytest.mark.parametrize("bs", [1, 2, 4, 8])
@pytest.mark.parametrize("forward_mode", [0, 1, 2])  # DECODE, TARGET_VERIFY, DRAFT_EXTEND
@pytest.mark.parametrize("has_real_page_table", [False, True])
@pytest.mark.parametrize("has_flashmla", [False, True])
def test_fused_metadata_copy(bs, forward_mode, has_real_page_table, has_flashmla):
    """Test fused metadata copy kernel against reference implementation."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    max_len = 128
    max_seqlen_k = 256
    seqlens_expanded_size = bs if forward_mode == 0 else bs * 2

    # Create test data
    data = create_test_metadata(
        bs=bs,
        max_len=max_len,
        max_seqlen_k=max_seqlen_k,
        seqlens_expanded_size=seqlens_expanded_size,
        has_real_page_table=has_real_page_table,
        has_flashmla=has_flashmla,
    )

    # Create separate destination tensors for reference and fused kernel
    dst_ref = {k: v.clone() for k, v in data["dst"].items()}
    dst_fused = {k: v.clone() for k, v in data["dst"].items()}

    # Run reference implementation
    if forward_mode == 0:  # DECODE
        reference_copy_decode(data["src"], dst_ref, max_len)
    elif forward_mode == 1:  # TARGET_VERIFY
        reference_copy_target_verify(data["src"], dst_ref, max_seqlen_k, seqlens_expanded_size)
    else:  # DRAFT_EXTEND
        reference_copy_draft_extend(data["src"], dst_ref, max_seqlen_k, seqlens_expanded_size)

    # Run fused kernel
    fused_metadata_copy_cuda(
        data["src"]["cache_seqlens"],
        data["src"]["cu_seqlens_k"],
        data["src"]["page_indices"],
        data["src"]["nsa_cache_seqlens"],
        data["src"]["seqlens_expanded"],
        data["src"]["nsa_cu_seqlens_k"],
        data["src"]["real_page_table"],
        data["src"]["flashmla_num_splits"],
        data["src"]["flashmla_metadata"],
        dst_fused["cache_seqlens"],
        dst_fused["cu_seqlens_k"],
        dst_fused["page_table_1"],
        dst_fused["nsa_cache_seqlens"],
        dst_fused["nsa_seqlens_expanded"],
        dst_fused["nsa_cu_seqlens_k"],
        dst_fused["real_page_table"],
        dst_fused["flashmla_num_splits"],
        dst_fused["flashmla_metadata"],
        forward_mode,
        bs,
        max_len,
        max_seqlen_k,
        seqlens_expanded_size,
    )

    # Compare results
    assert torch.equal(dst_ref["cache_seqlens"], dst_fused["cache_seqlens"]), \
        "cache_seqlens mismatch"
    assert torch.equal(dst_ref["cu_seqlens_k"], dst_fused["cu_seqlens_k"]), \
        "cu_seqlens_k mismatch"
    assert torch.equal(dst_ref["page_table_1"], dst_fused["page_table_1"]), \
        "page_table_1 mismatch"
    assert torch.equal(dst_ref["nsa_cache_seqlens"], dst_fused["nsa_cache_seqlens"]), \
        "nsa_cache_seqlens mismatch"
    assert torch.equal(dst_ref["nsa_seqlens_expanded"], dst_fused["nsa_seqlens_expanded"]), \
        "nsa_seqlens_expanded mismatch"
    assert torch.equal(dst_ref["nsa_cu_seqlens_k"], dst_fused["nsa_cu_seqlens_k"]), \
        "nsa_cu_seqlens_k mismatch"

    if has_real_page_table:
        assert torch.equal(dst_ref["real_page_table"], dst_fused["real_page_table"]), \
            "real_page_table mismatch"

    if has_flashmla:
        assert torch.equal(dst_ref["flashmla_num_splits"], dst_fused["flashmla_num_splits"]), \
            "flashmla_num_splits mismatch"
        assert torch.equal(dst_ref["flashmla_metadata"], dst_fused["flashmla_metadata"]), \
            "flashmla_metadata mismatch"


@pytest.mark.parametrize("bs", [16, 32])
def test_fused_metadata_copy_large_batch(bs):
    """Test with larger batch sizes."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    forward_mode = 0  # DECODE
    max_len = 128
    max_seqlen_k = 256
    seqlens_expanded_size = bs

    data = create_test_metadata(
        bs=bs,
        max_len=max_len,
        max_seqlen_k=max_seqlen_k,
        seqlens_expanded_size=seqlens_expanded_size,
        has_real_page_table=True,
        has_flashmla=True,
    )

    dst_ref = {k: v.clone() for k, v in data["dst"].items()}
    dst_fused = {k: v.clone() for k, v in data["dst"].items()}

    reference_copy_decode(data["src"], dst_ref, max_len)

    fused_metadata_copy_cuda(
        data["src"]["cache_seqlens"],
        data["src"]["cu_seqlens_k"],
        data["src"]["page_indices"],
        data["src"]["nsa_cache_seqlens"],
        data["src"]["seqlens_expanded"],
        data["src"]["nsa_cu_seqlens_k"],
        data["src"]["real_page_table"],
        data["src"]["flashmla_num_splits"],
        data["src"]["flashmla_metadata"],
        dst_fused["cache_seqlens"],
        dst_fused["cu_seqlens_k"],
        dst_fused["page_table_1"],
        dst_fused["nsa_cache_seqlens"],
        dst_fused["nsa_seqlens_expanded"],
        dst_fused["nsa_cu_seqlens_k"],
        dst_fused["real_page_table"],
        dst_fused["flashmla_num_splits"],
        dst_fused["flashmla_metadata"],
        forward_mode,
        bs,
        max_len,
        max_seqlen_k,
        seqlens_expanded_size,
    )

    # Verify all tensors match
    for key in dst_ref:
        if dst_ref[key] is not None:
            assert torch.equal(dst_ref[key], dst_fused[key]), f"{key} mismatch"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
