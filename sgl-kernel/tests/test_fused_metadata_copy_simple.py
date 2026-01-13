"""
Simple standalone test for fused metadata copy kernel (no pytest required).
Run with: python test_fused_metadata_copy_simple.py
"""

import torch
from sgl_kernel import fused_metadata_copy_cuda


def test_decode_mode():
    """Test DECODE mode with all optional tensors."""
    print("Testing DECODE mode...")

    bs = 4
    max_len = 128
    seqlens_expanded_size = bs

    # Create source tensors
    cache_seqlens_src = torch.randint(1, max_len, (bs,), dtype=torch.int32, device="cuda")
    cu_seqlens_k_src = torch.zeros(bs + 1, dtype=torch.int32, device="cuda")
    cu_seqlens_k_src[1:] = torch.cumsum(cache_seqlens_src, dim=0)

    page_indices_src = torch.randint(0, 1000, (bs, max_len), dtype=torch.int32, device="cuda")
    nsa_cache_seqlens_src = torch.randint(1, max_len, (seqlens_expanded_size,), dtype=torch.int32, device="cuda")
    seqlens_expanded_src = torch.randint(1, max_len, (seqlens_expanded_size,), dtype=torch.int32, device="cuda")
    nsa_cu_seqlens_k_src = torch.zeros(seqlens_expanded_size + 1, dtype=torch.int32, device="cuda")
    nsa_cu_seqlens_k_src[1:] = torch.cumsum(nsa_cache_seqlens_src, dim=0)

    real_page_table_src = torch.randint(0, 1000, (bs, 64), dtype=torch.int32, device="cuda")
    flashmla_num_splits_src = torch.randint(1, 10, (seqlens_expanded_size + 1,), dtype=torch.int32, device="cuda")
    flashmla_metadata_src = torch.randint(0, 100, (128,), dtype=torch.int32, device="cuda")

    # Create destination tensors for reference
    cache_seqlens_ref = torch.zeros(bs, dtype=torch.int32, device="cuda")
    cu_seqlens_k_ref = torch.zeros(bs + 1, dtype=torch.int32, device="cuda")
    page_table_1_ref = torch.zeros((bs, max_len + 16), dtype=torch.int32, device="cuda")
    nsa_cache_seqlens_ref = torch.zeros(seqlens_expanded_size, dtype=torch.int32, device="cuda")
    nsa_cu_seqlens_k_ref = torch.zeros(seqlens_expanded_size + 1, dtype=torch.int32, device="cuda")
    real_page_table_ref = torch.zeros((bs, 80), dtype=torch.int32, device="cuda")
    flashmla_num_splits_ref = torch.zeros(seqlens_expanded_size + 1, dtype=torch.int32, device="cuda")
    flashmla_metadata_ref = torch.zeros(128, dtype=torch.int32, device="cuda")

    # Reference implementation
    cache_seqlens_ref.copy_(cache_seqlens_src)
    cu_seqlens_k_ref[1:].copy_(cu_seqlens_k_src[1:])
    page_table_1_ref[:, :max_len].copy_(page_indices_src)
    nsa_cache_seqlens_ref.copy_(nsa_cache_seqlens_src)
    nsa_cu_seqlens_k_ref[1:bs+1].copy_(nsa_cu_seqlens_k_src[1:bs+1])
    real_page_table_ref[:, :64].copy_(real_page_table_src)
    flashmla_num_splits_ref[:bs+1].copy_(flashmla_num_splits_src[:bs+1])
    flashmla_metadata_ref.copy_(flashmla_metadata_src)

    # Create destination tensors for fused kernel
    cache_seqlens_fused = torch.zeros(bs, dtype=torch.int32, device="cuda")
    cu_seqlens_k_fused = torch.zeros(bs + 1, dtype=torch.int32, device="cuda")
    page_table_1_fused = torch.zeros((bs, max_len + 16), dtype=torch.int32, device="cuda")
    nsa_cache_seqlens_fused = torch.zeros(seqlens_expanded_size, dtype=torch.int32, device="cuda")
    nsa_seqlens_expanded_fused = torch.zeros(seqlens_expanded_size, dtype=torch.int32, device="cuda")
    nsa_cu_seqlens_k_fused = torch.zeros(seqlens_expanded_size + 1, dtype=torch.int32, device="cuda")
    real_page_table_fused = torch.zeros((bs, 80), dtype=torch.int32, device="cuda")
    flashmla_num_splits_fused = torch.zeros(seqlens_expanded_size + 1, dtype=torch.int32, device="cuda")
    flashmla_metadata_fused = torch.zeros(128, dtype=torch.int32, device="cuda")

    # Run fused kernel
    print(f"\nCalling kernel with:")
    print(f"  bs={bs}, max_len={max_len}, seqlens_expanded_size={seqlens_expanded_size}")
    print(f"  real_page_table_src shape: {real_page_table_src.shape}")
    print(f"  real_page_table_fused shape: {real_page_table_fused.shape}")

    # Ensure all tensors are contiguous
    cache_seqlens_src = cache_seqlens_src.contiguous()
    cu_seqlens_k_src = cu_seqlens_k_src.contiguous()
    page_indices_src = page_indices_src.contiguous()
    nsa_cache_seqlens_src = nsa_cache_seqlens_src.contiguous()
    seqlens_expanded_src = seqlens_expanded_src.contiguous()
    nsa_cu_seqlens_k_src = nsa_cu_seqlens_k_src.contiguous()
    real_page_table_src = real_page_table_src.contiguous()
    flashmla_num_splits_src = flashmla_num_splits_src.contiguous()
    flashmla_metadata_src = flashmla_metadata_src.contiguous()

    fused_metadata_copy_cuda(
        cache_seqlens_src,
        cu_seqlens_k_src,
        page_indices_src,
        nsa_cache_seqlens_src,
        seqlens_expanded_src,
        nsa_cu_seqlens_k_src,
        real_page_table_src,
        flashmla_num_splits_src,
        flashmla_metadata_src,
        cache_seqlens_fused,
        cu_seqlens_k_fused,
        page_table_1_fused,
        nsa_cache_seqlens_fused,
        nsa_seqlens_expanded_fused,
        nsa_cu_seqlens_k_fused,
        real_page_table_fused,
        flashmla_num_splits_fused,
        flashmla_metadata_fused,
        0,  # DECODE mode
        bs,
        max_len,
        256,  # max_seqlen_k (not used in DECODE)
        seqlens_expanded_size,
    )

    # Compare results
    assert torch.equal(cache_seqlens_ref, cache_seqlens_fused), "cache_seqlens mismatch"
    assert torch.equal(cu_seqlens_k_ref, cu_seqlens_k_fused), "cu_seqlens_k mismatch"
    assert torch.equal(page_table_1_ref, page_table_1_fused), "page_table_1 mismatch"
    assert torch.equal(nsa_cache_seqlens_ref, nsa_cache_seqlens_fused), "nsa_cache_seqlens mismatch"
    assert torch.equal(nsa_cu_seqlens_k_ref, nsa_cu_seqlens_k_fused), "nsa_cu_seqlens_k mismatch"

    # Debug real_page_table
    if not torch.equal(real_page_table_ref, real_page_table_fused):
        print(f"real_page_table_ref shape: {real_page_table_ref.shape}")
        print(f"real_page_table_fused shape: {real_page_table_fused.shape}")
        print(f"real_page_table_src shape: {real_page_table_src.shape}")
        print(f"real_page_table_src is_contiguous: {real_page_table_src.is_contiguous()}")
        print(f"real_page_table_src stride: {real_page_table_src.stride()}")
        print(f"real_page_table_ref is_contiguous: {real_page_table_ref.is_contiguous()}")
        print(f"real_page_table_ref stride: {real_page_table_ref.stride()}")
        print(f"real_page_table_fused is_contiguous: {real_page_table_fused.is_contiguous()}")
        print(f"real_page_table_fused stride: {real_page_table_fused.stride()}")
        print(f"\nreal_page_table_src (first 2 rows, first 10 cols):\n{real_page_table_src[:2, :10]}")
        print(f"\nreal_page_table_ref (first 2 rows, first 10 cols):\n{real_page_table_ref[:2, :10]}")
        print(f"\nreal_page_table_fused (first 2 rows, first 10 cols):\n{real_page_table_fused[:2, :10]}")
        print(f"\nFlat view of source [64:74]: {real_page_table_src.view(-1)[64:74]}")
        print(f"Source row 1: {real_page_table_src[1, :10]}")

    assert torch.equal(real_page_table_ref, real_page_table_fused), "real_page_table mismatch"
    assert torch.equal(flashmla_num_splits_ref, flashmla_num_splits_fused), "flashmla_num_splits mismatch"
    assert torch.equal(flashmla_metadata_ref, flashmla_metadata_fused), "flashmla_metadata mismatch"

    print("✓ DECODE mode test passed!")


def test_without_optional_tensors():
    """Test without optional tensors (real_page_table and flashmla)."""
    print("\nTesting without optional tensors...")

    bs = 2
    max_len = 64
    seqlens_expanded_size = bs

    # Create source tensors (no optional tensors)
    cache_seqlens_src = torch.randint(1, max_len, (bs,), dtype=torch.int32, device="cuda")
    cu_seqlens_k_src = torch.zeros(bs + 1, dtype=torch.int32, device="cuda")
    cu_seqlens_k_src[1:] = torch.cumsum(cache_seqlens_src, dim=0)

    page_indices_src = torch.randint(0, 1000, (bs, max_len), dtype=torch.int32, device="cuda")
    nsa_cache_seqlens_src = torch.randint(1, max_len, (seqlens_expanded_size,), dtype=torch.int32, device="cuda")
    seqlens_expanded_src = torch.randint(1, max_len, (seqlens_expanded_size,), dtype=torch.int32, device="cuda")
    nsa_cu_seqlens_k_src = torch.zeros(seqlens_expanded_size + 1, dtype=torch.int32, device="cuda")
    nsa_cu_seqlens_k_src[1:] = torch.cumsum(nsa_cache_seqlens_src, dim=0)

    # Create destination tensors
    cache_seqlens_dst = torch.zeros(bs, dtype=torch.int32, device="cuda")
    cu_seqlens_k_dst = torch.zeros(bs + 1, dtype=torch.int32, device="cuda")
    page_table_1_dst = torch.zeros((bs, max_len), dtype=torch.int32, device="cuda")
    nsa_cache_seqlens_dst = torch.zeros(seqlens_expanded_size, dtype=torch.int32, device="cuda")
    nsa_seqlens_expanded_dst = torch.zeros(seqlens_expanded_size, dtype=torch.int32, device="cuda")
    nsa_cu_seqlens_k_dst = torch.zeros(seqlens_expanded_size + 1, dtype=torch.int32, device="cuda")

    # Run fused kernel
    fused_metadata_copy_cuda(
        cache_seqlens_src,
        cu_seqlens_k_src,
        page_indices_src,
        nsa_cache_seqlens_src,
        seqlens_expanded_src,
        nsa_cu_seqlens_k_src,
        None,  # real_page_table_src
        None,  # flashmla_num_splits_src
        None,  # flashmla_metadata_src
        cache_seqlens_dst,
        cu_seqlens_k_dst,
        page_table_1_dst,
        nsa_cache_seqlens_dst,
        nsa_seqlens_expanded_dst,
        nsa_cu_seqlens_k_dst,
        None,  # real_page_table_dst
        None,  # flashmla_num_splits_dst
        None,  # flashmla_metadata_dst
        0,  # DECODE mode
        bs,
        max_len,
        256,
        seqlens_expanded_size,
    )

    # Verify results
    assert torch.equal(cache_seqlens_dst, cache_seqlens_src), "cache_seqlens mismatch"
    assert torch.equal(cu_seqlens_k_dst[1:], cu_seqlens_k_src[1:]), "cu_seqlens_k mismatch"
    assert torch.equal(page_table_1_dst, page_indices_src), "page_table_1 mismatch"
    assert torch.equal(nsa_cache_seqlens_dst, nsa_cache_seqlens_src), "nsa_cache_seqlens mismatch"

    print("✓ Test without optional tensors passed!")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests")
        exit(0)

    print("Running fused metadata copy kernel tests...\n")

    try:
        test_decode_mode()
        test_without_optional_tensors()
        print("\n✅ All tests passed!")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
