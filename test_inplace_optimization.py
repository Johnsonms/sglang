#!/usr/bin/env python3
"""
Test the in-place optimization that eliminates .copy_() operations.
"""

import torch
import time


def test_inplace_optimization():
    """Test that in-place version produces identical results and is faster."""
    print("=" * 60)
    print("Testing In-Place Optimization")
    print("=" * 60)

    from sglang.srt.layers.attention.nsa.triton_metadata_kernel import (
        fill_draft_extend_metadata_fused_simple,
        fill_draft_extend_metadata_inplace,
    )

    device = torch.device("cuda")

    # Test case
    extend_seq_lens = torch.tensor([2, 3, 5], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([10, 15, 20], dtype=torch.int32, device=device)
    nsa_index_topk = 128

    total_tokens = extend_seq_lens.sum().item()

    print(f"\nTest case:")
    print(f"  extend_seq_lens: {extend_seq_lens.tolist()}")
    print(f"  seq_lens: {seq_lens.tolist()}")
    print(f"  total_tokens: {total_tokens}")

    # Method 1: Original (allocate + copy)
    seqlens_expanded_1, nsa_cache_seqlens_1 = fill_draft_extend_metadata_fused_simple(
        extend_seq_lens, seq_lens, nsa_index_topk
    )

    # Simulate the .copy_() operations
    metadata_buffer_1 = torch.empty(100, dtype=torch.int32, device=device)
    metadata_buffer_2 = torch.empty(100, dtype=torch.int32, device=device)
    metadata_buffer_1[:total_tokens].copy_(seqlens_expanded_1)
    metadata_buffer_2[:total_tokens].copy_(nsa_cache_seqlens_1)

    # Method 2: In-place (direct write)
    metadata_buffer_3 = torch.empty(100, dtype=torch.int32, device=device)
    metadata_buffer_4 = torch.empty(100, dtype=torch.int32, device=device)

    returned_total = fill_draft_extend_metadata_inplace(
        extend_seq_lens, seq_lens, nsa_index_topk,
        metadata_buffer_3, metadata_buffer_4
    )

    # Verify correctness
    assert returned_total == total_tokens, f"Total tokens mismatch: {returned_total} != {total_tokens}"

    assert torch.equal(
        metadata_buffer_1[:total_tokens],
        metadata_buffer_3[:total_tokens]
    ), "seqlens_expanded mismatch between methods"

    assert torch.equal(
        metadata_buffer_2[:total_tokens],
        metadata_buffer_4[:total_tokens]
    ), "nsa_cache_seqlens mismatch between methods"

    print("\n✅ Correctness test passed!")
    print(f"   Method 1 (allocate+copy): {metadata_buffer_1[:total_tokens].tolist()}")
    print(f"   Method 2 (in-place):      {metadata_buffer_3[:total_tokens].tolist()}")

    # Performance comparison
    print("\n" + "-" * 60)
    print("Performance Comparison")
    print("-" * 60)

    n_iters = 1000

    # Warmup
    for _ in range(10):
        seqlens_expanded, nsa_cache_seqlens = fill_draft_extend_metadata_fused_simple(
            extend_seq_lens, seq_lens, nsa_index_topk
        )
        metadata_buffer_1[:total_tokens].copy_(seqlens_expanded)
        metadata_buffer_2[:total_tokens].copy_(nsa_cache_seqlens)

    for _ in range(10):
        fill_draft_extend_metadata_inplace(
            extend_seq_lens, seq_lens, nsa_index_topk,
            metadata_buffer_3, metadata_buffer_4
        )

    # Benchmark Method 1: allocate + copy
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iters):
        seqlens_expanded, nsa_cache_seqlens = fill_draft_extend_metadata_fused_simple(
            extend_seq_lens, seq_lens, nsa_index_topk
        )
        metadata_buffer_1[:total_tokens].copy_(seqlens_expanded)
        metadata_buffer_2[:total_tokens].copy_(nsa_cache_seqlens)
    torch.cuda.synchronize()
    time_method1 = (time.time() - start) / n_iters * 1000  # ms

    # Benchmark Method 2: in-place
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iters):
        fill_draft_extend_metadata_inplace(
            extend_seq_lens, seq_lens, nsa_index_topk,
            metadata_buffer_3, metadata_buffer_4
        )
    torch.cuda.synchronize()
    time_method2 = (time.time() - start) / n_iters * 1000  # ms

    speedup = time_method1 / time_method2

    print(f"\nMethod 1 (kernel + 2x copy): {time_method1:.4f} ms")
    print(f"Method 2 (kernel in-place):  {time_method2:.4f} ms")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Savings: {time_method1 - time_method2:.4f} ms")

    if speedup > 1.05:
        print(f"\n🚀 In-place optimization is {speedup:.2f}x faster!")
    else:
        print(f"\n⚠️  Speedup is minimal ({speedup:.2f}x), copy overhead was already small")

    return True


def test_edge_cases():
    """Test edge cases for in-place version."""
    print("\n" + "=" * 60)
    print("Edge Case Tests")
    print("=" * 60)

    from sglang.srt.layers.attention.nsa.triton_metadata_kernel import (
        fill_draft_extend_metadata_inplace,
    )

    device = torch.device("cuda")

    # Edge case 1: Empty batch
    print("\n1. Empty batch (bs=0)")
    extend_seq_lens = torch.empty(0, dtype=torch.int32, device=device)
    seq_lens = torch.empty(0, dtype=torch.int32, device=device)
    out1 = torch.empty(10, dtype=torch.int32, device=device)
    out2 = torch.empty(10, dtype=torch.int32, device=device)

    total = fill_draft_extend_metadata_inplace(
        extend_seq_lens, seq_lens, 128, out1, out2
    )
    assert total == 0, "Empty batch should return 0 tokens"
    print("   ✅ Passed")

    # Edge case 2: Single token
    print("\n2. Single token")
    extend_seq_lens = torch.tensor([1], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([10], dtype=torch.int32, device=device)
    out1 = torch.empty(10, dtype=torch.int32, device=device)
    out2 = torch.empty(10, dtype=torch.int32, device=device)

    total = fill_draft_extend_metadata_inplace(
        extend_seq_lens, seq_lens, 128, out1, out2
    )
    assert total == 1, "Single token should return 1"
    assert out1[0].item() == 10, f"Expected 10, got {out1[0].item()}"
    print(f"   ✅ Passed (value={out1[0].item()})")

    # Edge case 3: Large batch
    print("\n3. Large batch (bs=32)")
    extend_seq_lens = torch.randint(1, 10, (32,), dtype=torch.int32, device=device)
    seq_lens = torch.randint(50, 200, (32,), dtype=torch.int32, device=device)
    total_expected = extend_seq_lens.sum().item()

    out1 = torch.empty(1000, dtype=torch.int32, device=device)
    out2 = torch.empty(1000, dtype=torch.int32, device=device)

    total = fill_draft_extend_metadata_inplace(
        extend_seq_lens, seq_lens, 128, out1, out2
    )
    assert total == total_expected, f"Expected {total_expected}, got {total}"
    print(f"   ✅ Passed (total_tokens={total})")

    print("\n✅ All edge cases passed!")


def main():
    print("\n" + "🔬 " * 20)
    print("In-Place Optimization Test Suite")
    print("🔬 " * 20 + "\n")

    try:
        # Test correctness and performance
        test_inplace_optimization()

        # Test edge cases
        test_edge_cases()

        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        print("\nThe in-place optimization:")
        print("  ✅ Produces identical results")
        print("  ✅ Eliminates 2x .copy_() operations")
        print("  ✅ Writes directly to metadata buffers")
        print("  ✅ Handles edge cases correctly")
        print("\n🚀 Ready for production use!")
        print("=" * 60 + "\n")

        return 0

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
