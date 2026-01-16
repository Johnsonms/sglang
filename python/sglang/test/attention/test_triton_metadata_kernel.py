"""
Test for Triton fused metadata kernel.
Compares Triton implementation against the original Python implementation.
"""

import torch
import pytest

from sglang.srt.layers.attention.nsa.triton_metadata_kernel import (
    fill_draft_extend_metadata_fused_simple,
)


def compute_seqlens_expanded_reference(extend_seq_lens_cpu, seq_lens_cpu):
    """Original Python implementation from nsa_backend.py:1007-1021"""
    device = torch.device("cuda")
    seqlens_expanded = torch.cat(
        [
            torch.arange(
                kv_len - qo_len + 1,
                kv_len + 1,
                dtype=torch.int32,
                device=device,
            )
            for qo_len, kv_len in zip(
                extend_seq_lens_cpu,
                seq_lens_cpu,
            )
        ]
    )
    return seqlens_expanded


def compute_nsa_cache_seqlens_reference(seqlens_expanded, nsa_index_topk):
    """Original implementation: compute_nsa_seqlens"""
    return seqlens_expanded.clamp(max=nsa_index_topk)


@pytest.mark.parametrize("bs", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("nsa_index_topk", [128, 256, 512])
def test_triton_kernel_correctness(bs, nsa_index_topk):
    """Test that Triton kernel produces identical results to Python implementation."""
    device = torch.device("cuda")

    # Generate random test data
    torch.manual_seed(42 + bs)
    extend_seq_lens = torch.randint(1, 10, (bs,), dtype=torch.int32, device=device)
    seq_lens = torch.randint(50, 200, (bs,), dtype=torch.int32, device=device)

    # Ensure seq_lens >= extend_seq_lens
    seq_lens = torch.maximum(seq_lens, extend_seq_lens)

    # Reference implementation (original Python code)
    extend_seq_lens_cpu = extend_seq_lens.tolist()
    seq_lens_cpu = seq_lens.tolist()

    ref_seqlens_expanded = compute_seqlens_expanded_reference(
        extend_seq_lens_cpu, seq_lens_cpu
    )
    ref_nsa_cache_seqlens = compute_nsa_cache_seqlens_reference(
        ref_seqlens_expanded, nsa_index_topk
    )

    # Triton implementation
    triton_seqlens_expanded, triton_nsa_cache_seqlens = (
        fill_draft_extend_metadata_fused_simple(
            extend_seq_lens, seq_lens, nsa_index_topk
        )
    )

    # Compare results
    assert torch.equal(
        ref_seqlens_expanded, triton_seqlens_expanded
    ), f"seqlens_expanded mismatch:\nRef: {ref_seqlens_expanded}\nTriton: {triton_seqlens_expanded}"

    assert torch.equal(
        ref_nsa_cache_seqlens, triton_nsa_cache_seqlens
    ), f"nsa_cache_seqlens mismatch:\nRef: {ref_nsa_cache_seqlens}\nTriton: {triton_nsa_cache_seqlens}"

    print(f"✓ Test passed: bs={bs}, topk={nsa_index_topk}")


def test_edge_cases():
    """Test edge cases."""
    device = torch.device("cuda")

    # Case 1: Single batch, single token
    extend_seq_lens = torch.tensor([1], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([10], dtype=torch.int32, device=device)
    nsa_index_topk = 128

    ref = compute_seqlens_expanded_reference([1], [10])
    triton_result, _ = fill_draft_extend_metadata_fused_simple(
        extend_seq_lens, seq_lens, nsa_index_topk
    )

    assert torch.equal(ref, triton_result), "Single batch test failed"
    print("✓ Edge case 1 passed: single batch")

    # Case 2: Large extend_seq_lens
    extend_seq_lens = torch.tensor([32, 16], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([100, 80], dtype=torch.int32, device=device)

    ref = compute_seqlens_expanded_reference([32, 16], [100, 80])
    triton_result, _ = fill_draft_extend_metadata_fused_simple(
        extend_seq_lens, seq_lens, nsa_index_topk
    )

    assert torch.equal(ref, triton_result), "Large extend test failed"
    print("✓ Edge case 2 passed: large extend_seq_lens")

    # Case 3: Test clamping behavior
    extend_seq_lens = torch.tensor([5], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([200], dtype=torch.int32, device=device)
    nsa_index_topk = 100

    _, nsa_clamped = fill_draft_extend_metadata_fused_simple(
        extend_seq_lens, seq_lens, nsa_index_topk
    )

    # Should clamp values > 100 to 100
    assert nsa_clamped.max().item() == 100, "Clamping test failed"
    print("✓ Edge case 3 passed: clamping behavior")


def benchmark_performance():
    """Benchmark Triton kernel vs Python implementation."""
    import time

    device = torch.device("cuda")
    bs = 32
    nsa_index_topk = 256

    # Generate larger test case
    extend_seq_lens = torch.randint(5, 20, (bs,), dtype=torch.int32, device=device)
    seq_lens = torch.randint(100, 500, (bs,), dtype=torch.int32, device=device)

    # Warmup
    for _ in range(10):
        extend_seq_lens_cpu = extend_seq_lens.tolist()
        seq_lens_cpu = seq_lens.tolist()
        _ = compute_seqlens_expanded_reference(extend_seq_lens_cpu, seq_lens_cpu)

    for _ in range(10):
        _ = fill_draft_extend_metadata_fused_simple(
            extend_seq_lens, seq_lens, nsa_index_topk
        )

    # Benchmark Python implementation
    torch.cuda.synchronize()
    n_iters = 1000
    start = time.time()
    for _ in range(n_iters):
        extend_seq_lens_cpu = extend_seq_lens.tolist()
        seq_lens_cpu = seq_lens.tolist()
        ref = compute_seqlens_expanded_reference(extend_seq_lens_cpu, seq_lens_cpu)
    torch.cuda.synchronize()
    python_time = (time.time() - start) / n_iters * 1000  # ms

    # Benchmark Triton implementation
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iters):
        triton_result, _ = fill_draft_extend_metadata_fused_simple(
            extend_seq_lens, seq_lens, nsa_index_topk
        )
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / n_iters * 1000  # ms

    speedup = python_time / triton_time

    print(f"\n{'='*60}")
    print(f"Performance Benchmark (bs={bs}, n_iters={n_iters})")
    print(f"{'='*60}")
    print(f"Python implementation: {python_time:.3f} ms")
    print(f"Triton implementation: {triton_time:.3f} ms")
    print(f"Speedup: {speedup:.2f}x")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    print("Running correctness tests...")
    print("=" * 60)

    # Run parametrized tests manually
    for bs in [1, 2, 4, 8, 16]:
        for topk in [128, 256, 512]:
            test_triton_kernel_correctness(bs, topk)

    print("\nRunning edge case tests...")
    print("=" * 60)
    test_edge_cases()

    print("\nRunning performance benchmark...")
    print("=" * 60)
    benchmark_performance()

    print("\n✅ All tests passed!")
