"""Test seqlens_expanded optimization correctness and performance."""

import torch
import time


def test_target_verify_correctness():
    """Test that optimized target_verify produces same results as original."""
    print("=" * 60)
    print("Testing Target Verify Mode Correctness")
    print("=" * 60)

    device = torch.device("cuda")
    bs = 32
    speculative_num_draft_tokens = 5

    # Simulate input
    seq_lens = torch.randint(100, 200, (bs,), dtype=torch.int32, device=device)
    cache_seqlens = seq_lens + speculative_num_draft_tokens

    # ORIGINAL IMPLEMENTATION (for reference)
    extend_seq_lens_cpu_old = [speculative_num_draft_tokens] * bs
    seqlens_int32_cpu_old = [
        speculative_num_draft_tokens + kv_len
        for kv_len in seq_lens.cpu().tolist()
    ]
    seqlens_expanded_old = torch.cat(
        [
            torch.arange(
                kv_len - qo_len + 1,
                kv_len + 1,
                dtype=torch.int32,
                device=device,
            )
            for qo_len, kv_len in zip(
                extend_seq_lens_cpu_old,
                seqlens_int32_cpu_old,
                strict=True,
            )
        ]
    )

    # NEW OPTIMIZED IMPLEMENTATION
    qo_len = speculative_num_draft_tokens
    kv_lens = cache_seqlens

    base = torch.arange(qo_len, dtype=torch.int32, device=device)
    base_repeated = base.unsqueeze(0).expand(bs, -1).contiguous().view(-1)
    start_vals = kv_lens - qo_len + 1
    start_vals_repeated = torch.repeat_interleave(start_vals, qo_len)
    seqlens_expanded_new = start_vals_repeated + base_repeated

    # VERIFY CORRECTNESS
    assert seqlens_expanded_old.shape == seqlens_expanded_new.shape, \
        f"Shape mismatch: {seqlens_expanded_old.shape} vs {seqlens_expanded_new.shape}"

    assert torch.all(seqlens_expanded_old == seqlens_expanded_new), \
        "Values don't match!"

    print("✓ Correctness test PASSED")
    print(f"  Output shape: {seqlens_expanded_new.shape}")
    print(f"  Sample values (first 10): {seqlens_expanded_new[:10].tolist()}")
    print()


def benchmark_target_verify():
    """Benchmark target_verify optimization."""
    print("=" * 60)
    print("Benchmarking Target Verify Mode")
    print("=" * 60)

    device = torch.device("cuda")
    bs = 32
    speculative_num_draft_tokens = 5

    # Simulate input
    seq_lens = torch.randint(100, 200, (bs,), dtype=torch.int32, device=device)
    cache_seqlens = seq_lens + speculative_num_draft_tokens

    # Warmup
    for _ in range(100):
        qo_len = speculative_num_draft_tokens
        kv_lens = cache_seqlens
        base = torch.arange(qo_len, dtype=torch.int32, device=device)
        base_repeated = base.unsqueeze(0).expand(bs, -1).contiguous().view(-1)
        start_vals = kv_lens - qo_len + 1
        start_vals_repeated = torch.repeat_interleave(start_vals, qo_len)
        seqlens_expanded = start_vals_repeated + base_repeated

    # Benchmark OLD implementation
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(1000):
        extend_seq_lens_cpu = [speculative_num_draft_tokens] * bs
        seqlens_int32_cpu = [
            speculative_num_draft_tokens + kv_len
            for kv_len in seq_lens.cpu().tolist()
        ]
        seqlens_expanded_old = torch.cat(
            [
                torch.arange(
                    kv_len - qo_len + 1,
                    kv_len + 1,
                    dtype=torch.int32,
                    device=device,
                )
                for qo_len, kv_len in zip(
                    extend_seq_lens_cpu,
                    seqlens_int32_cpu,
                    strict=True,
                )
            ]
        )
    torch.cuda.synchronize()
    time_old = (time.perf_counter() - start) / 1000 * 1e6

    # Benchmark NEW implementation
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(1000):
        qo_len = speculative_num_draft_tokens
        kv_lens = cache_seqlens
        base = torch.arange(qo_len, dtype=torch.int32, device=device)
        base_repeated = base.unsqueeze(0).expand(bs, -1).contiguous().view(-1)
        start_vals = kv_lens - qo_len + 1
        start_vals_repeated = torch.repeat_interleave(start_vals, qo_len)
        seqlens_expanded_new = start_vals_repeated + base_repeated
    torch.cuda.synchronize()
    time_new = (time.perf_counter() - start) / 1000 * 1e6

    print(f"Original implementation:  {time_old:.2f} μs")
    print(f"Optimized implementation: {time_new:.2f} μs")
    print(f"Speedup:                  {time_old / time_new:.2f}x")
    print(f"Time saved:               {time_old - time_new:.2f} μs")
    print()


def test_draft_extend_correctness():
    """Test that optimized draft_extend produces same results as original."""
    print("=" * 60)
    print("Testing Draft Extend Mode Correctness")
    print("=" * 60)

    device = torch.device("cuda")
    bs = 32

    # Simulate input with variable accept lengths
    seq_lens = torch.randint(100, 200, (bs,), dtype=torch.int32, device=device)
    accept_lengths = torch.randint(1, 8, (bs,), dtype=torch.int32, device=device)
    cache_seqlens = seq_lens

    # ORIGINAL IMPLEMENTATION
    extend_seq_lens_cpu_old = accept_lengths.cpu().tolist()
    seq_lens_cpu_old = seq_lens.cpu().tolist()
    seqlens_expanded_old = torch.cat(
        [
            torch.arange(
                kv_len - qo_len + 1,
                kv_len + 1,
                dtype=torch.int32,
                device=device,
            )
            for qo_len, kv_len in zip(
                extend_seq_lens_cpu_old,
                seq_lens_cpu_old,
                strict=True,
            )
        ]
    )

    # NEW OPTIMIZED IMPLEMENTATION
    qo_lens = accept_lengths.to(torch.int32)
    kv_lens = cache_seqlens

    offsets = torch.cat([
        torch.tensor([0], device=device, dtype=torch.int32),
        torch.cumsum(qo_lens, dim=0, dtype=torch.int32)
    ])
    total_size = offsets[-1].item()

    seqlens_expanded_new = torch.empty(total_size, dtype=torch.int32, device=device)

    for i in range(bs):
        start_idx = offsets[i]
        end_idx = offsets[i + 1]
        qo_len = qo_lens[i].item()
        if qo_len > 0:
            start_val = kv_lens[i] - qo_len + 1
            seqlens_expanded_new[start_idx:end_idx] = torch.arange(
                start_val, kv_lens[i] + 1,
                dtype=torch.int32, device=device
            )

    # VERIFY CORRECTNESS
    assert seqlens_expanded_old.shape == seqlens_expanded_new.shape, \
        f"Shape mismatch: {seqlens_expanded_old.shape} vs {seqlens_expanded_new.shape}"

    assert torch.all(seqlens_expanded_old == seqlens_expanded_new), \
        "Values don't match!"

    print("✓ Correctness test PASSED")
    print(f"  Output shape: {seqlens_expanded_new.shape}")
    print(f"  Accept lengths: {accept_lengths[:5].tolist()}...")
    print(f"  Sample values (first 15): {seqlens_expanded_new[:15].tolist()}")
    print()


def benchmark_draft_extend():
    """Benchmark draft_extend optimization."""
    print("=" * 60)
    print("Benchmarking Draft Extend Mode")
    print("=" * 60)

    device = torch.device("cuda")
    bs = 32

    # Simulate input
    seq_lens = torch.randint(100, 200, (bs,), dtype=torch.int32, device=device)
    accept_lengths = torch.randint(1, 8, (bs,), dtype=torch.int32, device=device)
    cache_seqlens = seq_lens

    # Warmup
    for _ in range(100):
        qo_lens = accept_lengths.to(torch.int32)
        kv_lens = cache_seqlens
        offsets = torch.cat([
            torch.tensor([0], device=device, dtype=torch.int32),
            torch.cumsum(qo_lens, dim=0, dtype=torch.int32)
        ])
        total_size = offsets[-1].item()
        seqlens_expanded = torch.empty(total_size, dtype=torch.int32, device=device)
        for i in range(bs):
            start_idx = offsets[i]
            end_idx = offsets[i + 1]
            qo_len = qo_lens[i].item()
            if qo_len > 0:
                start_val = kv_lens[i] - qo_len + 1
                seqlens_expanded[start_idx:end_idx] = torch.arange(
                    start_val, kv_lens[i] + 1,
                    dtype=torch.int32, device=device
                )

    # Benchmark OLD implementation
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(1000):
        extend_seq_lens_cpu = accept_lengths.cpu().tolist()
        seq_lens_cpu = seq_lens.cpu().tolist()
        seqlens_expanded_old = torch.cat(
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
                    strict=True,
                )
            ]
        )
    torch.cuda.synchronize()
    time_old = (time.perf_counter() - start) / 1000 * 1e6

    # Benchmark NEW implementation
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(1000):
        qo_lens = accept_lengths.to(torch.int32)
        kv_lens = cache_seqlens
        offsets = torch.cat([
            torch.tensor([0], device=device, dtype=torch.int32),
            torch.cumsum(qo_lens, dim=0, dtype=torch.int32)
        ])
        total_size = offsets[-1].item()
        seqlens_expanded_new = torch.empty(total_size, dtype=torch.int32, device=device)
        for i in range(bs):
            start_idx = offsets[i]
            end_idx = offsets[i + 1]
            qo_len = qo_lens[i].item()
            if qo_len > 0:
                start_val = kv_lens[i] - qo_len + 1
                seqlens_expanded_new[start_idx:end_idx] = torch.arange(
                    start_val, kv_lens[i] + 1,
                    dtype=torch.int32, device=device
                )
    torch.cuda.synchronize()
    time_new = (time.perf_counter() - start) / 1000 * 1e6

    print(f"Original implementation:  {time_old:.2f} μs")
    print(f"Optimized implementation: {time_new:.2f} μs")
    print(f"Speedup:                  {time_old / time_new:.2f}x")
    print(f"Time saved:               {time_old - time_new:.2f} μs")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("seqlens_expanded Optimization Test Suite")
    print("=" * 60 + "\n")

    # Test correctness
    test_target_verify_correctness()
    test_draft_extend_correctness()

    # Benchmark performance
    benchmark_target_verify()
    benchmark_draft_extend()

    print("=" * 60)
    print("All tests completed successfully! ✅")
    print("=" * 60)
