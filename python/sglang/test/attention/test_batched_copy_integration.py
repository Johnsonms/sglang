"""Test batched copy integration with NSA metadata."""

import torch
from sglang.srt.layers.attention.nsa.triton_batched_copy import (
    BatchedCopyDescriptor,
    batched_copy_unified,
)


def test_simple_copy():
    """Test simple contiguous copy."""
    print("Testing simple contiguous copy...")

    device = torch.device("cuda")
    src = torch.arange(100, dtype=torch.int32, device=device)
    dst = torch.zeros(100, dtype=torch.int32, device=device)

    descriptor = BatchedCopyDescriptor(src, dst)
    batched_copy_unified([descriptor])

    assert torch.all(dst == src), "Simple copy failed"
    print("✓ Simple copy passed")


def test_slice_copy():
    """Test sliced tensor copy."""
    print("\nTesting sliced tensor copy...")

    device = torch.device("cuda")

    # Create source and destination
    src_full = torch.arange(100, dtype=torch.int32, device=device)
    dst_full = torch.zeros(100, dtype=torch.int32, device=device)

    # Test slice [1:] (skip first element)
    src_slice = src_full[1:]
    dst_slice = dst_full[1:]

    print(f"  src_slice.is_contiguous(): {src_slice.is_contiguous()}")
    print(f"  dst_slice.is_contiguous(): {dst_slice.is_contiguous()}")

    if src_slice.is_contiguous() and dst_slice.is_contiguous():
        descriptor = BatchedCopyDescriptor(src_slice, dst_slice)
        batched_copy_unified([descriptor])

        assert torch.all(dst_slice == src_slice), "Slice copy failed"
        assert dst_full[0] == 0, "First element should not be modified"
        print("✓ Slice copy passed")
    else:
        print("⚠ Slices are not contiguous, skipping test")


def test_multiple_copies():
    """Test multiple copies in single kernel launch."""
    print("\nTesting multiple copies...")

    device = torch.device("cuda")

    # Create 5 pairs of tensors
    n_copies = 5
    sizes = [32, 64, 128, 256, 512]

    src_tensors = [
        torch.arange(size, dtype=torch.int32, device=device) for size in sizes
    ]
    dst_tensors = [torch.zeros(size, dtype=torch.int32, device=device) for size in sizes]

    # Create descriptors
    descriptors = [
        BatchedCopyDescriptor(src, dst)
        for src, dst in zip(src_tensors, dst_tensors)
    ]

    # Single kernel launch for all copies
    batched_copy_unified(descriptors)

    # Verify all copies
    for i, (src, dst) in enumerate(zip(src_tensors, dst_tensors)):
        assert torch.all(dst == src), f"Copy {i} failed"

    print(f"✓ All {n_copies} copies passed")


def test_2d_copy():
    """Test 2D tensor copy."""
    print("\nTesting 2D tensor copy...")

    device = torch.device("cuda")

    # Create 2D tensors
    src = torch.arange(64, dtype=torch.int32, device=device).view(8, 8)
    dst = torch.zeros(8, 8, dtype=torch.int32, device=device)

    print(f"  src.is_contiguous(): {src.is_contiguous()}")
    print(f"  dst.is_contiguous(): {dst.is_contiguous()}")

    descriptor = BatchedCopyDescriptor(src, dst)
    batched_copy_unified([descriptor])

    assert torch.all(dst == src), "2D copy failed"
    print("✓ 2D copy passed")


def test_2d_slice_copy():
    """Test 2D sliced tensor copy."""
    print("\nTesting 2D sliced tensor copy...")

    device = torch.device("cuda")

    # Create 2D tensors
    src_full = torch.arange(100, dtype=torch.int32, device=device).view(10, 10)
    dst_full = torch.zeros(10, 10, dtype=torch.int32, device=device)

    # Test slice [:5, :8] (first 5 rows, first 8 cols)
    src_slice = src_full[:5, :8]
    dst_slice = dst_full[:5, :8]

    print(f"  src_slice.is_contiguous(): {src_slice.is_contiguous()}")
    print(f"  dst_slice.is_contiguous(): {dst_slice.is_contiguous()}")

    if src_slice.is_contiguous() and dst_slice.is_contiguous():
        descriptor = BatchedCopyDescriptor(src_slice, dst_slice)
        batched_copy_unified([descriptor])

        assert torch.all(dst_slice == src_slice), "2D slice copy failed"
        print("✓ 2D slice copy passed")
    else:
        print("⚠ 2D slices are not contiguous")
        print(f"  Using PyTorch .copy_() instead")
        dst_slice.copy_(src_slice)
        assert torch.all(dst_slice == src_slice), "PyTorch copy failed"
        print("✓ PyTorch .copy_() works for non-contiguous")


if __name__ == "__main__":
    print("=" * 60)
    print("Batched Copy Integration Test")
    print("=" * 60)

    test_simple_copy()
    test_slice_copy()
    test_multiple_copies()
    test_2d_copy()
    test_2d_slice_copy()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
