"""Test script for Triton batched copy kernels."""

import torch
from sglang.srt.layers.attention.nsa.triton_batched_copy import (
    batched_copy_to_3_backends_kernel,
    BatchedCopyDescriptor,
)


def test_triton_kernel_basic():
    """Test basic functionality of the Triton kernel."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    print("Testing Triton batched copy kernel...")

    # Create test data
    n_copies = 3
    sizes = [32, 64, 128]

    # Create source tensors
    src_tensors = [
        torch.arange(size, dtype=torch.int32, device=device) for size in sizes
    ]

    # Create destination tensors for 3 backends
    dst_tensors_b0 = [torch.zeros(size, dtype=torch.int32, device=device) for size in sizes]
    dst_tensors_b1 = [torch.zeros(size, dtype=torch.int32, device=device) for size in sizes]
    dst_tensors_b2 = [torch.zeros(size, dtype=torch.int32, device=device) for size in sizes]

    # Pad to 8 copies
    dummy = torch.zeros(1, dtype=torch.int32, device=device)
    while len(src_tensors) < 8:
        src_tensors.append(dummy)
        dst_tensors_b0.append(dummy)
        dst_tensors_b1.append(dummy)
        dst_tensors_b2.append(dummy)

    # Calculate grid
    max_size = max(sizes)
    block_size = 1024
    max_blocks = (max_size + block_size - 1) // block_size
    grid = (3 * n_copies, max_blocks)

    # Launch kernel
    batched_copy_to_3_backends_kernel[grid](
        # Sources
        src_tensors[0], src_tensors[1], src_tensors[2], src_tensors[3],
        src_tensors[4], src_tensors[5], src_tensors[6], src_tensors[7],
        # Sizes
        sizes[0], sizes[1], sizes[2], 1, 1, 1, 1, 1,
        # Backend 0 destinations
        dst_tensors_b0[0], dst_tensors_b0[1], dst_tensors_b0[2], dst_tensors_b0[3],
        dst_tensors_b0[4], dst_tensors_b0[5], dst_tensors_b0[6], dst_tensors_b0[7],
        # Backend 1 destinations
        dst_tensors_b1[0], dst_tensors_b1[1], dst_tensors_b1[2], dst_tensors_b1[3],
        dst_tensors_b1[4], dst_tensors_b1[5], dst_tensors_b1[6], dst_tensors_b1[7],
        # Backend 2 destinations
        dst_tensors_b2[0], dst_tensors_b2[1], dst_tensors_b2[2], dst_tensors_b2[3],
        dst_tensors_b2[4], dst_tensors_b2[5], dst_tensors_b2[6], dst_tensors_b2[7],
        n_copies=n_copies,
        BLOCK_SIZE=block_size,
    )

    torch.cuda.synchronize()

    # Verify results
    for i in range(n_copies):
        expected = torch.arange(sizes[i], dtype=torch.int32, device=device)

        assert torch.equal(dst_tensors_b0[i], expected), f"Backend 0 copy {i} failed"
        assert torch.equal(dst_tensors_b1[i], expected), f"Backend 1 copy {i} failed"
        assert torch.equal(dst_tensors_b2[i], expected), f"Backend 2 copy {i} failed"

    print("✓ All copies successful!")
    print(f"✓ Verified {n_copies} copies to 3 backends")
    print("✓ Test passed!")


if __name__ == "__main__":
    test_triton_kernel_basic()
