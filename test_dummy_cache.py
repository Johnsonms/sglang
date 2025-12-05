"""Test to verify dummy tensor is cached globally and reused."""

import torch
from sglang.srt.layers.attention.nsa.triton_batched_copy import _get_dummy_tuple, _DUMMY_TENSOR_CACHE


def test_dummy_tensor_caching():
    """Verify that dummy tensors are cached and reused across calls."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not torch.cuda.is_available():
        print("CUDA not available, skipping cache test")
        return

    print("Testing dummy tensor caching...")

    # Clear cache to start fresh
    _DUMMY_TENSOR_CACHE.clear()
    print(f"✓ Cache cleared: {len(_DUMMY_TENSOR_CACHE)} entries")

    # First call - should create new tensor
    tuple1 = _get_dummy_tuple(device)
    print(f"✓ First call created tuple: {id(tuple1)}")
    print(f"✓ Cache now has {len(_DUMMY_TENSOR_CACHE)} entries")
    print(f"✓ Tensor data pointer: {tuple1[0].data_ptr()}")

    # Second call - should reuse cached tensor
    tuple2 = _get_dummy_tuple(device)
    print(f"✓ Second call got tuple: {id(tuple2)}")
    print(f"✓ Cache still has {len(_DUMMY_TENSOR_CACHE)} entries")
    print(f"✓ Tensor data pointer: {tuple2[0].data_ptr()}")

    # Verify they're the same object
    assert tuple1 is tuple2, "Tuples should be identical objects!"
    assert tuple1[0].data_ptr() == tuple2[0].data_ptr(), "Tensors should share same memory!"

    print("\n✓ SUCCESS: Dummy tensor is cached globally and reused!")
    print(f"✓ Same tuple object: {tuple1 is tuple2}")
    print(f"✓ Same tensor memory: {tuple1[0].data_ptr() == tuple2[0].data_ptr()}")
    print(f"✓ Cache entries: {len(_DUMMY_TENSOR_CACHE)}")

    # Third call - still the same
    tuple3 = _get_dummy_tuple(device)
    assert tuple1 is tuple3, "Should still be the same cached tuple!"
    print(f"✓ Third call also reused cached tuple!")

    print("\n✓ All tests passed! Dummy tensor is properly cached and reused globally.")


if __name__ == "__main__":
    test_dummy_tensor_caching()
