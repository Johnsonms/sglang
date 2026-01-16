#!/usr/bin/env python3
"""
Quick integration test for Triton metadata kernel in nsa_backend.py
"""

import os
import sys

# Set environment variables before importing
os.environ["SGLANG_NSA_USE_TRITON_METADATA"] = "1"

import torch


def test_triton_import():
    """Test that Triton kernel can be imported."""
    print("=" * 60)
    print("Test 1: Import Triton Kernel")
    print("=" * 60)

    try:
        from sglang.srt.layers.attention.nsa.triton_metadata_kernel import (
            fill_draft_extend_metadata_fused_simple,
        )
        print("✅ Triton kernel imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import Triton kernel: {e}")
        return False


def test_nsa_backend_import():
    """Test that nsa_backend can be imported with Triton integration."""
    print("\n" + "=" * 60)
    print("Test 2: Import NSA Backend with Triton")
    print("=" * 60)

    try:
        from sglang.srt.layers.attention import nsa_backend

        if hasattr(nsa_backend, "TRITON_KERNEL_AVAILABLE"):
            if nsa_backend.TRITON_KERNEL_AVAILABLE:
                print("✅ NSA backend imported with Triton kernel available")
                print(f"   TRITON_KERNEL_AVAILABLE = {nsa_backend.TRITON_KERNEL_AVAILABLE}")
            else:
                print("⚠️  NSA backend imported but Triton kernel not available")
                print(f"   TRITON_KERNEL_AVAILABLE = {nsa_backend.TRITON_KERNEL_AVAILABLE}")
        else:
            print("❌ TRITON_KERNEL_AVAILABLE flag not found in nsa_backend")
            return False

        return True
    except ImportError as e:
        print(f"❌ Failed to import nsa_backend: {e}")
        return False


def test_environment_variable():
    """Test that environment variable is properly set."""
    print("\n" + "=" * 60)
    print("Test 3: Environment Variable")
    print("=" * 60)

    try:
        from sglang.srt.environ import envs

        use_triton = envs.SGLANG_NSA_USE_TRITON_METADATA.get()
        print(f"✅ Environment variable accessible")
        print(f"   SGLANG_NSA_USE_TRITON_METADATA = {use_triton}")

        if use_triton:
            print("   → Triton kernel will be used when available")
        else:
            print("   → Fallback to Python implementation")

        return True
    except AttributeError as e:
        print(f"❌ Environment variable not found: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality of the Triton kernel."""
    print("\n" + "=" * 60)
    print("Test 4: Basic Functionality")
    print("=" * 60)

    try:
        from sglang.srt.layers.attention.nsa.triton_metadata_kernel import (
            fill_draft_extend_metadata_fused_simple,
        )

        if not torch.cuda.is_available():
            print("⚠️  CUDA not available, skipping functionality test")
            return True

        device = torch.device("cuda")

        # Simple test case
        extend_seq_lens = torch.tensor([2, 3], dtype=torch.int32, device=device)
        seq_lens = torch.tensor([10, 15], dtype=torch.int32, device=device)
        nsa_index_topk = 128

        seqlens_expanded, nsa_cache_seqlens = fill_draft_extend_metadata_fused_simple(
            extend_seq_lens=extend_seq_lens,
            seq_lens=seq_lens,
            nsa_index_topk=nsa_index_topk,
        )

        # Verify output shapes
        expected_total_tokens = extend_seq_lens.sum().item()
        assert seqlens_expanded.shape[0] == expected_total_tokens, \
            f"Expected {expected_total_tokens} tokens, got {seqlens_expanded.shape[0]}"
        assert nsa_cache_seqlens.shape[0] == expected_total_tokens, \
            f"Expected {expected_total_tokens} tokens, got {nsa_cache_seqlens.shape[0]}"

        print("✅ Basic functionality test passed")
        print(f"   Input: extend_seq_lens={extend_seq_lens.tolist()}, seq_lens={seq_lens.tolist()}")
        print(f"   Output: seqlens_expanded={seqlens_expanded.tolist()}")
        print(f"   Output: nsa_cache_seqlens={nsa_cache_seqlens.tolist()}")

        return True
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "🚀 " * 20)
    print("Triton Metadata Kernel Integration Test")
    print("🚀 " * 20 + "\n")

    results = []

    # Run tests
    results.append(("Triton Import", test_triton_import()))
    results.append(("NSA Backend Import", test_nsa_backend_import()))
    results.append(("Environment Variable", test_environment_variable()))
    results.append(("Basic Functionality", test_basic_functionality()))

    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(result[1] for result in results)

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed! Integration successful!")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    print("=" * 60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
