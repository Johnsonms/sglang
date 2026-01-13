/*
 * Fused metadata copy kernel for NSA backend CUDA graph replay.
 *
 * This kernel fuses multiple tensor copy operations into a single kernel launch,
 * reducing kernel launch overhead and improving CUDA graph replay performance.
 *
 * Three specialized kernels avoid branch divergence in GPU.
 */

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "pytorch_extension_utils.h"

// Forward mode enum (must match Python ForwardMode)
enum ForwardModeEnum {
    DECODE = 0,
    TARGET_VERIFY = 1,
    DRAFT_EXTEND = 2
};

/**
 * Specialized kernel for DECODE mode - optimized for single token decode.
 * Template parameters eliminate runtime branches.
 */
template<bool HAS_REAL_PAGE_TABLE, bool HAS_FLASHMLA>
__global__ void fused_metadata_copy_decode_kernel(
    const int32_t* __restrict__ cache_seqlens_src,
    const int32_t* __restrict__ cu_seqlens_k_src,
    const int32_t* __restrict__ page_indices_src,
    const int32_t* __restrict__ nsa_cache_seqlens_src,
    const int32_t* __restrict__ nsa_cu_seqlens_k_src,
    const int32_t* __restrict__ real_page_table_src,
    const int32_t* __restrict__ flashmla_num_splits_src,
    const int32_t* __restrict__ flashmla_metadata_src,

    int32_t* __restrict__ cache_seqlens_dst,
    int32_t* __restrict__ cu_seqlens_k_dst,
    int32_t* __restrict__ page_table_1_dst,
    int32_t* __restrict__ nsa_cache_seqlens_dst,
    int32_t* __restrict__ nsa_cu_seqlens_k_dst,
    int32_t* __restrict__ real_page_table_dst,
    int32_t* __restrict__ flashmla_num_splits_dst,
    int32_t* __restrict__ flashmla_metadata_dst,

    int bs,
    int max_len,
    int seqlens_expanded_size,
    int page_table_1_stride,
    int real_page_table_cols,
    int real_page_table_dst_stride,
    int flashmla_metadata_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    if (tid == 0 && blockIdx.x == 0) {
        printf("===== DECODE KERNEL START: HAS_REAL=%d, HAS_FLASHMLA=%d =====\n",
               (int)HAS_REAL_PAGE_TABLE, (int)HAS_FLASHMLA);
        printf("bs=%d, max_len=%d, real_cols=%d, real_dst_stride=%d\n",
               bs, max_len, real_page_table_cols, real_page_table_dst_stride);
    }

    // Copy cache_seqlens (bs elements)
    #pragma unroll 8
    for (int i = tid; i < bs; i += total_threads) {
        cache_seqlens_dst[i] = cache_seqlens_src[i];
    }

    // Copy cu_seqlens_k (skip first element)
    #pragma unroll 8
    for (int i = tid; i < bs; i += total_threads) {
        cu_seqlens_k_dst[i + 1] = cu_seqlens_k_src[i + 1];
    }

    // DECODE mode: copy page_table_1 and nsa_cache_seqlens
    int page_table_elements = bs * max_len;
    #pragma unroll 4
    for (int i = tid; i < page_table_elements; i += total_threads) {
        int row = i / max_len;
        int col = i % max_len;
        page_table_1_dst[row * page_table_1_stride + col] = page_indices_src[i];
    }

    #pragma unroll 8
    for (int i = tid; i < bs; i += total_threads) {
        nsa_cache_seqlens_dst[i] = nsa_cache_seqlens_src[i];
    }

    // Copy NSA cu_seqlens (in decode mode, size == bs)
    #pragma unroll 8
    for (int i = tid; i < bs; i += total_threads) {
        nsa_cu_seqlens_k_dst[i + 1] = nsa_cu_seqlens_k_src[i + 1];
    }

    // Copy real page table - compile-time branch, respecting both source and destination stride
    // TEMPORARY DEBUG: Always execute this regardless of HAS_REAL_PAGE_TABLE
    //if constexpr (HAS_REAL_PAGE_TABLE) {
    if (real_page_table_src != nullptr && real_page_table_dst != nullptr) {
        int real_table_elements = bs * real_page_table_cols;
        // Debug: print once from first thread
        if (tid == 0) {
            printf("CUDA Kernel DECODE: bs=%d, real_page_table_cols=%d, real_page_table_dst_stride=%d, elements=%d\n",
                   bs, real_page_table_cols, real_page_table_dst_stride, real_table_elements);
            // Print first few values from source
            printf("Source[0]=%d, Source[64]=%d, Source[128]=%d\n",
                   real_page_table_src[0], real_page_table_src[64], real_page_table_src[128]);
        }
        #pragma unroll 2
        for (int i = tid; i < real_table_elements; i += total_threads) {
            int row = i / real_page_table_cols;
            int col = i % real_page_table_cols;
            int src_idx = row * real_page_table_cols + col;

            // TEMPORARY DEBUG: Check if stride value is correct
            int test_dst_stride = real_page_table_dst_stride;
            if (tid == 0 && i == 0) {
                printf("KERNEL: real_page_table_cols=%d, real_page_table_dst_stride=%d\n",
                       real_page_table_cols, test_dst_stride);
            }

            int dst_idx = row * real_page_table_dst_stride + col;
            real_page_table_dst[dst_idx] = real_page_table_src[src_idx];
        }
    }

    // Copy FlashMLA num_splits and metadata - compile-time branch
    if constexpr (HAS_FLASHMLA) {
        int flashmla_size = bs + 1;
        #pragma unroll 8
        for (int i = tid; i < flashmla_size; i += total_threads) {
            flashmla_num_splits_dst[i] = flashmla_num_splits_src[i];
        }

        // Copy flashmla_metadata tensor (tile scheduler metadata)
        #pragma unroll 2
        for (int i = tid; i < flashmla_metadata_size; i += total_threads) {
            flashmla_metadata_dst[i] = flashmla_metadata_src[i];
        }
    }
}

/**
 * Specialized kernel for TARGET_VERIFY mode - optimized for speculative verification.
 * Template parameters eliminate runtime branches.
 */
template<bool HAS_REAL_PAGE_TABLE, bool HAS_FLASHMLA>
__global__ void fused_metadata_copy_target_verify_kernel(
    const int32_t* __restrict__ cache_seqlens_src,
    const int32_t* __restrict__ cu_seqlens_k_src,
    const int32_t* __restrict__ page_indices_src,
    const int32_t* __restrict__ nsa_cache_seqlens_src,
    const int32_t* __restrict__ seqlens_expanded_src,
    const int32_t* __restrict__ nsa_cu_seqlens_k_src,
    const int32_t* __restrict__ real_page_table_src,
    const int32_t* __restrict__ flashmla_num_splits_src,
    const int32_t* __restrict__ flashmla_metadata_src,

    int32_t* __restrict__ cache_seqlens_dst,
    int32_t* __restrict__ cu_seqlens_k_dst,
    int32_t* __restrict__ page_table_1_dst,
    int32_t* __restrict__ nsa_cache_seqlens_dst,
    int32_t* __restrict__ seqlens_expanded_dst,
    int32_t* __restrict__ nsa_cu_seqlens_k_dst,
    int32_t* __restrict__ real_page_table_dst,
    int32_t* __restrict__ flashmla_num_splits_dst,
    int32_t* __restrict__ flashmla_metadata_dst,

    int bs,
    int max_seqlen_k,
    int seqlens_expanded_size,
    int page_indices_rows,
    int page_table_1_stride,
    int real_page_table_cols,
    int real_page_table_dst_stride,
    int flashmla_metadata_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    // Copy cache_seqlens (bs elements)
    #pragma unroll 8
    for (int i = tid; i < bs; i += total_threads) {
        cache_seqlens_dst[i] = cache_seqlens_src[i];
    }

    // Copy cu_seqlens_k (skip first element)
    #pragma unroll 8
    for (int i = tid; i < bs; i += total_threads) {
        cu_seqlens_k_dst[i + 1] = cu_seqlens_k_src[i + 1];
    }

    // TARGET_VERIFY mode: copy page_table, seqlens_expanded, and nsa_cache_seqlens
    int page_table_elements = page_indices_rows * max_seqlen_k;
    #pragma unroll 4
    for (int i = tid; i < page_table_elements; i += total_threads) {
        int row = i / max_seqlen_k;
        int col = i % max_seqlen_k;
        page_table_1_dst[row * page_table_1_stride + col] = page_indices_src[i];
    }

    #pragma unroll 4
    for (int i = tid; i < seqlens_expanded_size; i += total_threads) {
        seqlens_expanded_dst[i] = seqlens_expanded_src[i];
    }

    #pragma unroll 4
    for (int i = tid; i < seqlens_expanded_size; i += total_threads) {
        nsa_cache_seqlens_dst[i] = nsa_cache_seqlens_src[i];
    }

    // Copy NSA cu_seqlens
    #pragma unroll 4
    for (int i = tid; i < seqlens_expanded_size; i += total_threads) {
        nsa_cu_seqlens_k_dst[i + 1] = nsa_cu_seqlens_k_src[i + 1];
    }

    // Copy real page table - compile-time branch, respecting both source and destination stride
    if constexpr (HAS_REAL_PAGE_TABLE) {
        int real_table_elements = page_indices_rows * real_page_table_cols;
        #pragma unroll 2
        for (int i = tid; i < real_table_elements; i += total_threads) {
            int row = i / real_page_table_cols;
            int col = i % real_page_table_cols;
            real_page_table_dst[row * real_page_table_dst_stride + col] =
                real_page_table_src[row * real_page_table_cols + col];
        }
    }

    // Copy FlashMLA num_splits and metadata - compile-time branch
    if constexpr (HAS_FLASHMLA) {
        int flashmla_size = seqlens_expanded_size + 1;
        #pragma unroll 4
        for (int i = tid; i < flashmla_size; i += total_threads) {
            flashmla_num_splits_dst[i] = flashmla_num_splits_src[i];
        }

        // Copy flashmla_metadata tensor (tile scheduler metadata)
        #pragma unroll 2
        for (int i = tid; i < flashmla_metadata_size; i += total_threads) {
            flashmla_metadata_dst[i] = flashmla_metadata_src[i];
        }
    }
}

/**
 * Specialized kernel for DRAFT_EXTEND mode - optimized for draft token generation.
 * Template parameters eliminate runtime branches.
 */
template<bool HAS_REAL_PAGE_TABLE, bool HAS_FLASHMLA>
__global__ void fused_metadata_copy_draft_extend_kernel(
    const int32_t* __restrict__ cache_seqlens_src,
    const int32_t* __restrict__ cu_seqlens_k_src,
    const int32_t* __restrict__ page_indices_src,
    const int32_t* __restrict__ nsa_cache_seqlens_src,
    const int32_t* __restrict__ seqlens_expanded_src,
    const int32_t* __restrict__ nsa_cu_seqlens_k_src,
    const int32_t* __restrict__ real_page_table_src,
    const int32_t* __restrict__ flashmla_num_splits_src,
    const int32_t* __restrict__ flashmla_metadata_src,

    int32_t* __restrict__ cache_seqlens_dst,
    int32_t* __restrict__ cu_seqlens_k_dst,
    int32_t* __restrict__ page_table_1_dst,
    int32_t* __restrict__ nsa_cache_seqlens_dst,
    int32_t* __restrict__ seqlens_expanded_dst,
    int32_t* __restrict__ nsa_cu_seqlens_k_dst,
    int32_t* __restrict__ real_page_table_dst,
    int32_t* __restrict__ flashmla_num_splits_dst,
    int32_t* __restrict__ flashmla_metadata_dst,

    int bs,
    int max_seqlen_k,
    int seqlens_expanded_size,
    int page_indices_rows,
    int page_table_1_stride,
    int real_page_table_cols,
    int real_page_table_dst_stride,
    int flashmla_metadata_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    // Copy cache_seqlens (bs elements)
    #pragma unroll 8
    for (int i = tid; i < bs; i += total_threads) {
        cache_seqlens_dst[i] = cache_seqlens_src[i];
    }

    // Copy cu_seqlens_k (skip first element)
    #pragma unroll 8
    for (int i = tid; i < bs; i += total_threads) {
        cu_seqlens_k_dst[i + 1] = cu_seqlens_k_src[i + 1];
    }

    // DRAFT_EXTEND mode: copy page_table, seqlens_expanded, and nsa_cache_seqlens
    int page_table_elements = page_indices_rows * max_seqlen_k;
    #pragma unroll 4
    for (int i = tid; i < page_table_elements; i += total_threads) {
        int row = i / max_seqlen_k;
        int col = i % max_seqlen_k;
        page_table_1_dst[row * page_table_1_stride + col] = page_indices_src[i];
    }

    #pragma unroll 4
    for (int i = tid; i < seqlens_expanded_size; i += total_threads) {
        seqlens_expanded_dst[i] = seqlens_expanded_src[i];
    }

    #pragma unroll 4
    for (int i = tid; i < seqlens_expanded_size; i += total_threads) {
        nsa_cache_seqlens_dst[i] = nsa_cache_seqlens_src[i];
    }

    // Copy NSA cu_seqlens
    #pragma unroll 4
    for (int i = tid; i < seqlens_expanded_size; i += total_threads) {
        nsa_cu_seqlens_k_dst[i + 1] = nsa_cu_seqlens_k_src[i + 1];
    }

    // Copy real page table - compile-time branch, respecting both source and destination stride
    if constexpr (HAS_REAL_PAGE_TABLE) {
        int real_table_elements = page_indices_rows * real_page_table_cols;
        #pragma unroll 2
        for (int i = tid; i < real_table_elements; i += total_threads) {
            int row = i / real_page_table_cols;
            int col = i % real_page_table_cols;
            real_page_table_dst[row * real_page_table_dst_stride + col] =
                real_page_table_src[row * real_page_table_cols + col];
        }
    }

    // Copy FlashMLA num_splits and metadata - compile-time branch
    if constexpr (HAS_FLASHMLA) {
        int flashmla_size = seqlens_expanded_size + 1;
        #pragma unroll 4
        for (int i = tid; i < flashmla_size; i += total_threads) {
            flashmla_num_splits_dst[i] = flashmla_num_splits_src[i];
        }

        // Copy flashmla_metadata tensor (tile scheduler metadata)
        #pragma unroll 2
        for (int i = tid; i < flashmla_metadata_size; i += total_threads) {
            flashmla_metadata_dst[i] = flashmla_metadata_src[i];
        }
    }
}

/**
 * PyTorch wrapper function for fused metadata copy.
 * Dispatches to the appropriate specialized kernel based on forward_mode.
 */
void fused_metadata_copy_cuda(
    at::Tensor cache_seqlens_src,
    at::Tensor cu_seqlens_k_src,
    at::Tensor page_indices_src,
    at::Tensor nsa_cache_seqlens_src,
    at::Tensor seqlens_expanded_src,
    at::Tensor nsa_cu_seqlens_k_src,
    c10::optional<at::Tensor> real_page_table_src,
    c10::optional<at::Tensor> flashmla_num_splits_src,
    c10::optional<at::Tensor> flashmla_metadata_src,
    at::Tensor cache_seqlens_dst,
    at::Tensor cu_seqlens_k_dst,
    at::Tensor page_table_1_dst,
    at::Tensor nsa_cache_seqlens_dst,
    at::Tensor seqlens_expanded_dst,
    at::Tensor nsa_cu_seqlens_k_dst,
    c10::optional<at::Tensor> real_page_table_dst,
    c10::optional<at::Tensor> flashmla_num_splits_dst,
    c10::optional<at::Tensor> flashmla_metadata_dst,
    int64_t forward_mode,
    int64_t bs,
    int64_t max_len,
    int64_t max_seqlen_k,
    int64_t seqlens_expanded_size
) {
    // Validate inputs
    CHECK_INPUT(cache_seqlens_src);
    CHECK_INPUT(cu_seqlens_k_src);
    CHECK_INPUT(page_indices_src);
    CHECK_INPUT(nsa_cache_seqlens_src);
    CHECK_INPUT(seqlens_expanded_src);
    CHECK_INPUT(nsa_cu_seqlens_k_src);

    // Calculate dimensions
    int page_indices_rows = page_indices_src.size(0);
    int page_table_1_stride = page_table_1_dst.size(1);

    int real_page_table_cols = 0;
    int real_page_table_dst_stride = 0;
    bool has_real_page_table = false;

    const int32_t* real_table_src_ptr = nullptr;
    int32_t* real_table_dst_ptr = nullptr;

    fprintf(stderr, "=== ENTER fused_metadata_copy_cuda, forward_mode=%ld, bs=%ld ===\n",
            forward_mode, bs);
    fprintf(stderr, "real_page_table_src.has_value()=%d, real_page_table_dst.has_value()=%d\n",
            real_page_table_src.has_value(), real_page_table_dst.has_value());
    fflush(stderr);

    if (real_page_table_src.has_value() && real_page_table_dst.has_value()) {
        has_real_page_table = true;
        real_page_table_cols = real_page_table_src.value().size(1);
        // Use stride(0) which gives the row stride for a contiguous tensor
        real_page_table_dst_stride = real_page_table_dst.value().stride(0);
        real_table_src_ptr = real_page_table_src.value().data_ptr<int32_t>();
        real_table_dst_ptr = real_page_table_dst.value().data_ptr<int32_t>();
    }

    bool has_flashmla_metadata = false;
    const int32_t* flashmla_num_splits_src_ptr = nullptr;
    int32_t* flashmla_num_splits_dst_ptr = nullptr;
    const int32_t* flashmla_metadata_src_ptr = nullptr;
    int32_t* flashmla_metadata_dst_ptr = nullptr;
    int flashmla_metadata_size = 0;

    if (flashmla_num_splits_src.has_value() && flashmla_num_splits_dst.has_value() &&
        flashmla_metadata_src.has_value() && flashmla_metadata_dst.has_value()) {
        has_flashmla_metadata = true;
        flashmla_num_splits_src_ptr = flashmla_num_splits_src.value().data_ptr<int32_t>();
        flashmla_num_splits_dst_ptr = flashmla_num_splits_dst.value().data_ptr<int32_t>();
        flashmla_metadata_src_ptr = flashmla_metadata_src.value().data_ptr<int32_t>();
        flashmla_metadata_dst_ptr = flashmla_metadata_dst.value().data_ptr<int32_t>();
        flashmla_metadata_size = flashmla_metadata_src.value().numel();
    }

    // Launch configuration
    int threads_per_block = 256;
    int max_elements = std::max({
        static_cast<int>(bs),
        page_indices_rows * static_cast<int>(max_seqlen_k),
        static_cast<int>(seqlens_expanded_size),
        has_flashmla_metadata ? static_cast<int>(seqlens_expanded_size + 1) : 0,
        has_flashmla_metadata ? flashmla_metadata_size : 0
    });
    int num_blocks = (max_elements + threads_per_block - 1) / threads_per_block;
    num_blocks = std::min(num_blocks, 1024);  // Cap at 1024 blocks

    // Get current CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Dispatch to specialized kernel based on forward_mode and optional tensors
    // All branching is done on CPU, GPU kernels have no runtime branches
    if (forward_mode == DECODE) {
        if (has_real_page_table && has_flashmla_metadata) {
            fused_metadata_copy_decode_kernel<true, true><<<num_blocks, threads_per_block, 0, stream>>>(
                cache_seqlens_src.data_ptr<int32_t>(),
                cu_seqlens_k_src.data_ptr<int32_t>(),
                page_indices_src.data_ptr<int32_t>(),
                nsa_cache_seqlens_src.data_ptr<int32_t>(),
                nsa_cu_seqlens_k_src.data_ptr<int32_t>(),
                real_table_src_ptr,
                flashmla_num_splits_src_ptr,
                flashmla_metadata_src_ptr,
                cache_seqlens_dst.data_ptr<int32_t>(),
                cu_seqlens_k_dst.data_ptr<int32_t>(),
                page_table_1_dst.data_ptr<int32_t>(),
                nsa_cache_seqlens_dst.data_ptr<int32_t>(),
                nsa_cu_seqlens_k_dst.data_ptr<int32_t>(),
                real_table_dst_ptr,
                flashmla_num_splits_dst_ptr,
                flashmla_metadata_dst_ptr,
                static_cast<int>(bs),
                static_cast<int>(max_len),
                static_cast<int>(seqlens_expanded_size),
                page_table_1_stride,
                real_page_table_cols,
                real_page_table_dst_stride,
                flashmla_metadata_size
            );
        } else if (has_real_page_table) {
            fused_metadata_copy_decode_kernel<true, false><<<num_blocks, threads_per_block, 0, stream>>>(
                cache_seqlens_src.data_ptr<int32_t>(),
                cu_seqlens_k_src.data_ptr<int32_t>(),
                page_indices_src.data_ptr<int32_t>(),
                nsa_cache_seqlens_src.data_ptr<int32_t>(),
                nsa_cu_seqlens_k_src.data_ptr<int32_t>(),
                real_table_src_ptr,
                flashmla_num_splits_src_ptr,
                flashmla_metadata_src_ptr,
                cache_seqlens_dst.data_ptr<int32_t>(),
                cu_seqlens_k_dst.data_ptr<int32_t>(),
                page_table_1_dst.data_ptr<int32_t>(),
                nsa_cache_seqlens_dst.data_ptr<int32_t>(),
                nsa_cu_seqlens_k_dst.data_ptr<int32_t>(),
                real_table_dst_ptr,
                flashmla_num_splits_dst_ptr,
                flashmla_metadata_dst_ptr,
                static_cast<int>(bs),
                static_cast<int>(max_len),
                static_cast<int>(seqlens_expanded_size),
                page_table_1_stride,
                real_page_table_cols,
                real_page_table_dst_stride,
                flashmla_metadata_size
            );
        } else if (has_flashmla_metadata) {
            fused_metadata_copy_decode_kernel<false, true><<<num_blocks, threads_per_block, 0, stream>>>(
                cache_seqlens_src.data_ptr<int32_t>(),
                cu_seqlens_k_src.data_ptr<int32_t>(),
                page_indices_src.data_ptr<int32_t>(),
                nsa_cache_seqlens_src.data_ptr<int32_t>(),
                nsa_cu_seqlens_k_src.data_ptr<int32_t>(),
                real_table_src_ptr,
                flashmla_num_splits_src_ptr,
                flashmla_metadata_src_ptr,
                cache_seqlens_dst.data_ptr<int32_t>(),
                cu_seqlens_k_dst.data_ptr<int32_t>(),
                page_table_1_dst.data_ptr<int32_t>(),
                nsa_cache_seqlens_dst.data_ptr<int32_t>(),
                nsa_cu_seqlens_k_dst.data_ptr<int32_t>(),
                real_table_dst_ptr,
                flashmla_num_splits_dst_ptr,
                flashmla_metadata_dst_ptr,
                static_cast<int>(bs),
                static_cast<int>(max_len),
                static_cast<int>(seqlens_expanded_size),
                page_table_1_stride,
                real_page_table_cols,
                real_page_table_dst_stride,
                flashmla_metadata_size
            );
        } else {
            fused_metadata_copy_decode_kernel<false, false><<<num_blocks, threads_per_block, 0, stream>>>(
                cache_seqlens_src.data_ptr<int32_t>(),
                cu_seqlens_k_src.data_ptr<int32_t>(),
                page_indices_src.data_ptr<int32_t>(),
                nsa_cache_seqlens_src.data_ptr<int32_t>(),
                nsa_cu_seqlens_k_src.data_ptr<int32_t>(),
                real_table_src_ptr,
                flashmla_num_splits_src_ptr,
                flashmla_metadata_src_ptr,
                cache_seqlens_dst.data_ptr<int32_t>(),
                cu_seqlens_k_dst.data_ptr<int32_t>(),
                page_table_1_dst.data_ptr<int32_t>(),
                nsa_cache_seqlens_dst.data_ptr<int32_t>(),
                nsa_cu_seqlens_k_dst.data_ptr<int32_t>(),
                real_table_dst_ptr,
                flashmla_num_splits_dst_ptr,
                flashmla_metadata_dst_ptr,
                static_cast<int>(bs),
                static_cast<int>(max_len),
                static_cast<int>(seqlens_expanded_size),
                page_table_1_stride,
                real_page_table_cols,
                real_page_table_dst_stride,
                flashmla_metadata_size
            );
        }

        // Debug: Check for CUDA errors and synchronize to flush printf
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        }
        cudaDeviceSynchronize();  // Force synchronization to flush printf output
    } else if (forward_mode == TARGET_VERIFY) {
        if (has_real_page_table && has_flashmla_metadata) {
            fused_metadata_copy_target_verify_kernel<true, true><<<num_blocks, threads_per_block, 0, stream>>>(
                cache_seqlens_src.data_ptr<int32_t>(),
                cu_seqlens_k_src.data_ptr<int32_t>(),
                page_indices_src.data_ptr<int32_t>(),
                nsa_cache_seqlens_src.data_ptr<int32_t>(),
                seqlens_expanded_src.data_ptr<int32_t>(),
                nsa_cu_seqlens_k_src.data_ptr<int32_t>(),
                real_table_src_ptr,
                flashmla_num_splits_src_ptr,
                flashmla_metadata_src_ptr,
                cache_seqlens_dst.data_ptr<int32_t>(),
                cu_seqlens_k_dst.data_ptr<int32_t>(),
                page_table_1_dst.data_ptr<int32_t>(),
                nsa_cache_seqlens_dst.data_ptr<int32_t>(),
                seqlens_expanded_dst.data_ptr<int32_t>(),
                nsa_cu_seqlens_k_dst.data_ptr<int32_t>(),
                real_table_dst_ptr,
                flashmla_num_splits_dst_ptr,
                flashmla_metadata_dst_ptr,
                static_cast<int>(bs),
                static_cast<int>(max_seqlen_k),
                static_cast<int>(seqlens_expanded_size),
                page_indices_rows,
                page_table_1_stride,
                real_page_table_cols,
                real_page_table_dst_stride,
                flashmla_metadata_size
            );
        } else if (has_real_page_table) {
            fused_metadata_copy_target_verify_kernel<true, false><<<num_blocks, threads_per_block, 0, stream>>>(
                cache_seqlens_src.data_ptr<int32_t>(),
                cu_seqlens_k_src.data_ptr<int32_t>(),
                page_indices_src.data_ptr<int32_t>(),
                nsa_cache_seqlens_src.data_ptr<int32_t>(),
                seqlens_expanded_src.data_ptr<int32_t>(),
                nsa_cu_seqlens_k_src.data_ptr<int32_t>(),
                real_table_src_ptr,
                flashmla_num_splits_src_ptr,
                flashmla_metadata_src_ptr,
                cache_seqlens_dst.data_ptr<int32_t>(),
                cu_seqlens_k_dst.data_ptr<int32_t>(),
                page_table_1_dst.data_ptr<int32_t>(),
                nsa_cache_seqlens_dst.data_ptr<int32_t>(),
                seqlens_expanded_dst.data_ptr<int32_t>(),
                nsa_cu_seqlens_k_dst.data_ptr<int32_t>(),
                real_table_dst_ptr,
                flashmla_num_splits_dst_ptr,
                flashmla_metadata_dst_ptr,
                static_cast<int>(bs),
                static_cast<int>(max_seqlen_k),
                static_cast<int>(seqlens_expanded_size),
                page_indices_rows,
                page_table_1_stride,
                real_page_table_cols,
                real_page_table_dst_stride,
                flashmla_metadata_size
            );
        } else if (has_flashmla_metadata) {
            fused_metadata_copy_target_verify_kernel<false, true><<<num_blocks, threads_per_block, 0, stream>>>(
                cache_seqlens_src.data_ptr<int32_t>(),
                cu_seqlens_k_src.data_ptr<int32_t>(),
                page_indices_src.data_ptr<int32_t>(),
                nsa_cache_seqlens_src.data_ptr<int32_t>(),
                seqlens_expanded_src.data_ptr<int32_t>(),
                nsa_cu_seqlens_k_src.data_ptr<int32_t>(),
                real_table_src_ptr,
                flashmla_num_splits_src_ptr,
                flashmla_metadata_src_ptr,
                cache_seqlens_dst.data_ptr<int32_t>(),
                cu_seqlens_k_dst.data_ptr<int32_t>(),
                page_table_1_dst.data_ptr<int32_t>(),
                nsa_cache_seqlens_dst.data_ptr<int32_t>(),
                seqlens_expanded_dst.data_ptr<int32_t>(),
                nsa_cu_seqlens_k_dst.data_ptr<int32_t>(),
                real_table_dst_ptr,
                flashmla_num_splits_dst_ptr,
                flashmla_metadata_dst_ptr,
                static_cast<int>(bs),
                static_cast<int>(max_seqlen_k),
                static_cast<int>(seqlens_expanded_size),
                page_indices_rows,
                page_table_1_stride,
                real_page_table_cols,
                real_page_table_dst_stride,
                flashmla_metadata_size
            );
        } else {
            fused_metadata_copy_target_verify_kernel<false, false><<<num_blocks, threads_per_block, 0, stream>>>(
                cache_seqlens_src.data_ptr<int32_t>(),
                cu_seqlens_k_src.data_ptr<int32_t>(),
                page_indices_src.data_ptr<int32_t>(),
                nsa_cache_seqlens_src.data_ptr<int32_t>(),
                seqlens_expanded_src.data_ptr<int32_t>(),
                nsa_cu_seqlens_k_src.data_ptr<int32_t>(),
                real_table_src_ptr,
                flashmla_num_splits_src_ptr,
                flashmla_metadata_src_ptr,
                cache_seqlens_dst.data_ptr<int32_t>(),
                cu_seqlens_k_dst.data_ptr<int32_t>(),
                page_table_1_dst.data_ptr<int32_t>(),
                nsa_cache_seqlens_dst.data_ptr<int32_t>(),
                seqlens_expanded_dst.data_ptr<int32_t>(),
                nsa_cu_seqlens_k_dst.data_ptr<int32_t>(),
                real_table_dst_ptr,
                flashmla_num_splits_dst_ptr,
                flashmla_metadata_dst_ptr,
                static_cast<int>(bs),
                static_cast<int>(max_seqlen_k),
                static_cast<int>(seqlens_expanded_size),
                page_indices_rows,
                page_table_1_stride,
                real_page_table_cols,
                real_page_table_dst_stride,
                flashmla_metadata_size
            );
        }
    } else if (forward_mode == DRAFT_EXTEND) {
        if (has_real_page_table && has_flashmla_metadata) {
            fused_metadata_copy_draft_extend_kernel<true, true><<<num_blocks, threads_per_block, 0, stream>>>(
                cache_seqlens_src.data_ptr<int32_t>(),
                cu_seqlens_k_src.data_ptr<int32_t>(),
                page_indices_src.data_ptr<int32_t>(),
                nsa_cache_seqlens_src.data_ptr<int32_t>(),
                seqlens_expanded_src.data_ptr<int32_t>(),
                nsa_cu_seqlens_k_src.data_ptr<int32_t>(),
                real_table_src_ptr,
                flashmla_num_splits_src_ptr,
                flashmla_metadata_src_ptr,
                cache_seqlens_dst.data_ptr<int32_t>(),
                cu_seqlens_k_dst.data_ptr<int32_t>(),
                page_table_1_dst.data_ptr<int32_t>(),
                nsa_cache_seqlens_dst.data_ptr<int32_t>(),
                seqlens_expanded_dst.data_ptr<int32_t>(),
                nsa_cu_seqlens_k_dst.data_ptr<int32_t>(),
                real_table_dst_ptr,
                flashmla_num_splits_dst_ptr,
                flashmla_metadata_dst_ptr,
                static_cast<int>(bs),
                static_cast<int>(max_seqlen_k),
                static_cast<int>(seqlens_expanded_size),
                page_indices_rows,
                page_table_1_stride,
                real_page_table_cols,
                real_page_table_dst_stride,
                flashmla_metadata_size
            );
        } else if (has_real_page_table) {
            fused_metadata_copy_draft_extend_kernel<true, false><<<num_blocks, threads_per_block, 0, stream>>>(
                cache_seqlens_src.data_ptr<int32_t>(),
                cu_seqlens_k_src.data_ptr<int32_t>(),
                page_indices_src.data_ptr<int32_t>(),
                nsa_cache_seqlens_src.data_ptr<int32_t>(),
                seqlens_expanded_src.data_ptr<int32_t>(),
                nsa_cu_seqlens_k_src.data_ptr<int32_t>(),
                real_table_src_ptr,
                flashmla_num_splits_src_ptr,
                flashmla_metadata_src_ptr,
                cache_seqlens_dst.data_ptr<int32_t>(),
                cu_seqlens_k_dst.data_ptr<int32_t>(),
                page_table_1_dst.data_ptr<int32_t>(),
                nsa_cache_seqlens_dst.data_ptr<int32_t>(),
                seqlens_expanded_dst.data_ptr<int32_t>(),
                nsa_cu_seqlens_k_dst.data_ptr<int32_t>(),
                real_table_dst_ptr,
                flashmla_num_splits_dst_ptr,
                flashmla_metadata_dst_ptr,
                static_cast<int>(bs),
                static_cast<int>(max_seqlen_k),
                static_cast<int>(seqlens_expanded_size),
                page_indices_rows,
                page_table_1_stride,
                real_page_table_cols,
                real_page_table_dst_stride,
                flashmla_metadata_size
            );
        } else if (has_flashmla_metadata) {
            fused_metadata_copy_draft_extend_kernel<false, true><<<num_blocks, threads_per_block, 0, stream>>>(
                cache_seqlens_src.data_ptr<int32_t>(),
                cu_seqlens_k_src.data_ptr<int32_t>(),
                page_indices_src.data_ptr<int32_t>(),
                nsa_cache_seqlens_src.data_ptr<int32_t>(),
                seqlens_expanded_src.data_ptr<int32_t>(),
                nsa_cu_seqlens_k_src.data_ptr<int32_t>(),
                real_table_src_ptr,
                flashmla_num_splits_src_ptr,
                flashmla_metadata_src_ptr,
                cache_seqlens_dst.data_ptr<int32_t>(),
                cu_seqlens_k_dst.data_ptr<int32_t>(),
                page_table_1_dst.data_ptr<int32_t>(),
                nsa_cache_seqlens_dst.data_ptr<int32_t>(),
                seqlens_expanded_dst.data_ptr<int32_t>(),
                nsa_cu_seqlens_k_dst.data_ptr<int32_t>(),
                real_table_dst_ptr,
                flashmla_num_splits_dst_ptr,
                flashmla_metadata_dst_ptr,
                static_cast<int>(bs),
                static_cast<int>(max_seqlen_k),
                static_cast<int>(seqlens_expanded_size),
                page_indices_rows,
                page_table_1_stride,
                real_page_table_cols,
                real_page_table_dst_stride,
                flashmla_metadata_size
            );
        } else {
            fused_metadata_copy_draft_extend_kernel<false, false><<<num_blocks, threads_per_block, 0, stream>>>(
                cache_seqlens_src.data_ptr<int32_t>(),
                cu_seqlens_k_src.data_ptr<int32_t>(),
                page_indices_src.data_ptr<int32_t>(),
                nsa_cache_seqlens_src.data_ptr<int32_t>(),
                seqlens_expanded_src.data_ptr<int32_t>(),
                nsa_cu_seqlens_k_src.data_ptr<int32_t>(),
                real_table_src_ptr,
                flashmla_num_splits_src_ptr,
                flashmla_metadata_src_ptr,
                cache_seqlens_dst.data_ptr<int32_t>(),
                cu_seqlens_k_dst.data_ptr<int32_t>(),
                page_table_1_dst.data_ptr<int32_t>(),
                nsa_cache_seqlens_dst.data_ptr<int32_t>(),
                seqlens_expanded_dst.data_ptr<int32_t>(),
                nsa_cu_seqlens_k_dst.data_ptr<int32_t>(),
                real_table_dst_ptr,
                flashmla_num_splits_dst_ptr,
                flashmla_metadata_dst_ptr,
                static_cast<int>(bs),
                static_cast<int>(max_seqlen_k),
                static_cast<int>(seqlens_expanded_size),
                page_indices_rows,
                page_table_1_stride,
                real_page_table_cols,
                real_page_table_dst_stride,
                flashmla_metadata_size
            );
        }
    } else {
        TORCH_CHECK(false, "Invalid forward_mode: ", forward_mode);
    }

    // Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "fused_metadata_copy_kernel failed: ", cudaGetErrorString(err));
}
