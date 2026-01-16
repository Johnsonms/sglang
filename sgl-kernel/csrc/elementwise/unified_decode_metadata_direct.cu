/*
 * Optimized variant of unified_decode_metadata for small number of backends (≤3).
 * Passes pointers directly as kernel arguments instead of through GPU tensor.
 *
 * Benefits:
 * - No CPU tensor allocation
 * - No CPU->GPU transfer
 * - Lower latency for small backend counts
 * - Pointers passed directly in kernel constant memory
 */

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include "pytorch_extension_utils.h"

// Maximum batch size that can fit in shared memory
#define MAX_SHARED_BS 256
#define MAX_BACKENDS_DIRECT 3

/**
 * Parallel prefix sum (inclusive scan) in shared memory.
 */
static __device__ void inclusive_scan_shared_direct(int32_t* data, int n, int tid, int block_size) {
    // Up-sweep (reduce) phase
    for (int stride = 1; stride < n; stride *= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < n) {
            data[index] += data[index - stride];
        }
        __syncthreads();
    }

    // Down-sweep phase
    for (int stride = n / 2; stride > 0; stride /= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index + stride < n) {
            data[index + stride] += data[index];
        }
        __syncthreads();
    }
}

/**
 * Unified DECODE metadata kernel with direct pointer arguments (max 3 backends).
 * All pointers passed as individual int64 arguments - no GPU tensor needed!
 */
template<typename SeqLenType>
__global__ void unified_decode_metadata_kernel_direct(
    // === Inputs ===
    const SeqLenType* __restrict__ seq_lens_src,
    const int32_t* __restrict__ req_pool_indices,
    const int32_t* __restrict__ req_to_token,

    // === Backend 0 pointers (always present) ===
    int64_t cache_seqlens_ptr0,
    int64_t cu_seqlens_k_ptr0,
    int64_t page_indices_ptr0,
    int64_t nsa_cache_seqlens_ptr0,
    int64_t nsa_cu_seqlens_k_ptr0,
    int64_t real_page_table_ptr0,
    int64_t seqlens_expanded_ptr0,

    // === Backend 1 pointers (if num_backends >= 2) ===
    int64_t cache_seqlens_ptr1,
    int64_t cu_seqlens_k_ptr1,
    int64_t page_indices_ptr1,
    int64_t nsa_cache_seqlens_ptr1,
    int64_t nsa_cu_seqlens_k_ptr1,
    int64_t real_page_table_ptr1,
    int64_t seqlens_expanded_ptr1,

    // === Backend 2 pointers (if num_backends == 3) ===
    int64_t cache_seqlens_ptr2,
    int64_t cu_seqlens_k_ptr2,
    int64_t page_indices_ptr2,
    int64_t nsa_cache_seqlens_ptr2,
    int64_t nsa_cu_seqlens_k_ptr2,
    int64_t real_page_table_ptr2,
    int64_t seqlens_expanded_ptr2,

    // === Parameters ===
    int num_backends,
    int bs,
    int max_len_allocated,
    int req_to_token_stride,
    int page_indices_dst_stride,
    int nsa_index_topk,
    int real_page_size,
    int real_page_table_cols,
    int real_page_table_dst_stride
) {
    // ===== Shared Memory =====
    __shared__ int32_t shared_cache_seqlens[MAX_SHARED_BS];
    __shared__ int32_t shared_nsa_cache_seqlens[MAX_SHARED_BS];
    __shared__ int32_t shared_cu_seqlens[MAX_SHARED_BS + 1];
    __shared__ int32_t shared_nsa_cu_seqlens[MAX_SHARED_BS + 1];
    __shared__ int32_t shared_max_len;

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    if (bs > MAX_SHARED_BS) {
        if (tid == 0) {
            printf("ERROR: Batch size %d exceeds MAX_SHARED_BS %d\n", bs, MAX_SHARED_BS);
        }
        return;
    }

    // ===== STEP 1: Load & Convert seq_lens =====
    for (int i = tid; i < bs; i += block_size) {
        shared_cache_seqlens[i] = static_cast<int32_t>(seq_lens_src[i]);
    }
    __syncthreads();

    // ===== STEP 2: Compute Max Length =====
    if (tid == 0) {
        shared_max_len = 0;
    }
    __syncthreads();

    int local_max = 0;
    for (int i = tid; i < bs; i += block_size) {
        local_max = max(local_max, shared_cache_seqlens[i]);
    }

    if (local_max > 0) {
        atomicMax(&shared_max_len, local_max);
    }
    __syncthreads();

    int max_len = min(shared_max_len, max_len_allocated);

    // ===== STEP 3: Compute cu_seqlens_k =====
    if (tid == 0) {
        shared_cu_seqlens[0] = 0;
    }
    for (int i = tid; i < bs; i += block_size) {
        shared_cu_seqlens[i + 1] = shared_cache_seqlens[i];
    }
    __syncthreads();

    inclusive_scan_shared_direct(shared_cu_seqlens + 1, bs, tid, block_size);

    // ===== STEP 4: NSA Clamp =====
    for (int i = tid; i < bs; i += block_size) {
        shared_nsa_cache_seqlens[i] = min(shared_cache_seqlens[i], nsa_index_topk);
    }
    __syncthreads();

    // ===== STEP 5: NSA Cumsum =====
    if (tid == 0) {
        shared_nsa_cu_seqlens[0] = 0;
    }
    for (int i = tid; i < bs; i += block_size) {
        shared_nsa_cu_seqlens[i + 1] = shared_nsa_cache_seqlens[i];
    }
    __syncthreads();

    inclusive_scan_shared_direct(shared_nsa_cu_seqlens + 1, bs, tid, block_size);

    // ===== STEP 6: Broadcast to backends =====
    // Create array of pointers for each backend
    int64_t cache_seqlens_ptrs[MAX_BACKENDS_DIRECT] = {cache_seqlens_ptr0, cache_seqlens_ptr1, cache_seqlens_ptr2};
    int64_t cu_seqlens_k_ptrs[MAX_BACKENDS_DIRECT] = {cu_seqlens_k_ptr0, cu_seqlens_k_ptr1, cu_seqlens_k_ptr2};
    int64_t page_indices_ptrs[MAX_BACKENDS_DIRECT] = {page_indices_ptr0, page_indices_ptr1, page_indices_ptr2};
    int64_t nsa_cache_seqlens_ptrs[MAX_BACKENDS_DIRECT] = {nsa_cache_seqlens_ptr0, nsa_cache_seqlens_ptr1, nsa_cache_seqlens_ptr2};
    int64_t nsa_cu_seqlens_k_ptrs[MAX_BACKENDS_DIRECT] = {nsa_cu_seqlens_k_ptr0, nsa_cu_seqlens_k_ptr1, nsa_cu_seqlens_k_ptr2};
    int64_t real_page_table_ptrs[MAX_BACKENDS_DIRECT] = {real_page_table_ptr0, real_page_table_ptr1, real_page_table_ptr2};
    int64_t seqlens_expanded_ptrs[MAX_BACKENDS_DIRECT] = {seqlens_expanded_ptr0, seqlens_expanded_ptr1, seqlens_expanded_ptr2};

    for (int backend_idx = 0; backend_idx < num_backends; backend_idx++) {
        int32_t* cache_seqlens_dst = reinterpret_cast<int32_t*>(cache_seqlens_ptrs[backend_idx]);
        int32_t* cu_seqlens_k_dst = reinterpret_cast<int32_t*>(cu_seqlens_k_ptrs[backend_idx]);
        int32_t* nsa_cache_seqlens_dst = reinterpret_cast<int32_t*>(nsa_cache_seqlens_ptrs[backend_idx]);
        int32_t* nsa_cu_seqlens_k_dst = reinterpret_cast<int32_t*>(nsa_cu_seqlens_k_ptrs[backend_idx]);
        int32_t* page_indices_dst = reinterpret_cast<int32_t*>(page_indices_ptrs[backend_idx]);
        int32_t* seqlens_expanded_dst = reinterpret_cast<int32_t*>(seqlens_expanded_ptrs[backend_idx]);
        int32_t* real_page_table_dst = reinterpret_cast<int32_t*>(real_page_table_ptrs[backend_idx]);

        // 6.1: Copy basic metadata
        for (int i = tid; i < bs; i += block_size) {
            cache_seqlens_dst[i] = shared_cache_seqlens[i];
            nsa_cache_seqlens_dst[i] = shared_nsa_cache_seqlens[i];
            seqlens_expanded_dst[i] = shared_cache_seqlens[i];
        }

        for (int i = tid; i <= bs; i += block_size) {
            cu_seqlens_k_dst[i] = shared_cu_seqlens[i];
            nsa_cu_seqlens_k_dst[i] = shared_nsa_cu_seqlens[i];
        }

        // 6.2: Gather page_indices
        int total_page_elements = bs * max_len;
        for (int i = tid; i < total_page_elements; i += block_size) {
            int row = i / max_len;
            int col = i % max_len;
            int req_idx = req_pool_indices[row];
            int src_offset = req_idx * req_to_token_stride + col;
            int dst_offset = row * page_indices_dst_stride + col;
            page_indices_dst[dst_offset] = req_to_token[src_offset];
        }

        // 6.3: Transform real_page_table
        if (real_page_table_dst != nullptr && real_page_size > 1) {
            int total_real_elements = bs * real_page_table_cols;
            for (int i = tid; i < total_real_elements; i += block_size) {
                int row = i / real_page_table_cols;
                int col = i % real_page_table_cols;
                int src_col = col * real_page_size;
                int src_offset = row * page_indices_dst_stride + src_col;
                int value = page_indices_dst[src_offset] / real_page_size;
                int dst_offset = row * real_page_table_dst_stride + col;
                real_page_table_dst[dst_offset] = value;
            }
        }
    }
}

/**
 * PyTorch wrapper - passes pointers directly as arguments (no GPU tensor needed!)
 */
void unified_decode_metadata_cuda_direct(
    torch::Tensor seq_lens,
    torch::Tensor req_pool_indices,
    torch::Tensor req_to_token,

    // Backend 0 pointers
    int64_t cache_seqlens_ptr0,
    int64_t cu_seqlens_k_ptr0,
    int64_t page_indices_ptr0,
    int64_t nsa_cache_seqlens_ptr0,
    int64_t nsa_cu_seqlens_k_ptr0,
    int64_t real_page_table_ptr0,
    int64_t seqlens_expanded_ptr0,

    // Backend 1 pointers
    int64_t cache_seqlens_ptr1,
    int64_t cu_seqlens_k_ptr1,
    int64_t page_indices_ptr1,
    int64_t nsa_cache_seqlens_ptr1,
    int64_t nsa_cu_seqlens_k_ptr1,
    int64_t real_page_table_ptr1,
    int64_t seqlens_expanded_ptr1,

    // Backend 2 pointers
    int64_t cache_seqlens_ptr2,
    int64_t cu_seqlens_k_ptr2,
    int64_t page_indices_ptr2,
    int64_t nsa_cache_seqlens_ptr2,
    int64_t nsa_cu_seqlens_k_ptr2,
    int64_t real_page_table_ptr2,
    int64_t seqlens_expanded_ptr2,

    // Parameters
    int64_t num_backends,
    int64_t max_len_allocated,
    int64_t nsa_index_topk,
    int64_t real_page_size,
    int64_t real_page_table_cols,
    int64_t real_page_table_dst_stride
) {
    CHECK_INPUT(seq_lens);
    CHECK_INPUT(req_pool_indices);
    CHECK_INPUT(req_to_token);

    TORCH_CHECK(num_backends > 0 && num_backends <= MAX_BACKENDS_DIRECT,
        "num_backends must be in range [1, ", MAX_BACKENDS_DIRECT, "], got ", num_backends);

    // Convert to int32 if needed
    if (req_pool_indices.dtype() != torch::kInt32) {
        req_pool_indices = req_pool_indices.to(torch::kInt32);
    }
    if (req_to_token.dtype() != torch::kInt32) {
        req_to_token = req_to_token.to(torch::kInt32);
    }

    int bs = seq_lens.size(0);
    int req_to_token_stride = req_to_token.size(1);
    int page_indices_dst_stride = static_cast<int>(max_len_allocated);

    TORCH_CHECK(bs <= MAX_SHARED_BS,
        "Batch size ", bs, " exceeds maximum ", MAX_SHARED_BS);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int threads_per_block = 256;
    int num_blocks = 1;

    if (seq_lens.dtype() == torch::kInt32) {
        unified_decode_metadata_kernel_direct<int32_t><<<num_blocks, threads_per_block, 0, stream>>>(
            seq_lens.data_ptr<int32_t>(),
            req_pool_indices.data_ptr<int32_t>(),
            req_to_token.data_ptr<int32_t>(),
            cache_seqlens_ptr0, cu_seqlens_k_ptr0, page_indices_ptr0,
            nsa_cache_seqlens_ptr0, nsa_cu_seqlens_k_ptr0,
            real_page_table_ptr0, seqlens_expanded_ptr0,
            cache_seqlens_ptr1, cu_seqlens_k_ptr1, page_indices_ptr1,
            nsa_cache_seqlens_ptr1, nsa_cu_seqlens_k_ptr1,
            real_page_table_ptr1, seqlens_expanded_ptr1,
            cache_seqlens_ptr2, cu_seqlens_k_ptr2, page_indices_ptr2,
            nsa_cache_seqlens_ptr2, nsa_cu_seqlens_k_ptr2,
            real_page_table_ptr2, seqlens_expanded_ptr2,
            static_cast<int>(num_backends),
            bs,
            static_cast<int>(max_len_allocated),
            req_to_token_stride,
            page_indices_dst_stride,
            static_cast<int>(nsa_index_topk),
            static_cast<int>(real_page_size),
            static_cast<int>(real_page_table_cols),
            static_cast<int>(real_page_table_dst_stride)
        );
    } else if (seq_lens.dtype() == torch::kInt64) {
        unified_decode_metadata_kernel_direct<int64_t><<<num_blocks, threads_per_block, 0, stream>>>(
            seq_lens.data_ptr<int64_t>(),
            req_pool_indices.data_ptr<int32_t>(),
            req_to_token.data_ptr<int32_t>(),
            cache_seqlens_ptr0, cu_seqlens_k_ptr0, page_indices_ptr0,
            nsa_cache_seqlens_ptr0, nsa_cu_seqlens_k_ptr0,
            real_page_table_ptr0, seqlens_expanded_ptr0,
            cache_seqlens_ptr1, cu_seqlens_k_ptr1, page_indices_ptr1,
            nsa_cache_seqlens_ptr1, nsa_cu_seqlens_k_ptr1,
            real_page_table_ptr1, seqlens_expanded_ptr1,
            cache_seqlens_ptr2, cu_seqlens_k_ptr2, page_indices_ptr2,
            nsa_cache_seqlens_ptr2, nsa_cu_seqlens_k_ptr2,
            real_page_table_ptr2, seqlens_expanded_ptr2,
            static_cast<int>(num_backends),
            bs,
            static_cast<int>(max_len_allocated),
            req_to_token_stride,
            page_indices_dst_stride,
            static_cast<int>(nsa_index_topk),
            static_cast<int>(real_page_size),
            static_cast<int>(real_page_table_cols),
            static_cast<int>(real_page_table_dst_stride)
        );
    } else {
        TORCH_CHECK(false, "Unsupported seq_lens dtype: ", seq_lens.dtype());
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
        "unified_decode_metadata_kernel_direct failed: ", cudaGetErrorString(err));
}
