/*
 * Fused kernel for NSA backend: precompute + broadcast to multiple backends
 *
 * This kernel combines two operations:
 * 1. Precompute decode metadata (from precompute_decode_metadata.cu)
 * 2. Broadcast to N backend destinations (from fused_metadata_copy.cu)
 *
 * Benefits:
 * - Single kernel launch instead of 1 + N launches
 * - Shared memory reuse: precomputed results directly broadcast
 * - Reduced memory bandwidth: avoid intermediate writes
 *
 * Use case: Multi-step speculative decoding with N draft steps
 */

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include "pytorch_extension_utils.h"

// Maximum number of backends to broadcast to
#define MAX_NUM_BACKENDS 8

// Maximum batch size that can fit in shared memory
#define MAX_SHARED_BS 256

/**
 * Parallel prefix sum (inclusive scan) in shared memory.
 */
static __device__ void inclusive_scan_shared(int32_t* data, int n, int tid, int block_size) {
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
 * Fused kernel: precompute decode metadata and broadcast to N backends.
 *
 * This kernel performs:
 * 1. dtype conversion (seq_lens -> cache_seqlens) - in shared mem
 * 2. cumulative sum (cu_seqlens_k) - in shared mem
 * 3. NSA seqlens computation - in shared mem
 * 4. cumulative sum for NSA (nsa_cu_seqlens_k) - in shared mem
 * 5. Broadcast all results to N backend destinations
 * 6. Page table gathering (per backend, in parallel)
 * 7. Real page table transformation (if needed, per backend)
 */
template<typename SeqLenType>
__global__ void fused_metadata_precompute_and_broadcast_kernel(
    // === Input ===
    const SeqLenType* __restrict__ seq_lens_src,           // [bs]
    const int32_t* __restrict__ req_pool_indices,          // [bs]
    const int32_t* __restrict__ req_to_token,              // [total_requests, req_to_token_stride]

    // === Output: N backend destinations ===
    int32_t** __restrict__ cache_seqlens_dst_array,        // [num_backends] -> each [bs]
    int32_t** __restrict__ cu_seqlens_k_dst_array,         // [num_backends] -> each [bs+1]
    int32_t** __restrict__ page_indices_dst_array,         // [num_backends] -> each [bs, max_len]
    int32_t** __restrict__ nsa_cache_seqlens_dst_array,    // [num_backends] -> each [bs]
    int32_t** __restrict__ nsa_cu_seqlens_k_dst_array,     // [num_backends] -> each [bs+1]
    int32_t** __restrict__ real_page_table_dst_array,      // [num_backends] -> each [bs, real_page_table_cols] or nullptr

    // === Parameters ===
    int num_backends,
    int bs,
    int max_len,
    int req_to_token_stride,
    int page_indices_dst_stride,
    int nsa_index_topk,
    int real_page_size,
    int real_page_table_cols,
    int real_page_table_dst_stride
) {
    // Shared memory for precomputed results
    __shared__ int32_t shared_cache_seqlens[MAX_SHARED_BS];
    __shared__ int32_t shared_nsa_cache_seqlens[MAX_SHARED_BS];
    __shared__ int32_t shared_cu_seqlens[MAX_SHARED_BS + 1];
    __shared__ int32_t shared_nsa_cu_seqlens[MAX_SHARED_BS + 1];

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Check batch size limit
    if (bs > MAX_SHARED_BS) {
        if (tid == 0) {
            printf("ERROR: Batch size %d exceeds MAX_SHARED_BS %d\n", bs, MAX_SHARED_BS);
        }
        return;
    }

    // ====== Step 1: Precompute metadata in shared memory ======

    // Load seq_lens -> cache_seqlens
    for (int i = tid; i < bs; i += block_size) {
        shared_cache_seqlens[i] = static_cast<int32_t>(seq_lens_src[i]);
    }
    __syncthreads();

    // Compute nsa_cache_seqlens (clamp)
    for (int i = tid; i < bs; i += block_size) {
        shared_nsa_cache_seqlens[i] = min(shared_cache_seqlens[i], nsa_index_topk);
    }
    __syncthreads();

    // Compute cu_seqlens_k
    if (tid == 0) {
        shared_cu_seqlens[0] = 0;
    }
    for (int i = tid; i < bs; i += block_size) {
        shared_cu_seqlens[i + 1] = shared_cache_seqlens[i];
    }
    __syncthreads();
    inclusive_scan_shared(shared_cu_seqlens + 1, bs, tid, block_size);

    // Compute nsa_cu_seqlens_k
    if (tid == 0) {
        shared_nsa_cu_seqlens[0] = 0;
    }
    for (int i = tid; i < bs; i += block_size) {
        shared_nsa_cu_seqlens[i + 1] = shared_nsa_cache_seqlens[i];
    }
    __syncthreads();
    inclusive_scan_shared(shared_nsa_cu_seqlens + 1, bs, tid, block_size);

    // ====== Step 2: Broadcast to all backends ======

    for (int backend_idx = 0; backend_idx < num_backends; backend_idx++) {
        int32_t* cache_seqlens_dst = cache_seqlens_dst_array[backend_idx];
        int32_t* cu_seqlens_k_dst = cu_seqlens_k_dst_array[backend_idx];
        int32_t* nsa_cache_seqlens_dst = nsa_cache_seqlens_dst_array[backend_idx];
        int32_t* nsa_cu_seqlens_k_dst = nsa_cu_seqlens_k_dst_array[backend_idx];

        // Broadcast basic metadata
        for (int i = tid; i < bs; i += block_size) {
            cache_seqlens_dst[i] = shared_cache_seqlens[i];
            nsa_cache_seqlens_dst[i] = shared_nsa_cache_seqlens[i];
        }

        for (int i = tid; i <= bs; i += block_size) {
            cu_seqlens_k_dst[i] = shared_cu_seqlens[i];
            nsa_cu_seqlens_k_dst[i] = shared_nsa_cu_seqlens[i];
        }

        // Gather page indices (each backend needs this)
        int32_t* page_indices_dst = page_indices_dst_array[backend_idx];
        int total_page_elements = bs * max_len;
        for (int i = tid; i < total_page_elements; i += block_size) {
            int row = i / max_len;
            int col = i % max_len;
            int req_idx = req_pool_indices[row];
            int src_offset = req_idx * req_to_token_stride + col;
            int dst_offset = row * page_indices_dst_stride + col;
            page_indices_dst[dst_offset] = req_to_token[src_offset];
        }

        // Transform real page table if needed
        int32_t* real_page_table_dst = real_page_table_dst_array[backend_idx];
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
 * PyTorch wrapper: precompute and broadcast to multiple backends in one kernel.
 *
 * This function combines:
 * 1. precompute_decode_metadata_cuda - compute once
 * 2. broadcast to N backends - single parallel write
 *
 * Benefits over separate calls:
 * - 1 kernel launch instead of N+1
 * - Shared memory reuse
 * - Better memory coalescing
 * - Zero-copy: pointers extracted in Python, already on GPU
 */
void fused_metadata_precompute_and_broadcast_cuda(
    // Input
    torch::Tensor seq_lens,                              // [bs], int32 or int64
    torch::Tensor req_pool_indices,                      // [bs], int32 or int64
    torch::Tensor req_to_token,                          // [total_requests, stride], int32 or int64

    // Backend pointers: [6, num_backends] tensor of int64 GPU pointers
    // Row 0: cache_seqlens_int32 pointers
    // Row 1: cu_seqlens_k pointers
    // Row 2: page_table_1 pointers
    // Row 3: nsa_cache_seqlens_int32 pointers
    // Row 4: nsa_cu_seqlens_k pointers
    // Row 5: real_page_table pointers (or 0 if nullptr)
    torch::Tensor backend_pointers,                      // [6, N], int64 (CPU or GPU, auto-transferred)

    // Parameters
    int64_t max_len,
    int64_t page_indices_dst_stride,
    int64_t nsa_index_topk,
    int64_t real_page_size,
    int64_t real_page_table_cols,
    int64_t real_page_table_dst_stride
) {
    // Validate inputs
    CHECK_INPUT(seq_lens);
    CHECK_INPUT(req_pool_indices);
    CHECK_INPUT(req_to_token);
    CHECK_INPUT(backend_pointers);

    TORCH_CHECK(backend_pointers.size(0) == 6, "backend_pointers must have 6 rows");
    int num_backends = backend_pointers.size(1);
    TORCH_CHECK(num_backends > 0 && num_backends <= MAX_NUM_BACKENDS,
        "num_backends must be in range [1, ", MAX_NUM_BACKENDS, "], got ", num_backends);
    TORCH_CHECK(backend_pointers.dtype() == torch::kInt64, "backend_pointers must be int64");

    // Transfer backend_pointers to GPU if it's on CPU (single batched transfer is much faster
    // than multiple individual CPU->GPU copies in the caller loop)
    // Note: In PyTorch C++ API, is_cuda() is a method
    if (!backend_pointers.is_cuda()) {
        backend_pointers = backend_pointers.to(seq_lens.device());
    }

    // Convert input tensors to int32 if needed
    if (req_pool_indices.dtype() != torch::kInt32) {
        req_pool_indices = req_pool_indices.to(torch::kInt32);
    }
    if (req_to_token.dtype() != torch::kInt32) {
        req_to_token = req_to_token.to(torch::kInt32);
    }

    int bs = seq_lens.size(0);
    int req_to_token_stride = req_to_token.size(1);

    TORCH_CHECK(bs <= MAX_SHARED_BS,
        "Batch size ", bs, " exceeds maximum supported batch size ", MAX_SHARED_BS);

    // Extract pointer arrays - backend_pointers is already on GPU!
    // No CPU extraction, no cudaMemcpy needed!
    int64_t* backend_pointers_data = backend_pointers.data_ptr<int64_t>();
    int64_t backend_pointers_stride = backend_pointers.stride(0);

    auto cache_seqlens_ptrs = backend_pointers_data + 0 * backend_pointers_stride;
    auto cu_seqlens_k_ptrs = backend_pointers_data + 1 * backend_pointers_stride;
    auto page_indices_ptrs = backend_pointers_data + 2 * backend_pointers_stride;
    auto nsa_cache_seqlens_ptrs = backend_pointers_data + 3 * backend_pointers_stride;
    auto nsa_cu_seqlens_k_ptrs = backend_pointers_data + 4 * backend_pointers_stride;
    auto real_page_table_ptrs = backend_pointers_data + 5 * backend_pointers_stride;

    // Get current CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Launch configuration
    int threads_per_block = 256;
    int num_blocks = 1;

    // Dispatch based on seq_lens dtype
    // Pointers are already on GPU - zero copy!
    if (seq_lens.dtype() == torch::kInt32) {
        fused_metadata_precompute_and_broadcast_kernel<int32_t><<<num_blocks, threads_per_block, 0, stream>>>(
            seq_lens.data_ptr<int32_t>(),
            req_pool_indices.data_ptr<int32_t>(),
            req_to_token.data_ptr<int32_t>(),
            reinterpret_cast<int32_t**>(cache_seqlens_ptrs),
            reinterpret_cast<int32_t**>(cu_seqlens_k_ptrs),
            reinterpret_cast<int32_t**>(page_indices_ptrs),
            reinterpret_cast<int32_t**>(nsa_cache_seqlens_ptrs),
            reinterpret_cast<int32_t**>(nsa_cu_seqlens_k_ptrs),
            reinterpret_cast<int32_t**>(real_page_table_ptrs),
            num_backends,
            bs,
            static_cast<int>(max_len),
            req_to_token_stride,
            static_cast<int>(page_indices_dst_stride),
            static_cast<int>(nsa_index_topk),
            static_cast<int>(real_page_size),
            static_cast<int>(real_page_table_cols),
            static_cast<int>(real_page_table_dst_stride)
        );
    } else if (seq_lens.dtype() == torch::kInt64) {
        fused_metadata_precompute_and_broadcast_kernel<int64_t><<<num_blocks, threads_per_block, 0, stream>>>(
            seq_lens.data_ptr<int64_t>(),
            req_pool_indices.data_ptr<int32_t>(),
            req_to_token.data_ptr<int32_t>(),
            reinterpret_cast<int32_t**>(cache_seqlens_ptrs),
            reinterpret_cast<int32_t**>(cu_seqlens_k_ptrs),
            reinterpret_cast<int32_t**>(page_indices_ptrs),
            reinterpret_cast<int32_t**>(nsa_cache_seqlens_ptrs),
            reinterpret_cast<int32_t**>(nsa_cu_seqlens_k_ptrs),
            reinterpret_cast<int32_t**>(real_page_table_ptrs),
            num_backends,
            bs,
            static_cast<int>(max_len),
            req_to_token_stride,
            static_cast<int>(page_indices_dst_stride),
            static_cast<int>(nsa_index_topk),
            static_cast<int>(real_page_size),
            static_cast<int>(real_page_table_cols),
            static_cast<int>(real_page_table_dst_stride)
        );
    } else {
        TORCH_CHECK(false, "Unsupported seq_lens dtype: ", seq_lens.dtype());
    }

    // Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
        "fused_metadata_precompute_and_broadcast_kernel failed: ", cudaGetErrorString(err));
}
