/*
 * Fused precomputation kernel for NSA backend decode mode metadata.
 *
 * This kernel fuses multiple operations into a single kernel launch:
 * 1. dtype conversion (seq_lens -> cache_seqlens)
 * 2. cumulative sum with padding (cache_seqlens -> cu_seqlens_k)
 * 3. NSA seqlens computation (clamp)
 * 4. cumulative sum for NSA (nsa_cache_seqlens -> nsa_cu_seqlens_k)
 * 5. page table gathering from req_to_token
 * 6. page table transformation (if real_page_size > 1)
 *
 * Uses shared memory for intermediate results to minimize memory traffic.
 */

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include "pytorch_extension_utils.h"

// Maximum batch size that can fit in shared memory
// Adjust based on GPU shared memory limits
#define MAX_SHARED_BS 256

/**
 * Parallel prefix sum (inclusive scan) in shared memory.
 * Uses Blelloch scan algorithm for efficiency.
 *
 * Assumes:
 * - data is in shared memory
 * - n elements to scan
 * - threads are synchronized before and after
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
 * Fused kernel for decode mode metadata precomputation.
 *
 * Workflow:
 * 1. Load seq_lens into shared memory as cache_seqlens (with dtype conversion if needed)
 * 2. Compute cu_seqlens_k using parallel prefix sum
 * 3. Compute nsa_cache_seqlens (clamp operation)
 * 4. Compute nsa_cu_seqlens_k using parallel prefix sum
 * 5. Gather page_indices from req_to_token in parallel
 * 6. Transform page table if needed (real_page_size > 1)
 * 7. Write all results to global memory
 */
template<typename SeqLenType>
__global__ void precompute_decode_metadata_kernel(
    // Inputs
    const SeqLenType* __restrict__ seq_lens_src,    // [bs]
    const int32_t* __restrict__ req_pool_indices,   // [bs]
    const int32_t* __restrict__ req_to_token,       // [total_requests, req_to_token_stride]

    // Outputs
    int32_t* __restrict__ cache_seqlens_dst,        // [bs]
    int32_t* __restrict__ cu_seqlens_k_dst,         // [bs+1]
    int32_t* __restrict__ page_indices_dst,         // [bs, max_len]
    int32_t* __restrict__ nsa_cache_seqlens_dst,    // [bs]
    int32_t* __restrict__ nsa_cu_seqlens_k_dst,     // [bs+1]
    int32_t* __restrict__ real_page_table_dst,      // [bs, real_page_table_cols] or nullptr

    // Parameters
    int bs,
    int max_len,
    int req_to_token_stride,
    int page_indices_dst_stride,
    int nsa_index_topk,
    int real_page_size,
    int real_page_table_cols,
    int real_page_table_dst_stride
) {
    // Shared memory for intermediate results
    __shared__ int32_t shared_cache_seqlens[MAX_SHARED_BS];
    __shared__ int32_t shared_nsa_cache_seqlens[MAX_SHARED_BS];
    __shared__ int32_t shared_cu_seqlens[MAX_SHARED_BS + 1];
    __shared__ int32_t shared_nsa_cu_seqlens[MAX_SHARED_BS + 1];

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Check if batch size fits in shared memory
    if (bs > MAX_SHARED_BS) {
        if (tid == 0) {
            printf("ERROR: Batch size %d exceeds MAX_SHARED_BS %d\n", bs, MAX_SHARED_BS);
        }
        return;
    }

    // Step 1: Load seq_lens into shared memory as int32 (cache_seqlens)
    for (int i = tid; i < bs; i += block_size) {
        shared_cache_seqlens[i] = static_cast<int32_t>(seq_lens_src[i]);
    }
    __syncthreads();

    // Step 2: Compute nsa_cache_seqlens (clamp to nsa_index_topk)
    for (int i = tid; i < bs; i += block_size) {
        shared_nsa_cache_seqlens[i] = min(shared_cache_seqlens[i], nsa_index_topk);
    }
    __syncthreads();

    // Step 3: Compute cu_seqlens_k (cumsum with padding)
    // Initialize: shared_cu_seqlens[0] = 0, shared_cu_seqlens[1:] = cache_seqlens
    if (tid == 0) {
        shared_cu_seqlens[0] = 0;
    }
    for (int i = tid; i < bs; i += block_size) {
        shared_cu_seqlens[i + 1] = shared_cache_seqlens[i];
    }
    __syncthreads();

    // Parallel prefix sum on shared_cu_seqlens[1:bs+1]
    inclusive_scan_shared(shared_cu_seqlens + 1, bs, tid, block_size);

    // Step 4: Compute nsa_cu_seqlens_k (cumsum with padding)
    if (tid == 0) {
        shared_nsa_cu_seqlens[0] = 0;
    }
    for (int i = tid; i < bs; i += block_size) {
        shared_nsa_cu_seqlens[i + 1] = shared_nsa_cache_seqlens[i];
    }
    __syncthreads();

    // Parallel prefix sum on shared_nsa_cu_seqlens[1:bs+1]
    inclusive_scan_shared(shared_nsa_cu_seqlens + 1, bs, tid, block_size);

    // Step 5: Write cache_seqlens, cu_seqlens_k, nsa_cache_seqlens, nsa_cu_seqlens_k to global memory
    for (int i = tid; i < bs; i += block_size) {
        cache_seqlens_dst[i] = shared_cache_seqlens[i];
        nsa_cache_seqlens_dst[i] = shared_nsa_cache_seqlens[i];
    }

    for (int i = tid; i <= bs; i += block_size) {
        cu_seqlens_k_dst[i] = shared_cu_seqlens[i];
        nsa_cu_seqlens_k_dst[i] = shared_nsa_cu_seqlens[i];
    }
    __syncthreads();

    // Step 6: Gather page_indices from req_to_token
    // Total elements: bs * max_len
    int total_page_elements = bs * max_len;
    for (int i = tid; i < total_page_elements; i += block_size) {
        int row = i / max_len;
        int col = i % max_len;
        int req_idx = req_pool_indices[row];
        int src_offset = req_idx * req_to_token_stride + col;
        int dst_offset = row * page_indices_dst_stride + col;
        page_indices_dst[dst_offset] = req_to_token[src_offset];
    }

    // Step 7: Transform page table if needed (real_page_size > 1)
    if (real_page_table_dst != nullptr && real_page_size > 1) {
        int total_real_elements = bs * real_page_table_cols;
        for (int i = tid; i < total_real_elements; i += block_size) {
            int row = i / real_page_table_cols;
            int col = i % real_page_table_cols;

            // Strided indexing: col * real_page_size
            int src_col = col * real_page_size;
            int src_offset = row * page_indices_dst_stride + src_col;

            // Read from page_indices_dst and divide by page_size
            int value = page_indices_dst[src_offset] / real_page_size;

            int dst_offset = row * real_page_table_dst_stride + col;
            real_page_table_dst[dst_offset] = value;
        }
    }
}

/**
 * PyTorch wrapper function for precompute_decode_metadata.
 *
 * This function replaces the Python implementation of _precompute_decode_mode
 * by fusing all operations into a single kernel launch.
 */
void precompute_decode_metadata_cuda(
    at::Tensor seq_lens,                    // Input: [bs], dtype could be int32 or int64
    at::Tensor req_pool_indices,            // Input: [bs], can be int32 or int64, will be converted
    at::Tensor req_to_token,                // Input: [total_requests, req_to_token_stride], can be int32 or int64, will be converted
    at::Tensor cache_seqlens,               // Output: [bs], int32
    at::Tensor cu_seqlens_k,                // Output: [bs+1], int32
    at::Tensor page_indices,                // Output: [bs, max_len], int32
    at::Tensor nsa_cache_seqlens,           // Output: [bs], int32
    at::Tensor nsa_cu_seqlens_k,            // Output: [bs+1], int32
    c10::optional<at::Tensor> real_page_table,  // Output: [bs, real_page_table_cols], int32 or None
    int64_t max_len,
    int64_t nsa_index_topk,
    int64_t real_page_size
) {
    // Validate inputs
    CHECK_INPUT(seq_lens);
    CHECK_INPUT(req_pool_indices);
    CHECK_INPUT(req_to_token);
    CHECK_INPUT(cache_seqlens);
    CHECK_INPUT(cu_seqlens_k);
    CHECK_INPUT(page_indices);
    CHECK_INPUT(nsa_cache_seqlens);
    CHECK_INPUT(nsa_cu_seqlens_k);

    // Convert req_pool_indices to int32 if needed
    if (req_pool_indices.dtype() != torch::kInt32) {
        req_pool_indices = req_pool_indices.to(torch::kInt32);
    }

    // Convert req_to_token to int32 if needed
    if (req_to_token.dtype() != torch::kInt32) {
        req_to_token = req_to_token.to(torch::kInt32);
    }

    int bs = seq_lens.size(0);
    int req_to_token_stride = req_to_token.size(1);
    int page_indices_dst_stride = page_indices.size(1);

    // Check batch size limit
    TORCH_CHECK(bs <= MAX_SHARED_BS,
        "Batch size ", bs, " exceeds maximum supported batch size ", MAX_SHARED_BS,
        " for precompute_decode_metadata_cuda");

    // Real page table parameters
    int32_t* real_page_table_ptr = nullptr;
    int real_page_table_cols = 0;
    int real_page_table_dst_stride = 0;

    if (real_page_table.has_value() && real_page_size > 1) {
        CHECK_INPUT(real_page_table.value());
        real_page_table_ptr = real_page_table.value().data_ptr<int32_t>();
        real_page_table_cols = real_page_table.value().size(1);
        real_page_table_dst_stride = real_page_table.value().stride(0);
    }

    // Launch configuration
    // Use enough threads to handle the largest array (page_indices gathering)
    int threads_per_block = 256;
    int num_blocks = 1;  // Use single block to utilize shared memory efficiently

    // Get current CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Dispatch based on seq_lens dtype
    if (seq_lens.dtype() == torch::kInt32) {
        precompute_decode_metadata_kernel<int32_t><<<num_blocks, threads_per_block, 0, stream>>>(
            seq_lens.data_ptr<int32_t>(),
            req_pool_indices.data_ptr<int32_t>(),
            req_to_token.data_ptr<int32_t>(),
            cache_seqlens.data_ptr<int32_t>(),
            cu_seqlens_k.data_ptr<int32_t>(),
            page_indices.data_ptr<int32_t>(),
            nsa_cache_seqlens.data_ptr<int32_t>(),
            nsa_cu_seqlens_k.data_ptr<int32_t>(),
            real_page_table_ptr,
            bs,
            static_cast<int>(max_len),
            req_to_token_stride,
            page_indices_dst_stride,
            static_cast<int>(nsa_index_topk),
            static_cast<int>(real_page_size),
            real_page_table_cols,
            real_page_table_dst_stride
        );
    } else if (seq_lens.dtype() == torch::kInt64) {
        precompute_decode_metadata_kernel<int64_t><<<num_blocks, threads_per_block, 0, stream>>>(
            seq_lens.data_ptr<int64_t>(),
            req_pool_indices.data_ptr<int32_t>(),
            req_to_token.data_ptr<int32_t>(),
            cache_seqlens.data_ptr<int32_t>(),
            cu_seqlens_k.data_ptr<int32_t>(),
            page_indices.data_ptr<int32_t>(),
            nsa_cache_seqlens.data_ptr<int32_t>(),
            nsa_cu_seqlens_k.data_ptr<int32_t>(),
            real_page_table_ptr,
            bs,
            static_cast<int>(max_len),
            req_to_token_stride,
            page_indices_dst_stride,
            static_cast<int>(nsa_index_topk),
            static_cast<int>(real_page_size),
            real_page_table_cols,
            real_page_table_dst_stride
        );
    } else {
        TORCH_CHECK(false, "Unsupported seq_lens dtype: ", seq_lens.dtype());
    }

    // Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
        "precompute_decode_metadata_kernel failed: ", cudaGetErrorString(err));
}
