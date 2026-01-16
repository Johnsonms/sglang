/*
 * Optimized CUDA kernel with fused prefix sum computation
 *
 * Eliminates torch::zeros, torch::cumsum, and .copy_() operations
 * by computing prefix sum directly in CUDA
 */

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include "utils.h"

// Compute prefix sum in-place using a single thread block
// This is efficient for small bs (typically < 128)
__global__ void compute_prefix_sum_kernel(
    const int* extend_seq_lens,  // [bs]
    int* extend_offsets,         // [bs+1] - output
    int bs
) {
    // Only use one thread block for simplicity and correctness
    // For bs < 128, this is very efficient
    if (blockIdx.x > 0) return;

    int tid = threadIdx.x;

    // Initialize first element to 0
    if (tid == 0) {
        extend_offsets[0] = 0;
    }
    __syncthreads();

    // Compute prefix sum sequentially (fine for small bs)
    // For bs=32, this takes ~32 iterations on one thread - very fast
    if (tid == 0) {
        for (int i = 0; i < bs; i++) {
            extend_offsets[i + 1] = extend_offsets[i] + extend_seq_lens[i];
        }
    }
}

// Alternative: Parallel prefix sum for larger batches (if needed)
// Using Blelloch scan algorithm
__global__ void compute_prefix_sum_parallel_kernel(
    const int* extend_seq_lens,  // [bs]
    int* extend_offsets,         // [bs+1] - output
    int bs
) {
    extern __shared__ int temp[];
    int tid = threadIdx.x;
    int offset = 1;

    // Load input into shared memory (shifted by 1)
    temp[tid] = (tid == 0) ? 0 : ((tid <= bs) ? extend_seq_lens[tid - 1] : 0);
    __syncthreads();

    // Up-sweep phase
    for (int d = blockDim.x >> 1; d > 0; d >>= 1) {
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
        __syncthreads();
    }

    // Clear last element
    if (tid == 0) {
        temp[blockDim.x - 1] = 0;
    }
    __syncthreads();

    // Down-sweep phase
    for (int d = 1; d < blockDim.x; d *= 2) {
        offset >>= 1;
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
        __syncthreads();
    }

    // Write results
    if (tid <= bs) {
        extend_offsets[tid] = temp[tid];
    }
}

// Main metadata computation kernel (same as before)
__global__ void fill_metadata_kernel(
    const int* extend_seq_lens,
    const int* seq_lens,
    const int* extend_offsets,
    int nsa_index_topk,
    int bs,
    int total_tokens,
    int* out_seqlens_expanded,
    int* out_nsa_cache_seqlens
) {
    int token_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_id >= total_tokens) return;

    // Binary search for batch_id
    int left = 0, right = bs - 1, batch_id = 0;
    while (left <= right) {
        int mid = (left + right) / 2;
        if (token_id >= extend_offsets[mid]) {
            batch_id = mid;
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    int extend_len = extend_seq_lens[batch_id];
    int kv_len = seq_lens[batch_id];
    int offset_start = extend_offsets[batch_id];
    int local_token_id = token_id - offset_start;

    int seq_val = kv_len - extend_len + 1 + local_token_id;
    int nsa_seq_val = min(seq_val, nsa_index_topk);

    out_seqlens_expanded[token_id] = seq_val;
    out_nsa_cache_seqlens[token_id] = nsa_seq_val;
}

// Linear search variant (same as before)
__global__ void fill_metadata_kernel_linear(
    const int* extend_seq_lens,
    const int* seq_lens,
    const int* extend_offsets,
    int nsa_index_topk,
    int bs,
    int total_tokens,
    int* out_seqlens_expanded,
    int* out_nsa_cache_seqlens
) {
    int token_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_id >= total_tokens) return;

    int batch_id = 0;
    for (int b = 0; b < bs; b++) {
        if (token_id < extend_offsets[b + 1]) {
            batch_id = b;
            break;
        }
    }

    int extend_len = extend_seq_lens[batch_id];
    int kv_len = seq_lens[batch_id];
    int offset_start = extend_offsets[batch_id];
    int local_token_id = token_id - offset_start;

    int seq_val = kv_len - extend_len + 1 + local_token_id;
    int nsa_seq_val = min(seq_val, nsa_index_topk);

    out_seqlens_expanded[token_id] = seq_val;
    out_nsa_cache_seqlens[token_id] = nsa_seq_val;
}

// **优化后的主函数：消除了 torch 操作**
at::Tensor fill_draft_extend_metadata_cuda_optimized(
    at::Tensor extend_seq_lens,
    at::Tensor seq_lens,
    int64_t nsa_index_topk,
    at::Tensor out_seqlens_expanded,
    at::Tensor out_nsa_cache_seqlens
) {
    CHECK_INPUT(extend_seq_lens);
    CHECK_INPUT(seq_lens);
    CHECK_INPUT(out_seqlens_expanded);
    CHECK_INPUT(out_nsa_cache_seqlens);

    int bs = extend_seq_lens.size(0);
    auto device = extend_seq_lens.device();

    // 分配 extend_offsets buffer（只需要一次分配，无需 zeros）
    at::Tensor extend_offsets = at::empty({bs + 1},
        at::TensorOptions().dtype(c10::ScalarType::Int).device(device));

    // **优化 1：使用自定义 CUDA kernel 计算 prefix sum**
    // 替代了: torch::zeros + torch::cumsum + .copy_()
    // 对于小 bs（典型情况），顺序 prefix sum 非常快
    if (bs < 128) {
        // 使用单线程块的顺序实现（简单且高效）
        compute_prefix_sum_kernel<<<1, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            extend_seq_lens.data_ptr<int>(),
            extend_offsets.data_ptr<int>(),
            bs
        );
    } else {
        // 对于大 bs，使用并行 prefix sum
        int next_pow2 = 1;
        while (next_pow2 < bs + 1) next_pow2 *= 2;
        compute_prefix_sum_parallel_kernel<<<1, next_pow2, next_pow2 * sizeof(int),
                                            at::cuda::getCurrentCUDAStream()>>>(
            extend_seq_lens.data_ptr<int>(),
            extend_offsets.data_ptr<int>(),
            bs
        );
    }

    // 获取 total_tokens（仍需要一次 CPU sync，无法避免）
    int total_tokens;
    cudaMemcpyAsync(&total_tokens,
                    extend_offsets.data_ptr<int>() + bs,
                    sizeof(int),
                    cudaMemcpyDeviceToHost,
                    at::cuda::getCurrentCUDAStream());
    cudaStreamSynchronize(at::cuda::getCurrentCUDAStream());

    if (total_tokens == 0) {
        return at::tensor({0}, at::TensorOptions().dtype(c10::ScalarType::Int).device(device));
    }

    // 启动主 kernel
    const int BLOCK_SIZE = 256;
    int grid_size = (total_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (bs > 16) {
        fill_metadata_kernel<<<grid_size, BLOCK_SIZE, 0, at::cuda::getCurrentCUDAStream()>>>(
            extend_seq_lens.data_ptr<int>(),
            seq_lens.data_ptr<int>(),
            extend_offsets.data_ptr<int>(),
            static_cast<int>(nsa_index_topk),
            bs,
            total_tokens,
            out_seqlens_expanded.data_ptr<int>(),
            out_nsa_cache_seqlens.data_ptr<int>()
        );
    } else {
        fill_metadata_kernel_linear<<<grid_size, BLOCK_SIZE, 0, at::cuda::getCurrentCUDAStream()>>>(
            extend_seq_lens.data_ptr<int>(),
            seq_lens.data_ptr<int>(),
            extend_offsets.data_ptr<int>(),
            static_cast<int>(nsa_index_topk),
            bs,
            total_tokens,
            out_seqlens_expanded.data_ptr<int>(),
            out_nsa_cache_seqlens.data_ptr<int>()
        );
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel error: ", cudaGetErrorString(err));

    return at::tensor({total_tokens},
        at::TensorOptions().dtype(c10::ScalarType::Int).device(device));
}
