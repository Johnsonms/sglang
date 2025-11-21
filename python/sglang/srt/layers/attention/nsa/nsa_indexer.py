from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch
import triton
import triton.language as tl
from einops import rearrange

from sglang.srt.custom_op import CustomOp
from sglang.srt.layers.layernorm import LayerNorm
from sglang.srt.utils import add_prefix, ceil_align, is_cuda, is_hip, is_npu

if is_cuda():
    try:
        import deep_gemm
    except ImportError as e:
        deep_gemm = e


from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.attention.nsa.utils import (
    NSA_DUAL_STREAM,
    cp_all_gather_rerange_output,
    is_nsa_enable_prefill_cp,
)
from sglang.srt.layers.dp_attention import (
    get_attention_tp_group,
    get_attention_tp_rank,
    get_attention_tp_size,
)
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.rotary_embedding import get_rope_wrapper
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.server_args import get_global_server_args

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool

DUAL_STREAM_TOKEN_THRESHOLD = 1024 if is_cuda() else 0


class BaseIndexerMetadata(ABC):
    @abstractmethod
    def get_seqlens_int32(self) -> torch.Tensor:
        """
        Return: (batch_size,) int32 tensor
        """

    @abstractmethod
    def get_page_table_64(self) -> torch.Tensor:
        """
        Return: (batch_size, num_blocks) int32, page table.
                The page size of the table is 64.
        """

    @abstractmethod
    def get_seqlens_expanded(self) -> torch.Tensor:
        """
        Return: (sum_extend_seq_len,) int32 tensor
        """

    @abstractmethod
    def topk_transform(
        self,
        logits: torch.Tensor,
        topk: int,
    ) -> torch.Tensor:
        """
        Perform topk selection on the logits and possibly transform the result.

        NOTE that attention backend may override this function to do some
        transformation, which means the result of this topk_transform may not
        be the topk indices of the input logits.

        Return: Anything, since it will be passed to the attention backend
                for further processing on sparse attention computation.
                Don't assume it is the topk indices of the input logits.
        """


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.bfloat16
    from sgl_kernel import hadamard_transform

    hidden_size = x.size(-1)
    assert (
        hidden_size & (hidden_size - 1)
    ) == 0, "Hidden size must be a power of 2 for Hadamard transform."
    return hadamard_transform(x, scale=hidden_size**-0.5)


@triton.jit
def _concat_1d_kernel(
    # Flattened input tensors (all concatenated into one buffer for indexing)
    in_buffer_ptr,
    # Cumulative offsets for each tensor in the input buffer
    in_offsets_ptr,
    # Sizes of each input tensor
    in_sizes_ptr,
    # Output buffer
    out_ptr,
    # Total output size
    total_size,
    # Number of input tensors
    num_tensors,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Efficient 1D concatenation kernel.
    Each thread block handles a contiguous chunk of output elements.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_size

    # For each output position, find which input tensor it belongs to
    # and copy the data
    for i in range(BLOCK_SIZE):
        out_idx = block_start + i
        if out_idx >= total_size:
            break

        # Binary search to find which tensor this output index belongs to
        # For simplicity, use linear search (can optimize later)
        cumsum = 0
        for tensor_idx in range(num_tensors):
            size = tl.load(in_sizes_ptr + tensor_idx)
            if out_idx < cumsum + size:
                # Found the tensor
                in_offset = tl.load(in_offsets_ptr + tensor_idx)
                local_offset = out_idx - cumsum
                value = tl.load(in_buffer_ptr + in_offset + local_offset)
                tl.store(out_ptr + out_idx, value)
                break
            cumsum += size


@triton.jit
def _concat_2d_kernel(
    # Input pointers for each tensor (stored as int64)
    in_ptr_0,
    in_ptr_1,
    in_ptr_2,
    in_ptr_3,
    # Sizes along dim 0 for each tensor
    size_0,
    size_1,
    size_2,
    size_3,
    # Common size along dim 1
    dim1_size,
    # Output pointer
    out_ptr,
    # Number of valid input tensors (1-4)
    num_tensors,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized 2D concatenation kernel for up to 4 tensors along dim 0.
    Handles the common case where tensors have shape (N_i, D) and we concat along dim 0.
    """
    # Each program handles one row in the output
    row_idx = tl.program_id(0)

    # Initialize variables (required for Triton compiler)
    in_ptr = in_ptr_0
    local_row = 0
    valid = True

    # Determine which input tensor this row belongs to
    # Calculate cumulative sums
    cumsum_0 = 0
    cumsum_1 = size_0
    cumsum_2 = cumsum_1 + size_1
    cumsum_3 = cumsum_2 + size_2
    cumsum_4 = cumsum_3 + size_3

    # Find which tensor this row belongs to
    if row_idx < cumsum_1:
        in_ptr = in_ptr_0
        local_row = row_idx - cumsum_0
    elif row_idx < cumsum_2 and num_tensors >= 2:
        in_ptr = in_ptr_1
        local_row = row_idx - cumsum_1
    elif row_idx < cumsum_3 and num_tensors >= 3:
        in_ptr = in_ptr_2
        local_row = row_idx - cumsum_2
    elif row_idx < cumsum_4 and num_tensors >= 4:
        in_ptr = in_ptr_3
        local_row = row_idx - cumsum_3
    else:
        valid = False

    # Early exit if row is out of bounds
    if not valid:
        return

    # Copy the entire row using vectorized loads/stores
    num_blocks = tl.cdiv(dim1_size, BLOCK_SIZE)
    for block_idx in range(num_blocks):
        col_start = block_idx * BLOCK_SIZE
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < dim1_size

        # Load from input tensor
        in_offset = local_row * dim1_size + cols
        data = tl.load(in_ptr + in_offset, mask=mask)

        # Store to output
        out_offset = row_idx * dim1_size + cols
        tl.store(out_ptr + out_offset, data, mask=mask)


def concat_1d_triton(tensors: List[torch.Tensor]) -> torch.Tensor:
    """
    Concatenate 1D tensors using Triton.

    Args:
        tensors: List of 1D tensors to concatenate

    Returns:
        Concatenated 1D tensor
    """
    if not tensors:
        return None
    if len(tensors) == 1:
        return tensors[0]

    # Calculate total size
    sizes = [t.numel() for t in tensors]
    total_size = sum(sizes)
    device = tensors[0].device
    dtype = tensors[0].dtype

    # Allocate output
    out = torch.empty(total_size, dtype=dtype, device=device)

    # For small concatenations, torch.cat is faster due to kernel launch overhead
    if total_size < 10000 or len(tensors) <= 2:
        return torch.cat(tensors, dim=0)

    # Prepare metadata
    sizes_tensor = torch.tensor(sizes, dtype=torch.int32, device=device)
    offsets = torch.zeros(len(tensors), dtype=torch.int32, device=device)
    offsets[1:] = torch.cumsum(sizes_tensor[:-1], dim=0)

    # Flatten all input tensors into a single buffer for efficient access
    in_buffer = torch.cat([t.flatten().view(torch.uint8) for t in tensors])

    # Launch kernel
    BLOCK_SIZE = 256
    grid = (triton.cdiv(total_size, BLOCK_SIZE),)
    _concat_1d_kernel[grid](
        in_buffer,
        offsets,
        sizes_tensor,
        out,
        total_size,
        len(tensors),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


def concat_2d_triton(tensors: List[torch.Tensor], dim: int = 0) -> torch.Tensor:
    """
    Concatenate 2D tensors using Triton (optimized for dim=0).

    Args:
        tensors: List of 2D tensors with shape (N_i, D) to concatenate
        dim: Dimension to concatenate along (must be 0)

    Returns:
        Concatenated 2D tensor with shape (sum(N_i), D)
    """
    if not tensors:
        return None
    if len(tensors) == 1:
        return tensors[0]

    assert dim == 0, "Only concatenation along dim=0 is supported"
    assert all(t.ndim == 2 for t in tensors), "All tensors must be 2D"

    # Check that all tensors have the same size in dim 1
    dim1_size = tensors[0].shape[1]
    assert all(t.shape[1] == dim1_size for t in tensors), "All tensors must have same size in dim 1"

    # For small or many tensors, use torch.cat
    total_rows = sum(t.shape[0] for t in tensors)
    if total_rows < 1000 or len(tensors) > 4:
        return torch.cat(tensors, dim=0)

    device = tensors[0].device
    dtype = tensors[0].dtype

    # Allocate output
    out = torch.empty((total_rows, dim1_size), dtype=dtype, device=device)

    # Pad tensors list to 4 elements (required by kernel)
    padded_tensors = tensors + [torch.empty(0, dim1_size, dtype=dtype, device=device)] * (4 - len(tensors))
    sizes = [t.shape[0] for t in tensors] + [0] * (4 - len(tensors))

    # Launch kernel - one program per output row
    BLOCK_SIZE = 128 if dim1_size <= 128 else 256
    grid = (total_rows,)
    _concat_2d_kernel[grid](
        padded_tensors[0],
        padded_tensors[1],
        padded_tensors[2],
        padded_tensors[3],
        sizes[0],
        sizes[1],
        sizes[2],
        sizes[3],
        dim1_size,
        out,
        len(tensors),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


def fused_concat_triton(
    tensor_lists: List[List[torch.Tensor]],
    dtypes: List[torch.dtype],
    squeeze_dims: List[Optional[int]] = None,
) -> List[torch.Tensor]:
    """
    Fused concatenation of multiple tensor lists using Triton.

    Args:
        tensor_lists: List of lists of tensors to concatenate
        dtypes: Target dtypes for each output tensor (for view operation)
        squeeze_dims: Optional dimensions to squeeze for each output (-1 for last dim, None for no squeeze)

    Returns:
        List of concatenated tensors with applied dtype views and squeezes
    """
    if squeeze_dims is None:
        squeeze_dims = [None] * len(tensor_lists)

    outputs = []

    for tensors, dtype, squeeze_dim in zip(tensor_lists, dtypes, squeeze_dims):
        if not tensors:
            outputs.append(None)
            continue

        # Calculate total size and allocate output
        total_size = sum(t.numel() for t in tensors)
        device = tensors[0].device

        # Allocate output as uint8 first (for generic storage)
        out = torch.empty(total_size, dtype=torch.uint8, device=device)

        # Prepare metadata
        num_tensors = len(tensors)
        in_sizes = torch.tensor([t.numel() for t in tensors], dtype=torch.int64, device=device)

        # Calculate output offsets (cumulative sum)
        out_offsets = torch.zeros(num_tensors, dtype=torch.int64, device=device)
        out_offsets[1:] = torch.cumsum(in_sizes[:-1], dim=0)

        # Create pointer array (we'll use torch.cat as fallback for now since
        # passing pointer arrays to Triton is complex)
        # For simplicity, use torch.cat but with pre-allocated output
        offset = 0
        for t in tensors:
            size = t.numel()
            out[offset:offset + size] = t.flatten().view(torch.uint8)
            offset += size

        # Reshape and apply view to target dtype
        # Reconstruct the shape
        first_shape = list(tensors[0].shape)
        first_shape[0] = sum(t.shape[0] for t in tensors)
        out = out[:offset].view(first_shape).view(dtype)

        # Apply squeeze if needed
        if squeeze_dim is not None:
            out = out.squeeze(squeeze_dim)

        outputs.append(out)

    return outputs


def fused_concat_k_scale_ks_ke(
    k_fp8_list: List[torch.Tensor],
    k_scale_list: List[torch.Tensor],
    ks_list: List[torch.Tensor],
    ke_list: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Specialized fused concatenation for k_fp8, k_scale, ks, and ke tensors.
    Uses Triton kernels for optimized memory operations.

    This function uses custom Triton kernels for efficient concatenation:
    - k_fp8_list: 2D tensors (seq_len_i, 128) -> use concat_2d_triton
    - k_scale_list: 2D tensors (seq_len_i, 1) -> use concat_2d_triton + squeeze
    - ks_list, ke_list: 1D tensors -> use concat_1d_triton

    Returns:
        k_fp8: concatenated and viewed as float8_e4m3fn
        k_scale: concatenated, viewed as float32, and squeezed
        kv_fp8: tuple of (k_fp8, k_scale)
        ks: concatenated
        ke: concatenated
    """
    # Use Triton kernels for efficient concatenation
    k_fp8 = None
    k_scale = None
    kv_fp8 = None
    ks = None
    ke = None

    if k_fp8_list:
        # Concatenate k_fp8 using 2D Triton kernel
        k_fp8_concat = concat_2d_triton(k_fp8_list, dim=0)
        if k_fp8_concat is not None:
            k_fp8 = k_fp8_concat.view(torch.float8_e4m3fn)

    if k_scale_list:
        # Concatenate k_scale using 2D Triton kernel, then squeeze
        k_scale_concat = concat_2d_triton(k_scale_list, dim=0)
        if k_scale_concat is not None:
            k_scale = k_scale_concat.view(torch.float32).squeeze(-1)

    if k_fp8 is not None and k_scale is not None:
        kv_fp8 = (k_fp8, k_scale)

    if ks_list:
        # Concatenate ks using 1D Triton kernel
        ks = concat_1d_triton(ks_list)

    if ke_list:
        # Concatenate ke using 1D Triton kernel
        ke = concat_1d_triton(ke_list)

    return k_fp8, k_scale, kv_fp8, ks, ke


class Indexer(CustomOp):
    def __init__(
        self,
        hidden_size: int,
        index_n_heads: int,
        index_head_dim: int,
        rope_head_dim: int,
        index_topk: int,
        q_lora_rank: int,
        max_position_embeddings: int,
        rope_theta: float,
        layer_id: int,
        scale_fmt: Optional[str],
        block_size: int = 128,
        rope_scaling: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        quant_config: Optional[QuantizationConfig] = None,
        alt_stream: Optional[torch.cuda.Stream] = None,
        fuse_wk_and_weights_proj: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = index_n_heads
        self.head_dim = index_head_dim
        self.rope_head_dim = rope_head_dim
        self.index_topk = index_topk
        self.q_lora_rank = q_lora_rank
        self.layer_id = layer_id
        self.alt_stream = alt_stream
        self.fuse_wk_and_weights_proj = fuse_wk_and_weights_proj
        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()
        if self.nsa_enable_prefill_cp:
            self.cp_size = get_attention_tp_size()
            self.cp_rank = get_attention_tp_rank()
        else:
            self.cp_size = None
            self.cp_rank = None
        if is_cuda():
            self.sm_count = deep_gemm.get_num_sms()
            self.half_device_sm_count = ceil_align(self.sm_count // 2, 8)

        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wq_b", prefix),
        )
        if self.fuse_wk_and_weights_proj:
            self.fused_wk_and_weights_proj = ReplicatedLinear(
                self.hidden_size,
                self.head_dim + self.n_heads,
                bias=False,
                prefix=add_prefix("fused_wk_and_weights_proj", prefix),
            )
        else:
            self.wk = ReplicatedLinear(
                self.hidden_size,
                self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("wk", prefix),
            )
            # NOTE: weight_proj is not quantized
            self.weights_proj = ReplicatedLinear(
                self.hidden_size,
                self.n_heads,
                bias=False,
                prefix=add_prefix("weights_proj", prefix),
            )
        self.k_norm = LayerNorm(self.head_dim, dtype=torch.float32)
        self.rotary_emb = get_rope_wrapper(
            rope_head_dim,
            rotary_dim=rope_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,  # type: ignore
            rope_scaling=rope_scaling,
            is_neox_style=True,
            device=get_global_server_args().device,
        )
        self.block_size = block_size
        self.scale_fmt = scale_fmt
        self.softmax_scale = self.head_dim**-0.5

    @torch.compile(dynamic=True)
    def _get_logits_head_gate(self, weights: torch.Tensor, q_scale: torch.Tensor):
        weights = weights * self.n_heads**-0.5
        weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale
        return weights

    def _get_q_k_bf16(
        self,
        q_lora: torch.Tensor,
        x: torch.Tensor,
        positions: torch.Tensor,
        enable_dual_stream: bool,
        forward_batch: ForwardBatch,
    ):
        weights = None
        if enable_dual_stream:
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)

            with deep_gemm_wrapper.configure_deep_gemm_num_sms(
                self.half_device_sm_count
            ):
                query, _ = self.wq_b(q_lora)
                query = rearrange(query, "l (h d) -> l h d", d=self.head_dim)
                q_rope, _ = torch.split(
                    query,
                    [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                    dim=-1,
                )
            with torch.cuda.stream(self.alt_stream):
                # TODO we should also put DeepGEMM half SM here?
                if self.fuse_wk_and_weights_proj:
                    key, weights = self.fused_wk_and_weights_proj(x)[0].split(
                        [self.head_dim, self.n_heads], dim=-1
                    )
                else:
                    key, _ = self.wk(x)
                key = self.k_norm(key)

                k_rope, _ = torch.split(
                    key,
                    [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                    dim=-1,
                )

            current_stream.wait_stream(self.alt_stream)
        else:
            query, _ = self.wq_b(q_lora)
            query = rearrange(query, "l (h d) -> l h d", d=self.head_dim)

            q_rope, _ = torch.split(
                query, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
            )

            if self.fuse_wk_and_weights_proj:
                key, weights = self.fused_wk_and_weights_proj(x)[0].split(
                    [self.head_dim, self.n_heads], dim=-1
                )
            else:
                key, _ = self.wk(x)
            key = self.k_norm(key)
            k_rope, _ = torch.split(
                key, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
            )

        q_rope, k_rope = self.rotary_emb(positions, q_rope, k_rope)

        query[..., : self.rope_head_dim] = q_rope
        key[..., : self.rope_head_dim] = k_rope

        # allgather+rerrange
        if forward_batch.nsa_cp_metadata is not None and self.nsa_enable_prefill_cp:
            key = cp_all_gather_rerange_output(
                key.contiguous(),
                self.cp_size,
                forward_batch,
                torch.cuda.current_stream(),
            )

        if enable_dual_stream:
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            query = rotate_activation(query)

            with torch.cuda.stream(self.alt_stream):
                key = rotate_activation(key)
            current_stream.wait_stream(self.alt_stream)
        else:
            query = rotate_activation(query)
            key = rotate_activation(key)

        return query, key, weights

    def _get_k_bf16(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        enable_dual_stream: bool,
    ):
        # Compute only key, skip query and weights (weights is discarded if fused)
        if self.fuse_wk_and_weights_proj:
            key, _ = self.fused_wk_and_weights_proj(x)[0].split(
                [self.head_dim, self.n_heads], dim=-1
            )
        else:
            key, _ = self.wk(x)
        key = self.k_norm(key)
        k_rope, _ = torch.split(
            key, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )

        _, k_rope = self.rotary_emb(positions, k_rope, k_rope)
        key[..., : self.rope_head_dim] = k_rope
        key = rotate_activation(key)

        return key

    def _get_topk_paged(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        metadata: BaseIndexerMetadata,
    ) -> torch.Tensor:
        if TYPE_CHECKING:
            assert isinstance(forward_batch.token_to_kv_pool, NSATokenToKVPool)

        page_size = forward_batch.token_to_kv_pool.page_size
        # NOTE(dark): blocksize = 64 is hardcoded in deep_gemm
        assert page_size == 64, "only support page size 64"

        # NOTE(dark): this support extend/decode/decode+graph
        block_tables = metadata.get_page_table_64()

        max_seq_len = block_tables.shape[1] * page_size
        kv_cache_fp8 = forward_batch.token_to_kv_pool.get_index_k_with_scale_buffer(
            layer_id=layer_id
        )

        blocksize = page_size
        if (
            forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend()
        ):
            seqlens_32 = metadata.get_seqlens_expanded()
        else:
            seqlens_32 = metadata.get_seqlens_int32()
        # NOTE(dark): 132 is SM count on H200/B200, not magic number
        schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
            seqlens_32, blocksize, self.sm_count
        )

        assert len(q_fp8.shape) == 3
        q_fp8 = q_fp8.unsqueeze(1)  # the next_n dim is 1 now
        assert len(kv_cache_fp8.shape) == 2
        block_kv = 64
        num_heads_kv = 1
        head_dim_with_sf = 132
        kv_cache_fp8 = kv_cache_fp8.view(
            kv_cache_fp8.shape[0], block_kv, num_heads_kv, head_dim_with_sf
        )
        assert len(weights.shape) == 3
        weights = weights.squeeze(2)

        logits = deep_gemm.fp8_paged_mqa_logits(
            q_fp8,
            kv_cache_fp8,
            weights,
            seqlens_32,
            block_tables,
            schedule_metadata,
            max_seq_len,
            clean_logits=False,
        )

        # NOTE(dark): logits should be cleaned in topk_transform
        topk_result = metadata.topk_transform(logits, self.index_topk)
        return topk_result

    def _get_topk_ragged(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        metadata: BaseIndexerMetadata,
    ) -> torch.Tensor:
        if TYPE_CHECKING:
            assert isinstance(forward_batch.token_to_kv_pool, NSATokenToKVPool)

        assert forward_batch.forward_mode.is_extend_without_speculative()

        page_size = forward_batch.token_to_kv_pool.page_size
        assert page_size == 64, "only support page size 64"
        assert len(weights.shape) == 3
        weights = weights.squeeze(-1)
        k_fp8_list = []
        k_scale_list = []
        ks_list = []
        ke_list = []

        q_offset = 0
        k_offset = 0

        seq_lens_expanded = metadata.get_seqlens_expanded()
        block_tables = metadata.get_page_table_64()

        assert (
            forward_batch.seq_lens_cpu is not None
            and forward_batch.extend_seq_lens_cpu is not None
        )

        for i in range(forward_batch.batch_size):
            seq_len = forward_batch.seq_lens_cpu[i].item()
            assert isinstance(seq_len, int)
            # Use fused Triton kernel to get both K and scale in a single call
            k_fp8, k_scale = forward_batch.token_to_kv_pool.get_index_k_scale_buffer(
                layer_id,
                seq_len,
                block_tables[i],
            )
            extend_seq_len = forward_batch.extend_seq_lens_cpu[i]
            ks = torch.full(
                (extend_seq_len,), k_offset, dtype=torch.int32, device="cuda"
            )
            ke = ks + seq_lens_expanded[q_offset : q_offset + extend_seq_len]
            k_fp8_list.append(k_fp8)
            k_scale_list.append(k_scale)
            ks_list.append(ks)
            ke_list.append(ke)

            q_offset += extend_seq_len
            k_offset += seq_len

        # Use fused concatenation for better performance
        k_fp8, k_scale, kv_fp8, ks, ke = fused_concat_k_scale_ks_ke(
            k_fp8_list, k_scale_list, ks_list, ke_list
        )

        # Suppose there are two requests, with extend_seq_len = [3, 2]
        # and seq_lens = [10, 4]
        # The logits matrix looks like this, with * representing the valid logits
        # and - representing the invalid logits:
        #
        #  ********--|----
        #  *********-|----
        #  **********|----
        #  ----------|***-
        #  ----------|****
        #
        # ks = [0, 0, 0, 10, 10]
        # ke = [8, 9, 10, 13, 14]

        logits = deep_gemm.fp8_mqa_logits(
            q_fp8[:q_offset],
            kv_fp8,
            weights[:q_offset],
            ks,
            ke,
            clean_logits=False,
        )

        token_nums, _, _ = q_fp8.shape
        assert logits.shape[0] == len(seq_lens_expanded)
        assert logits.shape[1] == k_offset

        raw_topk_result = metadata.topk_transform(logits, self.index_topk, ks=ks)
        topk_result = torch.full(
            (token_nums, self.index_topk), -1, device=q_fp8.device, dtype=torch.int32
        )
        topk_result[:q_offset] = raw_topk_result
        return topk_result

    def _forward_cuda_k_only(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        act_quant,
        enable_dual_stream: bool,
        metadata: BaseIndexerMetadata,
        return_indices: bool = True,
    ) -> Optional[torch.Tensor]:
        assert forward_batch.forward_mode.is_extend_without_speculative()

        # Fast path: only compute and store k cache, skip all q and weights ops
        key = self._get_k_bf16(x, positions, enable_dual_stream)
        k_fp8, k_scale = act_quant(key, self.block_size, self.scale_fmt)

        if not forward_batch.out_cache_loc.is_contiguous():
            forward_batch.out_cache_loc = forward_batch.out_cache_loc.contiguous()
        forward_batch.token_to_kv_pool.set_index_k_scale_buffer(
            layer_id=layer_id,
            loc=forward_batch.out_cache_loc,
            index_k=k_fp8,
            index_k_scale=k_scale,
        )

        # MHA doesn't need topk_indices
        if not return_indices:
            return None

        # MLA: use dummy logits with topk kernel's fast path to generate indices
        # When length <= 2048, naive_topk_cuda directly generates [0,1,...,length-1,-1,...]
        seq_lens_expanded = metadata.get_seqlens_expanded()
        dummy_logits = torch.zeros(
            seq_lens_expanded.shape[0],
            self.index_topk,
            dtype=torch.float32,
            device=x.device,
        )
        return metadata.topk_transform(dummy_logits, self.index_topk)

    def _get_topk_ragged_with_cp(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        metadata: BaseIndexerMetadata,
        kv_len: int,
        actual_seq_q: int,
        cp_index: List[Tuple[int, int, int]] = None,
    ) -> torch.Tensor:
        if TYPE_CHECKING:
            assert isinstance(forward_batch.token_to_kv_pool, NSATokenToKVPool)

        page_size = forward_batch.token_to_kv_pool.page_size
        assert page_size == 64, "only support page size 64"
        assert len(weights.shape) == 3
        weights = weights.squeeze(-1)
        k_fp8_list = []
        k_scale_list = []
        ks_list = []
        ke_offset_list = []
        offset = 0
        actual_seq_q_list = []
        batch_idx_list = []

        block_tables = metadata.get_page_table_64()

        assert (
            forward_batch.seq_lens_cpu is not None
            and forward_batch.extend_seq_lens_cpu is not None
        )
        if cp_index is not None:
            # TODO Multi-batch support has accuracy issues
            for batch_idx, start_seq_position, end_seq_position in cp_index:
                pre_chunk_offset = (
                    forward_batch.seq_lens_cpu[batch_idx].item()
                    - forward_batch.extend_seq_lens_cpu[batch_idx]
                )
                start_seq_position += pre_chunk_offset
                end_seq_position += pre_chunk_offset
                if offset == 0 and batch_idx != 0:
                    offset += forward_batch.extend_seq_lens_cpu[batch_idx - 1]
                # Use fused Triton kernel to get both K and scale in a single call
                k_fp8, k_scale = forward_batch.token_to_kv_pool.get_index_k_scale_buffer(
                    layer_id,
                    end_seq_position,
                    block_tables[batch_idx],
                )

                extend_seq_len = end_seq_position - start_seq_position
                ks = torch.full(
                    (extend_seq_len,), offset, dtype=torch.int32, device="cuda"
                )
                k_fp8_list.append(k_fp8)
                k_scale_list.append(k_scale)
                ks_list.append(ks)
                ke_offset = torch.arange(
                    start_seq_position + 1,
                    end_seq_position + 1,
                    dtype=torch.int32,
                    device="cuda",
                )
                ke_offset_list.append(ke_offset)
                actual_seq_q = torch.tensor(
                    [extend_seq_len], dtype=torch.int32, device="cuda"
                )
                actual_seq_q_list.append(actual_seq_q)
                batch_idx_list.append(batch_idx)

            # Use fused concatenation for better performance
            k_fp8, k_scale, kv_fp8, ks, _ = fused_concat_k_scale_ks_ke(
                k_fp8_list, k_scale_list, ks_list, []
            )
            ke_offset = torch.cat(ke_offset_list, dim=0)
            ke = ks + ke_offset
            actual_seq_q = torch.cat(actual_seq_q_list, dim=0)
            logits = deep_gemm.fp8_mqa_logits(
                q_fp8,
                kv_fp8,
                weights,
                ks,
                ke,
                clean_logits=False,
            )
            topk_result = metadata.topk_transform(
                logits,
                self.index_topk,
                ks=ks,
                cu_seqlens_q=actual_seq_q,
                ke_offset=ke_offset,
                batch_idx_list=batch_idx_list,
            )
        else:
            kv_len = (
                forward_batch.seq_lens_cpu[0].item()
                - forward_batch.extend_seq_lens_cpu[0]
                + kv_len
            )
            # Use fused Triton kernel to get both K and scale in a single call
            k_fp8, k_scale = forward_batch.token_to_kv_pool.get_index_k_scale_buffer(
                layer_id,
                kv_len,
                block_tables[0],
            )

            k_fp8 = k_fp8.view(torch.float8_e4m3fn)
            k_scale = k_scale.view(torch.float32).squeeze(-1)
            kv_fp8 = (k_fp8, k_scale)
            ks = torch.full((actual_seq_q,), offset, dtype=torch.int32, device="cuda")
            ke_offset = torch.arange(
                (kv_len - actual_seq_q) + 1,
                kv_len + 1,
                dtype=torch.int32,
                device="cuda",
            )
            ke = ks + ke_offset

            logits = deep_gemm.fp8_mqa_logits(
                q_fp8,
                kv_fp8,
                weights,
                ks,
                ke,
                clean_logits=False,
            )
            actual_seq_q = torch.tensor([actual_seq_q], dtype=torch.int32).to(
                device="cuda", non_blocking=True
            )
            topk_result = metadata.topk_transform(
                logits,
                self.index_topk,
                ks=ks,
                cu_seqlens_q=actual_seq_q,
                ke_offset=ke_offset,
            )

        return topk_result

    def forward_indexer(
        self,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        forward_batch: ForwardBatch,
        topk: int,
        layer_id: int,
    ) -> Optional[torch.Tensor]:
        if not is_npu():
            from sglang.srt.layers.attention.nsa.tilelang_kernel import fp8_index

        page_size = forward_batch.token_to_kv_pool.page_size
        assert page_size == 64, "only support page size 64"

        assert len(weights.shape) == 3
        weights = weights.squeeze(-1)

        # logits = deep_gemm.fp8_mqa_logits(q_fp8, kv_fp8, weights, ks, ke)
        k_fp8_list = []
        k_scale_list = []

        topk_indices_list = []

        block_tables = forward_batch.req_to_token_pool.req_to_token[
            forward_batch.req_pool_indices, :
        ]
        strided_indices = torch.arange(
            0, block_tables.shape[-1], page_size, device="cuda"
        )
        block_tables = block_tables[:, strided_indices] // page_size

        q_len_start = 0

        for i in range(forward_batch.batch_size):
            seq_len = forward_batch.seq_lens[i].item()
            q_len = (
                forward_batch.extend_seq_lens_cpu[i]
                if forward_batch.forward_mode.is_extend()
                else 1
            )
            q_len_end = q_len_start + q_len

            q_fp8_partial = q_fp8[q_len_start:q_len_end]
            q_fp8_partial = q_fp8_partial.unsqueeze(0).contiguous()

            weights_partial = weights[q_len_start:q_len_end]
            weights_partial = weights_partial.squeeze(-1).unsqueeze(0).contiguous()

            # Use fused Triton kernel to get both K and scale in a single call
            k_fp8, k_scale = forward_batch.token_to_kv_pool.get_index_k_scale_buffer(
                layer_id,
                seq_len,
                block_tables[i],
            )

            k_fp8 = k_fp8.view(torch.float8_e4m3fn).unsqueeze(0).contiguous()
            k_scale = k_scale.view(torch.float32).squeeze(-1).unsqueeze(0).contiguous()

            index_score = fp8_index(
                q_fp8_partial,
                weights_partial,
                k_fp8,
                k_scale,
            )
            end_pos = seq_len
            topk_indices = index_score.topk(min(topk, end_pos), dim=-1)[1].squeeze(0)

            pad_len = ceil_align(topk_indices.shape[-1], 2048) - topk_indices.shape[-1]
            topk_indices = torch.nn.functional.pad(
                topk_indices, (0, pad_len), "constant", -1
            )

            topk_indices_list.append(topk_indices)

            q_len_start = q_len_end

        topk_indices = torch.cat(topk_indices_list, dim=0)
        return topk_indices

    def forward_cuda(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        return_indices: bool = True,
    ) -> Optional[torch.Tensor]:
        if is_hip():
            from sglang.srt.layers.attention.nsa.tilelang_kernel import act_quant
        elif not is_npu():
            from sglang.srt.layers.attention.nsa.triton_kernel import act_quant

        if TYPE_CHECKING:
            assert isinstance(forward_batch.token_to_kv_pool, NSATokenToKVPool)

        metadata = forward_batch.attn_backend.get_indexer_metadata(
            layer_id, forward_batch
        )

        enable_dual_stream = (
            NSA_DUAL_STREAM
            and self.alt_stream is not None
            and get_is_capture_mode()
            and q_lora.shape[0] > 0
            and q_lora.shape[0] <= DUAL_STREAM_TOKEN_THRESHOLD
        )

        # skip NSA if attention backend choose to skip this batch
        if metadata is None:
            return None

        # Determine if should skip topk based on sequence length
        # We can only skip the logits computation if cuda graph is not involved
        skip_logits_computation = False
        if forward_batch.forward_mode.is_extend_without_speculative():
            if forward_batch.seq_lens_cpu is not None:
                max_kv_len = forward_batch.seq_lens_cpu.max().item()
                skip_logits_computation = max_kv_len <= self.index_topk

        # Optimization: fast path when skipping topk computation
        if skip_logits_computation and (not self.nsa_enable_prefill_cp):
            return self._forward_cuda_k_only(
                x,
                positions,
                forward_batch,
                layer_id,
                act_quant,
                enable_dual_stream,
                metadata,
                return_indices,
            )

        query, key, weights = self._get_q_k_bf16(
            q_lora, x, positions, enable_dual_stream, forward_batch=forward_batch
        )

        if enable_dual_stream:
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)

            q_fp8, q_scale = act_quant(query, self.block_size, self.scale_fmt)
            with torch.cuda.stream(self.alt_stream):
                k_fp8, k_scale = act_quant(key, self.block_size, self.scale_fmt)
            current_stream.wait_stream(self.alt_stream)
        else:
            q_fp8, q_scale = act_quant(query, self.block_size, self.scale_fmt)
            k_fp8, k_scale = act_quant(key, self.block_size, self.scale_fmt)

        # k_fp8: (seq_len, head_dim) fp8_e4m3fn
        # k_buffer: (num_total_tokens + page_size, head_dim) fp8_e4m3fn
        # k_scale: (seq_len, head_dim // block_size = 1) fp8_e4m3fn
        # k_scale_cache: (num_total_tokens + page_size, head_dim // block_size = 1) fp8_e4m3fn
        if not forward_batch.out_cache_loc.is_contiguous():
            forward_batch.out_cache_loc = forward_batch.out_cache_loc.contiguous()
        forward_batch.token_to_kv_pool.set_index_k_scale_buffer(
            layer_id=layer_id,
            loc=forward_batch.out_cache_loc,
            index_k=k_fp8,
            index_k_scale=k_scale,
        )

        if not self.fuse_wk_and_weights_proj:
            weights, _ = self.weights_proj(x)
        weights = self._get_logits_head_gate(weights, q_scale)

        if is_cuda():
            assert forward_batch.seq_lens_cpu is not None
            if len(forward_batch.seq_lens_cpu) == 0:
                # this seems b/c max-pad, no worries?
                # if x.shape[0] != 0:
                #     print(
                #         "HACK: seq_lens empty but x not empty, hackily return all-invalid topk_result"
                #     )
                return torch.full(
                    (x.shape[0], self.index_topk), -1, dtype=torch.int, device="cuda"
                )

            if (
                forward_batch.forward_mode.is_decode_or_idle()
                or forward_batch.forward_mode.is_target_verify()
                or forward_batch.forward_mode.is_draft_extend()
            ):
                topk_result = self._get_topk_paged(
                    forward_batch, layer_id, q_fp8, weights, metadata
                )
            else:
                if (
                    forward_batch.nsa_cp_metadata is not None
                    and self.nsa_enable_prefill_cp
                ):
                    kv_len_prev = forward_batch.nsa_cp_metadata.kv_len_prev
                    kv_len_next = forward_batch.nsa_cp_metadata.kv_len_next
                    actual_seq_q_prev = forward_batch.nsa_cp_metadata.actual_seq_q_prev
                    actual_seq_q_next = forward_batch.nsa_cp_metadata.actual_seq_q_next

                    # TODO support mutil-batch
                    # cp_batch_seq_index_prev = forward_batch.nsa_cp_metadata["cp_batch_seq_index_prev"]
                    # cp_batch_seq_index_next = forward_batch.nsa_cp_metadata["cp_batch_seq_index_next"]
                    # TODO prev, next, combined into a single call
                    q_fp8_prev, q_fp8_next = torch.split(
                        q_fp8, (q_fp8.shape[0] + 1) // 2, dim=0
                    )
                    weights_prev, weights_next = torch.split(
                        weights, (weights.shape[0] + 1) // 2, dim=0
                    )
                    topk_result_prev = self._get_topk_ragged_with_cp(
                        forward_batch,
                        layer_id,
                        q_fp8_prev,
                        weights_prev,
                        metadata,
                        kv_len_prev,
                        actual_seq_q_prev,
                    )

                    topk_result_next = self._get_topk_ragged_with_cp(
                        forward_batch,
                        layer_id,
                        q_fp8_next,
                        weights_next,
                        metadata,
                        kv_len_next,
                        actual_seq_q_next,
                    )
                    return torch.cat([topk_result_prev, topk_result_next], dim=0)
                else:
                    topk_result = self._get_topk_ragged(
                        forward_batch, layer_id, q_fp8, weights, metadata
                    )
        else:
            topk_result = self.forward_indexer(
                q_fp8.contiguous(),
                weights,
                forward_batch,
                topk=self.index_topk,
                layer_id=layer_id,
            )
        return topk_result

    def forward_npu(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
    ) -> torch.Tensor:
        import custom_ops  # noqa: F401
        import torch_npu

        from sglang.srt.layers.dp_attention import (
            get_attention_tp_rank,
            get_attention_tp_size,
        )
        from sglang.srt.utils import get_bool_env_var

        if forward_batch.attn_backend.forward_metadata.seq_lens_cpu_int is None:
            actual_seq_lengths_kv = forward_batch.attn_backend.forward_metadata.seq_lens
        else:
            actual_seq_lengths_kv = (
                forward_batch.attn_backend.forward_metadata.seq_lens_cpu_int
            )
        enable_index_cp = (
            get_bool_env_var("SGLANG_USE_AG_AFTER_QLORA") and layer_id >= 4
        )
        is_prefill = (
            forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_draft_extend_v2()
            and not forward_batch.forward_mode.is_target_verify()
            and not forward_batch.forward_mode.is_draft_extend()
        )

        attention_tp_rank = get_attention_tp_rank()
        attention_tp_size = get_attention_tp_size()

        cos_sin = self.rotary_emb.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        cos = cos.repeat(1, 2).view(-1, 1, 1, self.rope_head_dim)
        sin = sin.repeat(1, 2).view(-1, 1, 1, self.rope_head_dim)
        if is_prefill and enable_index_cp:
            slice_length = cos.shape[0] // attention_tp_size
            cos = cos[
                slice_length
                * attention_tp_rank : slice_length
                * (attention_tp_rank + 1)
            ]
            sin = sin[
                slice_length
                * attention_tp_rank : slice_length
                * (attention_tp_rank + 1)
            ]

        slot_mapping = forward_batch.out_cache_loc
        block_table = forward_batch.attn_backend.forward_metadata.block_tables

        bs = x.shape[0]

        q = self.wq_b(q_lora)[0]  # [bs, 1536] @ [1536, 64 * 128] = [bs, 64 * 128]
        q = q.view(bs, self.n_heads, self.head_dim)  # [bs, 64, 128]
        q_pe, q_nope = torch.split(
            q,
            [self.rope_head_dim, self.head_dim - self.rope_head_dim],
            dim=-1,
        )  # [bs, 64, 64 + 64]

        q_pe = q_pe.view(bs, self.n_heads, 1, self.rope_head_dim)
        q_pe = torch_npu.npu_interleave_rope(q_pe, cos, sin).view(
            bs, self.n_heads, self.rope_head_dim
        )  # [bs, n, d]
        q = torch.cat([q_pe, q_nope], dim=-1)

        k_proj = self.wk(x)[0]  # [b, s, 7168] @ [7168, 128] = [b, s, 128]
        k = self.k_norm(k_proj)
        k_pe, k_nope = torch.split(
            k,
            [self.rope_head_dim, self.head_dim - self.rope_head_dim],
            dim=-1,
        )  # [bs, 64 + 64]

        k_pe = k_pe.view(-1, 1, 1, self.rope_head_dim)
        k_pe = torch_npu.npu_interleave_rope(k_pe, cos, sin).view(
            bs, 1, self.rope_head_dim
        )  # [bs, 1, d]
        k = torch.cat([k_pe, k_nope.unsqueeze(1)], dim=-1)  # [bs, 1, 128]

        if is_prefill and enable_index_cp:
            k, local_k = (
                torch.empty(
                    (k.shape[0] * attention_tp_size, k.shape[1], k.shape[2]),
                    dtype=k.dtype,
                    device=k.device,
                ),
                k,
            )
            get_attention_tp_group().all_gather_into_tensor(k, local_k)

        forward_batch.token_to_kv_pool.set_index_k_buffer(layer_id, slot_mapping, k)

        indexer_input = {}
        if is_prefill:
            actual_seq_lengths_kv = forward_batch.seq_lens.to(device=q.device)
            actual_seq_lengths_q = forward_batch.seq_lens.cumsum(dim=0).to(
                device=q.device
            )
            if enable_index_cp:
                actual_seq_lengths_q -= bs * attention_tp_rank
                actual_seq_lengths_q = torch.max(
                    actual_seq_lengths_q,
                    torch.zeros_like(actual_seq_lengths_q).to(
                        device=actual_seq_lengths_q.device
                    ),
                )
                actual_seq_lengths_q = torch.min(
                    actual_seq_lengths_q,
                    torch.full(actual_seq_lengths_q.shape, bs).to(
                        device=actual_seq_lengths_q.device
                    ),
                )

        else:
            if forward_batch.attn_backend.forward_metadata.actual_seq_lengths_q is None:
                if (
                    forward_batch.forward_mode.is_draft_extend_v2()
                    or forward_batch.forward_mode.is_target_verify()
                    or forward_batch.forward_mode.is_draft_extend()
                ):
                    num_draft_tokens = (
                        forward_batch.attn_backend.speculative_num_draft_tokens
                    )
                    actual_seq_lengths_q = torch.arange(
                        num_draft_tokens,
                        num_draft_tokens + bs,
                        num_draft_tokens,
                        dtype=torch.int32,
                        device=k.device,
                    )
                else:
                    actual_seq_lengths_q = torch.tensor(
                        [1 + i * 1 for i in range(bs)],
                        dtype=torch.int32,
                        device=k.device,
                    )
            else:
                actual_seq_lengths_q = (
                    forward_batch.attn_backend.forward_metadata.actual_seq_lengths_q
                )

        past_key_states = forward_batch.token_to_kv_pool.get_index_k_buffer(layer_id)

        x = x.view(-1, self.hidden_size)
        weights = self.weights_proj(x)[0]
        block_table = (
            block_table[: actual_seq_lengths_q.size()[0]] if is_prefill else block_table
        )

        topk_indices = torch.ops.custom.npu_lightning_indexer(
            query=q.view(-1, self.n_heads, self.head_dim),
            key=past_key_states,
            weights=weights,
            actual_seq_lengths_query=actual_seq_lengths_q.to(torch.int32),
            actual_seq_lengths_key=actual_seq_lengths_kv.to(k.device).to(torch.int32),
            block_table=block_table,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=self.index_topk,
            sparse_mode=3,
        )

        if is_prefill and enable_index_cp:
            topk_indices, local_topk_indices = (
                torch.empty(
                    (
                        topk_indices.shape[0] * attention_tp_size,
                        topk_indices.shape[1],
                        topk_indices.shape[2],
                    ),
                    dtype=topk_indices.dtype,
                    device=topk_indices.device,
                ),
                topk_indices,
            )
            get_attention_tp_group().all_gather_into_tensor(
                topk_indices, local_topk_indices
            )

        return topk_indices
