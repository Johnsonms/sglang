# Fused Metadata Precompute and Broadcast Kernel

## 概述

这是一个高性能的融合kernel，用于多步推测解码（multi-step speculative decoding）场景中的metadata处理。它将原本需要 **N+1次kernel launch** 的操作融合到 **1次kernel launch**。

## 问题背景

在多步推测解码中（例如EAGLE-3），有N个draft步骤，每个步骤对应一个独立的attention backend。所有backend需要相同的metadata，但之前的实现需要：

```python
# 旧方案：N+1 次 kernel launch
precomputed = precompute_decode_metadata_cuda(...)  # 1次 launch

for i in range(N):  # N次 launch
    backend[i].copy_metadata(precomputed)
```

**性能问题：**
- N+1 次 kernel launch overhead
- 中间结果需要写回global memory
- 每次copy都是独立的memory transaction

## 优化方案

新的融合kernel `fused_metadata_precompute_and_broadcast_cuda` 将precompute和broadcast合并：

```python
# 新方案：1 次 kernel launch
fused_metadata_precompute_and_broadcast_cuda(
    seq_lens,
    req_pool_indices,
    req_to_token,
    cache_seqlens_list=[backend[i].metadata.cache_seqlens for i in range(N)],
    cu_seqlens_k_list=[backend[i].metadata.cu_seqlens_k for i in range(N)],
    page_indices_list=[backend[i].metadata.page_indices for i in range(N)],
    nsa_cache_seqlens_list=[backend[i].metadata.nsa_cache_seqlens for i in range(N)],
    nsa_cu_seqlens_k_list=[backend[i].metadata.nsa_cu_seqlens_k for i in range(N)],
    real_page_table_list=[backend[i].metadata.real_page_table for i in range(N)],
    max_len=max_len,
    nsa_index_topk=nsa_index_topk,
    real_page_size=real_page_size,
)
```

## 核心优化技术

### 1. Shared Memory Reuse
- Precompute的结果存储在shared memory中
- 直接从shared memory broadcast到所有backend destinations
- 避免中间结果写回global memory

### 2. 并行Broadcast
- 所有backend的写操作在同一个kernel中并行完成
- 使用stride访问模式，充分利用memory coalescing
- 每个thread处理多个backend的数据，减少warp divergence

### 3. 单次Kernel Launch
- 减少kernel launch overhead (每次launch约5-10微秒)
- 对于N=3的情况：节约 3 * (5~10μs) = 15~30μs
- 更好的CUDA graph兼容性（graph中的node数量减少）

## 性能提升

以EAGLE-3（N=3）为例：

| 方案 | Kernel Launch次数 | 预估开销 |
|------|------------------|---------|
| 旧方案 (precompute + 3x copy) | 4 | ~20-40 μs + memory BW |
| 新方案 (fused) | 1 | ~5-10 μs + memory BW |
| **提升** | **4x 减少** | **2-3x 更快** |

实际性能提升取决于：
- Batch size (BS越大，相对提升越明显)
- Backend数量 (N越大，提升越显著)
- GPU型号 (新GPU的launch overhead相对更低)

## 代码结构

### CUDA Kernel
- 文件：`csrc/elementwise/fused_metadata_precompute_and_broadcast.cu`
- 核心kernel：`fused_metadata_precompute_and_broadcast_kernel<SeqLenType>`

**执行流程：**
1. 将 seq_lens 加载到 shared memory 并转换为 int32
2. 在 shared memory 中计算 cache_seqlens, cu_seqlens_k
3. 在 shared memory 中计算 nsa_cache_seqlens, nsa_cu_seqlens_k
4. 并行broadcast到所有N个backend destinations
5. 每个backend独立gather page_indices (从 req_to_token)
6. 如果需要，转换 real_page_table

### Python API
- 文件：`python/sgl_kernel/attention.py`
- 函数：`fused_metadata_precompute_and_broadcast_cuda()`

**接口设计：**
- 输入：source tensors (seq_lens, req_pool_indices, req_to_token)
- 输出：N个backend的destination tensor lists
- 自动dtype转换（支持int32/int64输入）

## 使用场景

### 适用场景 ✅
1. **多步推测解码** (EAGLE, Medusa, SPEED)
   - N个draft步骤，每个有独立backend
   - 所有backend需要相同的基础metadata

2. **Decode模式的CUDA graph replay**
   - 固定batch size
   - Metadata计算是性能瓶颈

3. **High-throughput serving**
   - 频繁的metadata更新
   - Kernel launch overhead不可忽略

### 不适用场景 ❌
1. **Single backend** (N=1)
   - 直接用 `precompute_decode_metadata_cuda`
   - 不需要broadcast

2. **不同backend需要不同metadata**
   - 不能复用precompute结果
   - 应该分别调用各自的precompute

3. **Prefill模式**
   - 该kernel仅针对decode模式优化
   - Prefill应使用专门的实现

## 限制和约束

### Batch Size
- 最大batch size: **256** (MAX_SHARED_BS)
- 受限于shared memory大小
- 超过256会fallback或报错

### Backend数量
- 最大backend数量: **8** (MAX_NUM_BACKENDS)
- 可通过修改宏定义增加
- 更多backend会增加register压力

### 内存对齐
- 所有destination tensors必须正确对齐
- 建议使用torch.empty()创建，不要用slice

## 与其他Kernel的关系

### vs. `precompute_decode_metadata_cuda`
- 该kernel是precompute的增强版
- 额外支持broadcast到多个destinations
- 当N=1时，性能相当

### vs. `fused_metadata_copy_cuda`
- `fused_metadata_copy_cuda`: 从已有的precomputed复制到destination
- 该kernel: 直接从input precompute + broadcast，无中间副本

### 组合使用
可以根据场景选择：
```python
# 场景1: 多步推测解码，decode模式
fused_metadata_precompute_and_broadcast_cuda(...)  # 推荐

# 场景2: 需要多次复用precomputed结果
precomputed = precompute_decode_metadata_cuda(...)
for backend in backends:
    fused_metadata_copy_cuda(precomputed, backend.metadata, ...)

# 场景3: 单backend
precompute_decode_metadata_cuda(...)  # 最简单
```

## 调试和验证

### 正确性验证
```python
# 方法1: 与旧实现对比
old_result = [precompute + copy for each backend]
new_result = fused_metadata_precompute_and_broadcast_cuda(...)
assert torch.allclose(old_result, new_result)

# 方法2: 检查所有backend是否相同
for i in range(1, N):
    assert torch.equal(backends[0].metadata, backends[i].metadata)
```

### 性能profiling
```python
import torch.profiler

with torch.profiler.profile() as prof:
    fused_metadata_precompute_and_broadcast_cuda(...)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## 未来优化方向

1. **动态Shared Memory分配**
   - 支持更大的batch size
   - 自动选择最优shared memory策略

2. **支持更多forward modes**
   - 当前仅优化decode模式
   - 可扩展到target_verify和draft_extend

3. **Multi-stream优化**
   - 不同backend group用不同CUDA stream
   - 进一步提升并行度

4. **与FlashMLA metadata融合**
   - 同时处理NSA和MLA metadata
   - 进一步减少kernel launch

## 参考资料

- 相关kernels:
  - `precompute_decode_metadata.cu` - Precompute实现
  - `fused_metadata_copy.cu` - Metadata copy实现

- 论文:
  - EAGLE: "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"
  - SPEED: "Speed: Speculative Pipelined Execution for Efficient Decoding"

- SGLang NSA Backend:
  - `python/sglang/srt/layers/attention/nsa_backend.py`
  - `python/sglang/srt/layers/attention/nsa/nsa_backend_mtp_precompute.py`
