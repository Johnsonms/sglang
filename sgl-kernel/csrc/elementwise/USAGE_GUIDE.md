# 使用指南：Fused Metadata Precompute and Broadcast

## 快速开始

### 1. 编译安装

```bash
cd /sgl-workspace/sglang/sgl-kernel
pip install -e . --no-build-isolation
```

### 2. 启用优化（默认已启用）

```bash
# 环境变量控制
export SGLANG_NSA_ENABLE_MTP_PRECOMPUTE_METADATA=1  # 启用多步优化
export SGLANG_NSA_ENABLE_FUSED_BROADCAST=1          # 启用融合broadcast（新增）

# 运行SGLang服务
python -m sglang.launch_server ...
```

### 3. 验证优化生效

查看日志，应该看到类似信息：
```
[INFO] NSA backend: using fused precompute + broadcast kernel
[INFO] Kernel launches reduced from 4 to 1 (N=3 backends)
```

## 性能对比

### 三种模式

| 模式 | 环境变量设置 | Kernel Launch | 相对性能 |
|------|-------------|--------------|---------|
| **Ultra-fast** (新) | `SGLANG_NSA_ENABLE_FUSED_BROADCAST=1` | **1x** | **最快 (基准)** |
| Fast (原) | `SGLANG_NSA_ENABLE_MTP_PRECOMPUTE_METADATA=1`<br>`SGLANG_NSA_ENABLE_FUSED_BROADCAST=0` | 1 + N | 慢 30-50% |
| Fallback | `SGLANG_NSA_ENABLE_MTP_PRECOMPUTE_METADATA=0` | N 次完整计算 | 慢 2-3x |

### 实测数据（示例）

配置：EAGLE-3 (N=3 backends), Batch Size=32, H100 GPU

```
Mode           | Time (μs) | Speedup
---------------|-----------|--------
Ultra-fast     |    15     |   1.0x
Fast (old)     |    35     |   0.43x
Fallback       |    80     |   0.19x
```

## 适用场景

### ✅ 推荐使用

1. **多步推测解码** (EAGLE, Medusa, SPEED)
   - N个draft步骤 (通常N=2~4)
   - 所有backend共享相同metadata

2. **高吞吐serving**
   - Decode模式为主
   - Batch size ≤ 256
   - GPU kernel launch overhead显著

3. **CUDA graph replay**
   - 固定batch size
   - 追求极致性能

### ❌ 不推荐使用

1. **单backend** (N=1)
   - 无需broadcast，用 `precompute_decode_metadata_cuda`

2. **Prefill模式**
   - 该kernel仅针对decode优化

3. **超大batch** (BS > 256)
   - 超出shared memory限制

4. **异构backends**
   - 不同backend需要不同metadata

## 配置选项

### 环境变量

```bash
# 1. 启用/禁用多步precompute优化（总开关）
export SGLANG_NSA_ENABLE_MTP_PRECOMPUTE_METADATA=1  # 默认: 1

# 2. 启用/禁用融合broadcast（新优化）
export SGLANG_NSA_ENABLE_FUSED_BROADCAST=1          # 默认: 1

# 示例：禁用融合broadcast，回退到fast模式
export SGLANG_NSA_ENABLE_FUSED_BROADCAST=0

# 示例：完全禁用优化，使用fallback模式
export SGLANG_NSA_ENABLE_MTP_PRECOMPUTE_METADATA=0
```

### 代码级控制

如果需要在代码中动态控制：

```python
from sglang.srt.environ import envs

# 临时禁用融合broadcast
with envs.SGLANG_NSA_ENABLE_FUSED_BROADCAST.override(False):
    # 这里会使用fast模式
    forward_batch = ...
```

## 调试和验证

### 1. 正确性验证

```python
# 方法1: 对比不同模式的输出
import os
os.environ["SGLANG_NSA_ENABLE_FUSED_BROADCAST"] = "1"
output_fused = run_inference(...)

os.environ["SGLANG_NSA_ENABLE_FUSED_BROADCAST"] = "0"
output_fast = run_inference(...)

assert torch.allclose(output_fused, output_fast, atol=1e-5)
```

### 2. 性能profiling

```python
import torch
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CUDA]) as prof:
    # 运行推理
    forward_batch = ...
    model(forward_batch)

# 查看kernel launch次数
print(prof.key_averages().table(sort_by="cuda_time_total"))

# 搜索 "fused_metadata_precompute_and_broadcast" kernel
for event in prof.key_averages():
    if "fused_metadata" in event.key:
        print(f"{event.key}: {event.cuda_time_total/1000:.2f} ms")
```

### 3. 检查选择的优化路径

添加日志：

```python
# 在 nsa_backend.py 的 init_forward_metadata_replay_cuda_graph 中
if use_fused_broadcast:
    print("[INFO] Using ultra-fast path: fused precompute + broadcast")
else:
    print("[INFO] Using fast path: precompute + N copies")
```

## 故障排除

### 问题1: ImportError

```
ImportError: cannot import name 'fused_metadata_precompute_and_broadcast_cuda'
```

**解决方案：**
1. 确认已重新编译安装 sgl-kernel
2. 检查是否在正确的Python环境中
3. 尝试清理重新安装：
   ```bash
   pip uninstall sgl-kernel -y
   cd /sgl-workspace/sglang/sgl-kernel
   pip install -e . --no-build-isolation
   ```

### 问题2: CUDA错误

```
RuntimeError: CUDA kernel failed with error: ...
```

**解决方案：**
1. 检查batch size是否超过256
2. 检查backend数量是否超过8
3. 尝试禁用优化：
   ```bash
   export SGLANG_NSA_ENABLE_FUSED_BROADCAST=0
   ```

### 问题3: 性能没有提升

**可能原因：**
1. Batch size太小（<16），kernel launch overhead不明显
2. GPU utilization已经很高，瓶颈在其他地方
3. Backend数量N=1，没有broadcast的必要

**验证方法：**
```bash
# 使用 nsys profiling
nsys profile --trace=cuda --output=profile python -m sglang.launch_server ...

# 查看 timeline，对比不同模式的kernel launch数量
```

## 限制和约束

| 限制项 | 值 | 原因 |
|--------|---|------|
| Max batch size | 256 | Shared memory大小 |
| Max backends | 8 | Register压力 |
| Forward mode | Decode only | Kernel设计 |
| Tensor dtype | int32/int64 | 自动转换支持 |

## 未来改进

计划中的优化：

1. **扩展到其他forward modes**
   - Target verify mode
   - Draft extend mode

2. **动态shared memory**
   - 支持更大batch size
   - 根据可用SHMEM自动调整

3. **Multi-stream优化**
   - Backend groups使用不同streams
   - 进一步提升并行度

4. **与FlashMLA融合**
   - 同时处理NSA和MLA metadata
   - 减少更多kernel launches

## 参考资料

- **Kernel源码**: `csrc/elementwise/fused_metadata_precompute_and_broadcast.cu`
- **Python API**: `python/sgl_kernel/attention.py`
- **设计文档**: `FUSED_PRECOMPUTE_BROADCAST_README.md`
- **NSA Backend**: `python/sglang/srt/layers/attention/nsa_backend.py`

## 问题反馈

如遇到问题，请提供以下信息：

1. SGLang版本和commit hash
2. GPU型号和CUDA版本
3. 推测解码配置（backend数量、batch size）
4. 完整错误日志
5. 使用的环境变量设置

提交issue: https://github.com/sgl-project/sglang/issues
