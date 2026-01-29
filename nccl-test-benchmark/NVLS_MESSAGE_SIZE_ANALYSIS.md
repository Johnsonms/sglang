# NVLS Performance Analysis: Message Size Impact
## Testing When 460 GB/s Threshold is Reached

## Test Overview

Ran focused benchmarks with 3 different message size ranges on both 2-node and 4-node configurations to determine exactly when NCCL with NVLS reaches 460 GB/s bus bandwidth.

### Test Configurations

| Test Range | Message Sizes | Purpose |
|------------|---------------|---------|
| **Small** | 64M - 512M | Baseline performance |
| **Medium** | 512M - 2G | Finding 460 GB/s threshold |
| **Large** | 2G - 8G | Peak performance |

## Summary Results

### 2-Node Configuration (16 GPUs)

| Message Size Range | Avg Bus BW (GB/s) | Reaches 460 GB/s? |
|-------------------|-------------------|-------------------|
| Small (64M-512M) | 404.49 - 404.90 | ❌ NO |
| Medium (512M-2G) | 458.54 - 459.07 | ⚠️ CLOSE (458-459) |
| Large (2G-8G) | 475.76 - 476.15 | ✅ YES (475-476) |

### 4-Node Configuration (32 GPUs)

| Message Size Range | Avg Bus BW (GB/s) | Reaches 460 GB/s? |
|-------------------|-------------------|-------------------|
| Small (64M-512M) | 404.58 - 405.17 | ❌ NO |
| Medium (512M-2G) | 458.42 - 458.76 | ⚠️ CLOSE (458-459) |
| Large (2G-8G) | 475.94 - 476.05 | ✅ YES (475-476) |

## Detailed Performance Breakdown

### When Does Performance Reach 460 GB/s?

#### 2-Node Results

| Message Size | Bus BW (GB/s) | Exceeds 460? |
|--------------|---------------|--------------|
| 512M | 436.79 - 437.68 | ❌ NO |
| **1GB** | **466.55 - 467.41** | ✅ **YES - FIRST TIME!** |
| 2GB | 472.31 - 472.69 | ✅ YES |
| 4GB | 476.34 - 476.86 | ✅ YES |
| **8GB** | **478.73 - 479.08** | ✅ **YES - PEAK!** |

#### 4-Node Results

| Message Size | Bus BW (GB/s) | Exceeds 460? |
|--------------|---------------|--------------|
| 512M | 436.44 - 437.06 | ❌ NO |
| **1GB** | **466.49 - 467.46** | ✅ **YES - FIRST TIME!** |
| 2GB | 472.03 - 473.03 | ✅ YES |
| 4GB | 476.43 - 476.79 | ✅ YES |
| **8GB** | **478.08 - 479.14** | ✅ **YES - PEAK!** |

## Key Findings

### 🎯 460 GB/s Threshold Analysis

1. **First Achieved**: **1GB message size**
   - 2-Node: 466.55 - 467.41 GB/s
   - 4-Node: 466.49 - 467.46 GB/s
   - **Exceeds target by ~6-7 GB/s**

2. **Not Achieved Below 1GB**:
   - 512M messages: ~437 GB/s (falls short by ~23 GB/s)
   - Smaller messages: Even lower performance

3. **Sustained Above 460 GB/s**:
   - All message sizes ≥ 1GB consistently exceed 460 GB/s
   - Performance continues to improve with larger messages

### 📊 Performance Scaling

#### 2-Node Detailed Results

```
Message Size    AlgBW (GB/s)    BusBW (GB/s)    vs 460 GB/s
─────────────────────────────────────────────────────────────
512M            249.59 - 250.10  436.79 - 437.68  -23 GB/s ❌
1GB             266.60 - 267.09  466.55 - 467.41  +6-7 GB/s ✅
2GB             269.89 - 270.11  472.31 - 472.69  +12 GB/s ✅
4GB             272.19 - 272.49  476.34 - 476.86  +16 GB/s ✅
8GB             273.56 - 273.76  478.73 - 479.08  +18-19 GB/s ✅
```

#### 4-Node Detailed Results

```
Message Size    AlgBW (GB/s)    BusBW (GB/s)    vs 460 GB/s
─────────────────────────────────────────────────────────────
512M            249.39 - 249.75  436.44 - 437.06  -23 GB/s ❌
1GB             266.57 - 267.12  466.49 - 467.46  +6-7 GB/s ✅
2GB             269.73 - 270.31  472.03 - 473.03  +12 GB/s ✅
4GB             272.25 - 272.45  476.43 - 476.79  +16 GB/s ✅
8GB             273.19 - 273.79  478.08 - 479.14  +18-19 GB/s ✅
```

## Performance Characteristics

### 1. Small Messages (64M-512M) - 405 GB/s
- **Average**: ~405 GB/s
- **Result**: Does NOT reach 460 GB/s
- **Use Case**: Smaller models, frequent small synchronizations
- **Performance Gap**: -55 GB/s (12% below target)

### 2. Medium Messages (512M-2G) - 458-472 GB/s
- **512M**: ~437 GB/s (still below)
- **1GB**: **467 GB/s** (FIRST to exceed 460!)
- **2GB**: 472 GB/s
- **Critical Finding**: **1GB is the threshold**
- **Use Case**: Medium-large models, typical training workloads

### 3. Large Messages (2G-8G) - 472-479 GB/s
- **2GB**: 472 GB/s
- **4GB**: 476-477 GB/s
- **8GB**: **479 GB/s (PEAK)**
- **Result**: Consistently exceeds 460 GB/s
- **Use Case**: Very large models (LLMs), maximum throughput scenarios

## Scaling Consistency

### 2-Node vs 4-Node Comparison

| Metric | 2-Node | 4-Node | Difference |
|--------|--------|--------|------------|
| Small (avg) | 404.69 GB/s | 404.81 GB/s | +0.12 GB/s |
| Medium (avg) | 458.80 GB/s | 458.64 GB/s | -0.16 GB/s |
| Large (avg) | 475.96 GB/s | 475.99 GB/s | +0.03 GB/s |
| Peak (8GB) | 479.08 GB/s | 479.14 GB/s | +0.06 GB/s |

**Finding**: Performance is remarkably consistent across node counts - scaling is near-perfect!

## Recommendations

### ✅ To Achieve 460+ GB/s:

1. **Use message sizes ≥ 1GB**
   - Guaranteed to exceed 460 GB/s
   - Optimal performance at 2GB+

2. **Best Performance (475+ GB/s)**:
   - Use 2GB - 8GB message sizes
   - 8GB messages achieve peak ~479 GB/s

3. **Production Configuration**:
   - Enable NVLS (NCCL_NVLS_ENABLE=2)
   - Set NCCL_NVLS_NCHANNELS=16
   - Target message sizes ≥ 1GB for collective operations

### ⚠️ Limitations:

1. **Sub-1GB Messages**:
   - 512M: ~437 GB/s (23 GB/s short of 460)
   - Will NOT reach 460 GB/s target
   - Consider batching smaller messages if possible

2. **Small Messages**:
   - <512M: Performance drops further
   - Use only when necessary for application requirements

## Test Files

### 2-Node Tests
- Small: `nccl_2node_nvls_small_170.out`
- Medium: `nccl_2node_nvls_medium_171.out`
- Large: `nccl_2node_nvls_large_172.out`

### 4-Node Tests
- Small: `nccl_4node_nvls_small_173.out`
- Medium: `nccl_4node_nvls_medium_174.out`
- Large: `nccl_4node_nvls_large_175.out`

## Conclusion

### 🎯 Answer: **YES, we reach 460 GB/s!**

- **Threshold**: Starting at **1GB message size**
- **Performance**:
  - 1GB: ~467 GB/s (+7 GB/s above target)
  - 2GB: ~472 GB/s (+12 GB/s above target)
  - 4GB: ~476 GB/s (+16 GB/s above target)
  - 8GB: **~479 GB/s (+19 GB/s above target - PEAK)**

- **Consistency**: Both 2-node and 4-node achieve nearly identical results
- **Scalability**: Excellent - performance maintained across node counts

### 💡 Key Takeaway

For workloads requiring ≥460 GB/s bandwidth:
- **Minimum requirement**: 1GB+ message sizes
- **Optimal performance**: 2GB-8GB range
- **Expected bandwidth**: 472-479 GB/s

**Date**: January 29, 2026
**Tool**: nccl-tests (all_reduce_perf)
**Configuration**: NVLS enabled with optimized settings
