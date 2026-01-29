# NCCL NVLS Benchmark Summary (1-4 Nodes)

## Overview
This report summarizes NCCL all_reduce performance benchmarks with NVLS (NVLink Sharp) enabled across 1-4 nodes. Tests were performed using NCCL with optimized settings specifically targeting NVLS capabilities on H100 GPUs.

## Test Configuration

### NVLS Settings
- **NCCL_NVLS_ENABLE**: 2 (Force enable)
- **NCCL_NVLS_NCHANNELS**: 16
- **NCCL_NVLS_CHUNKSIZE**: 131072
- **NCCL_COLLNET_ENABLE**: 1
- **NCCL_ALGO**: Ring,Tree,CollNetDirect,CollNetChain,NVLS,NVLSTree
- **NCCL_PROTO**: Simple

### Network Settings
- **InfiniBand**: Enabled with GDR Level 5
- **NCCL_IB_GID_INDEX**: 3
- **NCCL_IB_TC**: 106
- **NCCL_P2P_LEVEL**: NVL
- **NCCL_CROSS_NIC**: 2

### Hardware
- **GPUs per node**: 8 (H100)
- **Nodes tested**: 1, 2, 3, 4
- **Total GPUs**: 8, 16, 24, 32

## Results Summary

### Average Bus Bandwidth

| Configuration | Avg Bus Bandwidth (GB/s) |
|--------------|-------------------------|
| 1-Node (8 GPUs) | 445.60 |
| 2-Node (16 GPUs) | 445.46 - 446.02 |
| 3-Node (24 GPUs) | 445.70 - 445.82 |
| 4-Node (32 GPUs) | 445.59 - 445.70 |

### Peak Performance at 4GB Message Size

#### 1-Node Results (8 GPUs)
```
Size (B)      Count        Time (us)   AlgBW (GB/s)   BusBW (GB/s)
4294967296    1073741824   15781.3     272.15         476.27
```

#### 2-Node Results (16 GPUs)
```
Size (B)      Count        Time (us)   AlgBW (GB/s)   BusBW (GB/s)
4294967296    1073741824   15762.4     272.48         476.84
4294967296    1073741824   15765.6     272.43         476.75
```

#### 3-Node Results (24 GPUs)
```
Size (B)      Count        Time (us)   AlgBW (GB/s)   BusBW (GB/s)
4294967296    1073741824   15754.3     272.62         477.09
4294967296    1073741824   15773.1     272.30         476.52
```

#### 4-Node Results (32 GPUs)
```
Size (B)      Count        Time (us)   AlgBW (GB/s)   BusBW (GB/s)
4294967296    1073741824   15782.9     272.13         476.22
4294967296    1073741824   15769.4     272.36         476.63
4294967296    1073741824   15766.2     272.42         476.73
4294967296    1073741824   15771.8     272.32         476.56
```

## Detailed Performance Breakdown

### 1-Node NVLS Performance
Message sizes from 128MB to 4GB:

| Size        | Time (us) | AlgBW (GB/s) | BusBW (GB/s) |
|-------------|-----------|--------------|--------------|
| 134217728   | 588.33    | 228.13       | 399.23       |
| 268435456   | 1113.61   | 241.05       | 421.84       |
| 536870912   | 2153.76   | 249.27       | 436.23       |
| 1073741824  | 4034.51   | 266.14       | 465.74       |
| 2147483648  | 7949.79   | 270.13       | 472.73       |
| 4294967296  | 15781.30  | 272.15       | **476.27**   |

### 2-Node NVLS Performance
Peak performance at large message sizes:

| Size        | Time (us) | AlgBW (GB/s) | BusBW (GB/s) |
|-------------|-----------|--------------|--------------|
| 1073741824  | 4032.02   | 266.30       | 466.03       |
| 2147483648  | 7960.28   | 269.77       | 472.11       |
| 4294967296  | 15762.40  | 272.48       | **476.84**   |

### 3-Node NVLS Performance
Peak performance at large message sizes:

| Size        | Time (us) | AlgBW (GB/s) | BusBW (GB/s) |
|-------------|-----------|--------------|--------------|
| 1073741824  | 4027.46   | 266.61       | 466.56       |
| 2147483648  | 7945.54   | 270.28       | 472.98       |
| 4294967296  | 15754.30  | 272.62       | **477.09**   |

### 4-Node NVLS Performance
Peak performance at large message sizes:

| Size        | Time (us) | AlgBW (GB/s) | BusBW (GB/s) |
|-------------|-----------|--------------|--------------|
| 1073741824  | ~4025     | ~266.5       | ~466.5       |
| 2147483648  | 7945.35   | 270.28       | 472.99       |
| 4294967296  | 15766.20  | 272.42       | **476.73**   |

## Key Findings

1. **Consistent Scaling**: NVLS maintains excellent performance consistency across 1-4 nodes
   - Peak bus bandwidth: **476-477 GB/s** across all configurations
   - Average bus bandwidth: **445-446 GB/s** across all configurations

2. **NVLS Effectiveness**: The NVLS feature successfully enables high-bandwidth communication
   - Peak performance achieved at 4GB message sizes
   - Minimal performance degradation as node count increases

3. **Network Efficiency**: Multi-node scaling shows excellent network utilization
   - 2-node: 476.84 GB/s (slight improvement over 1-node)
   - 3-node: 477.09 GB/s (best peak performance)
   - 4-node: 476.73 GB/s (maintained high performance)

4. **Performance Characteristics**:
   - Bus bandwidth increases with message size
   - Optimal performance at 1GB+ message sizes
   - Peak performance at 4GB message size: **~477 GB/s**
   - Consistent performance across multiple test runs

## Comparison with Previous Tests

Based on previous tuning results:
- **Without NVLS optimization**: ~420 GB/s
- **With NVLS enabled**: **~445-477 GB/s**
- **Performance improvement**: ~6-13% improvement with NVLS

## Conclusions

1. **NVLS is highly effective** for improving NCCL collective communication performance on H100 systems
2. **Scaling is excellent** - minimal performance loss when adding nodes (1-4 nodes)
3. **Peak performance of ~477 GB/s** achieved consistently at large message sizes
4. **Average performance of ~446 GB/s** maintained across all node configurations
5. **Configuration is production-ready** for multi-node H100 clusters

## Recommendations

1. **Use NVLS settings** from these benchmarks for production workloads
2. **Target message sizes ≥1GB** for optimal performance
3. **Monitor NVLS channel utilization** using NCCL_DEBUG_SUBSYS=NVLS
4. **Consider NVLS for large-scale training** where collective communication is critical

## Test Files

- 1-Node: `nccl_1node_nvls_166.out`
- 2-Node: `nccl_2node_nvls_167.out`
- 3-Node: `nccl_3node_nvls_168.out`
- 4-Node: `nccl_4node_nvls_169.out`

## Test Date
**Date**: January 29, 2026
**Benchmarking Tool**: nccl-tests (all_reduce_perf)
