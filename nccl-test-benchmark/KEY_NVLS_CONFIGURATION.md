# Key NVLS Configuration for 460+ GB/s Performance

## Configuration Summary

This document summarizes the critical NCCL configuration settings that enabled **460-479 GB/s** bus bandwidth performance with NVLS on H100 systems.

## Hardware Setup

```
GPUs per Node:    8x NVIDIA H100 80GB HBM3
Nodes Tested:     1-4 nodes (8-32 GPUs total)
Interconnect:     InfiniBand with NVLS support
GPU Topology:     NVLink connected within node
```

## Critical NVLS Settings

### 1. NVLS Core Configuration

```bash
export NCCL_NVLS_ENABLE=2              # Force enable NVLS (critical!)
export NCCL_NVLS_NCHANNELS=16          # Number of NVLS channels
export NCCL_NVLS_CHUNKSIZE=131072      # NVLS chunk size (128KB)
```

**Why Critical:**
- `NVLS_ENABLE=2` forces NVLS usage (vs. 1 which is auto-detect)
- 16 channels maximizes parallelism for H100 architecture
- Chunk size optimized for H100 memory bandwidth

### 2. CollNet Configuration

```bash
export NCCL_COLLNET_ENABLE=1           # Enable collective network operations
export NCCL_ALGO=Ring,Tree,CollNetDirect,CollNetChain,NVLS,NVLSTree
```

**Why Critical:**
- CollNet enables optimized collective algorithms
- Including NVLS and NVLSTree algorithms is essential for NVLS performance

### 3. Protocol Selection

```bash
export NCCL_PROTO=Simple               # Use Simple protocol
```

**Why Critical:**
- Simple protocol provides best performance for large messages
- Required for optimal NVLS operation

### 4. Channel and Thread Configuration

```bash
export NCCL_MIN_NCHANNELS=16           # Minimum 16 channels
export NCCL_MAX_NCHANNELS=32           # Maximum 32 channels
export NCCL_NTHREADS=1024              # Threads per channel
export NCCL_BUFFSIZE=16777216          # 16MB buffer size
```

**Why Critical:**
- More channels = more parallelism
- 1024 threads maximizes GPU utilization
- Large buffer size reduces overhead for big messages

### 5. InfiniBand Optimizations

```bash
export NCCL_IB_DISABLE=0               # Enable InfiniBand
export NCCL_IB_GID_INDEX=3             # Global ID index for IB
export NCCL_IB_HCA=mlx5                # Mellanox adapter
export NCCL_IB_TC=106                  # Traffic class
export NCCL_IB_TIMEOUT=22              # Timeout value
export NCCL_IB_QPS_PER_CONNECTION=4    # Queue pairs per connection
export NCCL_IB_SPLIT_DATA_ON_QPS=1     # Split data across QPs
```

**Why Critical:**
- Optimizes InfiniBand for low latency
- Multiple QPs per connection increases bandwidth
- Traffic class 106 provides QoS priority

### 6. GPU Direct RDMA (GDR)

```bash
export NCCL_NET_GDR_LEVEL=5            # Maximum GDR optimization
export NCCL_NET_GDR_READ=1             # Enable GDR read operations
```

**Why Critical:**
- Level 5 enables full GPU Direct RDMA
- Allows direct GPU-to-GPU transfers via InfiniBand
- Eliminates CPU bottlenecks

### 7. P2P and Topology Settings

```bash
export NCCL_P2P_LEVEL=NVL              # Use NVLink for P2P
export NCCL_P2P_DISABLE=0              # Enable P2P
export NCCL_SHM_DISABLE=0              # Enable shared memory
export NCCL_CROSS_NIC=2                # Cross-NIC communication level
```

**Why Critical:**
- NVL level forces NVLink usage within nodes
- Cross-NIC=2 optimizes multi-NIC communication
- P2P and SHM enable fast intra-node transfers

### 8. Debug Settings (Optional)

```bash
export NCCL_DEBUG=INFO                 # For monitoring
export NCCL_DEBUG_SUBSYS=INIT,ENV,NVLS # Debug specific subsystems
```

## Complete Configuration Script

```bash
#!/bin/bash

# ============================================
# NVLS Optimized Configuration for H100
# Target: 460+ GB/s Bus Bandwidth
# ============================================

# NVLS - NVLink Sharp (H100 feature) - CRITICAL!
export NCCL_NVLS_ENABLE=2              # Force enable
export NCCL_NVLS_NCHANNELS=16          # NVLS channels
export NCCL_NVLS_CHUNKSIZE=131072      # NVLS chunk size

# CollNet - CRITICAL!
export NCCL_COLLNET_ENABLE=1
export NCCL_ALGO=Ring,Tree,CollNetDirect,CollNetChain,NVLS,NVLSTree

# Protocol
export NCCL_PROTO=Simple

# Channel and Thread Configuration
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=32
export NCCL_NTHREADS=1024
export NCCL_BUFFSIZE=16777216

# InfiniBand Optimizations
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5
export NCCL_IB_TC=106
export NCCL_IB_TIMEOUT=22
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_SPLIT_DATA_ON_QPS=1

# GPU Direct RDMA
export NCCL_NET_GDR_LEVEL=5
export NCCL_NET_GDR_READ=1

# P2P and Topology
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_CROSS_NIC=2

# Debug (optional - can be removed in production)
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV,NVLS
```

## Performance Results with This Configuration

| Message Size | Bus Bandwidth | Node Count |
|--------------|---------------|------------|
| 1GB | 467 GB/s | 2-4 nodes |
| 2GB | 472 GB/s | 2-4 nodes |
| 4GB | 476 GB/s | 2-4 nodes |
| **8GB** | **479 GB/s** | **2-4 nodes** |

## Most Critical Settings (Top 5)

### 1. **NCCL_NVLS_ENABLE=2** ⭐⭐⭐⭐⭐
   - Most important setting
   - Without this, NVLS won't be used
   - Must be set to 2 (force) for guaranteed activation

### 2. **NCCL_COLLNET_ENABLE=1** ⭐⭐⭐⭐⭐
   - Required for NVLS algorithms
   - Enables optimized collective operations

### 3. **NCCL_ALGO=...NVLS,NVLSTree** ⭐⭐⭐⭐⭐
   - Must explicitly include NVLS algorithms
   - Without NVLS in the list, it won't be used

### 4. **NCCL_NET_GDR_LEVEL=5** ⭐⭐⭐⭐
   - Enables full GPU Direct RDMA
   - Critical for cross-node performance

### 5. **NCCL_P2P_LEVEL=NVL** ⭐⭐⭐⭐
   - Forces NVLink usage within nodes
   - Essential for intra-node bandwidth

## Message Size Recommendations

```
For 460+ GB/s Performance:
├── Minimum: 1GB messages
├── Recommended: 2-4GB messages
└── Peak: 8GB messages (479 GB/s)

Avoid:
└── Messages < 1GB will not reach 460 GB/s
    └── 512M: ~437 GB/s (23 GB/s short)
```

## Verification Commands

### Check if NVLS is Active

```bash
# Run with debug enabled
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=NVLS

# Look for these lines in output:
# "NVLS comm ... nHeads 8 buffSize ... nvlsPerRankSize ... nvlsTotalSize ..."
# "Using NVLS algorithm"
```

### Check Bus Bandwidth

```bash
# In nccl-tests output, look for:
# "Avg bus bandwidth : XXX.XXX"
# Should be 460+ for messages ≥1GB
```

## Environment-Specific Notes

### For SLURM Environments

```bash
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1

srun --cpu-bind=none ./build/all_reduce_perf -b 1G -e 8G -f 2 -g 8
```

### For Multi-Node

```bash
# Ensure all nodes have:
# - Same NCCL version
# - InfiniBand connectivity
# - NVLS-capable GPUs (H100)
```

## Testing Your Configuration

```bash
# Quick test to verify 460+ GB/s
./build/all_reduce_perf -b 1G -e 4G -f 2 -g 8

# Expected results:
# 1GB: ~467 GB/s ✓
# 2GB: ~472 GB/s ✓
# 4GB: ~476 GB/s ✓
```

## Common Issues and Solutions

### Issue: Not reaching 460 GB/s

**Check:**
1. `NCCL_NVLS_ENABLE=2` is set
2. Message size is ≥1GB
3. `NCCL_ALGO` includes "NVLS,NVLSTree"
4. `NCCL_COLLNET_ENABLE=1` is set

### Issue: NVLS not detected

**Solution:**
```bash
# Verify GPU supports NVLS
nvidia-smi topo -m  # Should show NVLink connections

# Check NCCL version
# Requires NCCL 2.18+ for NVLS support
```

### Issue: Performance varies

**Check:**
- GPU utilization (should be high)
- Network congestion
- Other jobs on the same nodes

## Version Information

```
NCCL Version:  2.17.8+
CUDA Version:  12.0+
Driver Version: 535.0+
Hardware:      NVIDIA H100 80GB HBM3
```

## References

- Test scripts: `/home/johnson/nccl-tests/run_nccl_*node_nvls*.sh`
- Results: `/home/johnson/nccl-tests/NVLS_MESSAGE_SIZE_ANALYSIS.md`
- Summary: `/home/johnson/nccl-tests/NVLS_BENCHMARK_SUMMARY.md`

---

**Last Updated**: January 29, 2026
**Validated Performance**: 460-479 GB/s on 2-4 node H100 systems
