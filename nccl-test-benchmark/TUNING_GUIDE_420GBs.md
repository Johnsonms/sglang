# NCCL Tuning Guide - Target: 420 GB/s Bus Bandwidth

## Current Status
- **Current Peak**: 347 GB/s bus bandwidth
- **Target**: 420 GB/s bus bandwidth
- **Gap**: +73 GB/s (+21% improvement needed)

## Tuning Strategy

### 1. Key Parameters to Optimize

#### Channels
```bash
export NCCL_MIN_NCHANNELS=16    # Increase from 8
export NCCL_MAX_NCHANNELS=32    # Increase from 16
```
More channels = more parallel communication paths = higher bandwidth

#### Threads
```bash
export NCCL_NTHREADS=1024       # Increase from 640
```
H100 can handle more threads efficiently

#### Buffer Sizes
```bash
export NCCL_BUFFSIZE=16777216   # 16MB (double from 8MB)
export NCCL_LL128_BUFFSIZE=33554432  # 32MB for LL128
```
Larger buffers reduce overhead for large messages

#### Cross-NIC Traffic
```bash
export NCCL_CROSS_NIC=2         # More aggressive (was 1)
```
Better load balancing across multiple NICs

#### InfiniBand QP Tuning
```bash
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_SPLIT_DATA_ON_QPS=1
export NCCL_IB_AR_THRESHOLD=8192
```
Better InfiniBand utilization

### 2. Protocol Selection

#### LL128 Protocol
- Can be faster than Simple for certain message sizes
- Lower latency overhead
- Better for 128MB-4GB range

```bash
export NCCL_PROTO=LL128,Simple
```

#### NVLS (NVLink Sharp)
- H100 feature for accelerated collectives
- Combines NVLink and network optimizations
- May provide significant speedup

```bash
export NCCL_NVLS_ENABLE=2
export NCCL_ALGO=Ring,Tree,CollNetDirect,CollNetChain,NVLS,NVLSTree
```

### 3. Test Configurations Created

I've created 3 tuning scripts to test different approaches:

#### Script 1: `run_nccl_4node_tune_420.sh`
**Focus**: General aggressive tuning
- 32 max channels
- 1024 threads
- 16MB buffers
- Multi-protocol (Simple + LL128)
- Enhanced IB settings

#### Script 2: `run_nccl_4node_tune_nvls.sh`
**Focus**: NVLS (NVLink Sharp) optimization
- NVLS algorithms enabled
- NVLS-specific tuning
- May leverage H100's advanced interconnect

#### Script 3: `run_nccl_4node_tune_ll128.sh`
**Focus**: LL128 protocol optimization
- Prioritizes LL128 protocol
- 32MB LL128 buffers
- Optimized for large message throughput

## Expected Results

### Realistic Targets by Configuration

| Configuration | Expected Bus BW | Notes |
|---------------|-----------------|-------|
| Baseline (current) | 347 GB/s | CollNet enabled |
| Aggressive tuning | 380-400 GB/s | More channels + buffers |
| NVLS enabled | 400-430 GB/s | If H100 NVLS works well |
| LL128 optimized | 370-410 GB/s | Protocol-dependent |

### Best Case Scenario
- **420+ GB/s** achievable if:
  - NVLS works optimally on H100
  - Network has capacity for 32 channels
  - No hardware bottlenecks
  - Large messages (2-4GB) fully saturate links

### Potential Limiting Factors

1. **Physical Network Bandwidth**
   - Your InfiniBand speed (200/400 Gbps per link?)
   - Number of IB links per GPU
   - Switch bandwidth/blocking ratio

2. **GPU PCIe/NVLink Bandwidth**
   - H100 PCIe Gen5: ~128 GB/s per direction
   - NVLink bandwidth: 900 GB/s total per GPU
   - Should not be the bottleneck

3. **Memory Bandwidth**
   - H100 HBM3: 3.35 TB/s
   - Should not be the bottleneck for network I/O

## How to Run Tests

```bash
# Make scripts executable
chmod +x run_nccl_4node_tune_*.sh

# Run all three tuning tests
sbatch run_nccl_4node_tune_420.sh
sbatch run_nccl_4node_tune_nvls.sh
sbatch run_nccl_4node_tune_ll128.sh

# Monitor progress
watch -n 5 'squeue -u $USER'
```

## Analyzing Results

After tests complete, check:

```bash
# Find peak bus bandwidth in each test
grep "Avg bus bandwidth" nccl_tune_*_*.out

# Check which algorithm/protocol was used
grep "Using.*for sizes" nccl_tune_*_*.out

# Look at large message performance
grep -A 5 "2147483648" nccl_tune_*_*.out
grep -A 5 "4294967296" nccl_tune_*_*.out
```

## If 420 GB/s Still Not Achieved

### Additional Tuning Options

1. **Check Network Topology**
```bash
export NCCL_TOPO_DUMP_FILE=topology.xml
```
Review if NCCL sees all network paths

2. **Force Specific Algorithms**
```bash
export NCCL_ALGO=CollNetChain  # Force CollNet only
# or
export NCCL_ALGO=NVLS          # Force NVLS if available
```

3. **Disable Slower Protocols**
```bash
export NCCL_PROTO=Simple  # Disable LL/LL128 if they're slower
```

4. **Tune Per-Operation**
Different operations may need different settings:
```bash
export NCCL_ALLREDUCE_ALGO=NVLS
export NCCL_REDUCE_ALGO=Ring
```

5. **Check Hardware Limits**
```bash
# Verify IB link speeds
ibstat

# Check for IB errors
ibdiagnet

# Monitor during test
nv-hostengine -n  # NVIDIA DCGM for GPU metrics
```

## Troubleshooting

### If Performance Decreases
- Too many channels can cause overhead
- Try reducing: `NCCL_MAX_NCHANNELS=24`
- Reduce threads: `NCCL_NTHREADS=768`

### If Seeing Errors
- "Out of memory": Reduce `NCCL_BUFFSIZE`
- "Timeout": Increase `NCCL_IB_TIMEOUT=30`
- "NVLS not available": Your hardware may not support it

### If Bandwidth Plateaus
- You may be hitting physical network limits
- Check: `ibstat` for actual link speeds
- Verify: Number of IB connections per GPU
- Confirm: No network contention from other jobs

## Next Steps

1. Run the three tuning scripts
2. Compare results
3. Identify the best configuration
4. Fine-tune the winner
5. If still short of 420 GB/s, we'll investigate hardware topology

Let me know the results and we'll iterate!
