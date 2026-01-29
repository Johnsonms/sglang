# JAX All-Reduce Benchmark Guide

## Quick Start

### Option 1: Run with Script (Recommended)
```bash
cd /home/johnson/benchmark
./run-jax-benchmark.sh
```

### Option 2: Run Manually
```bash
cd /home/johnson/benchmark
source jax-env/bin/activate
python3 physical-intelligence-run-complete.py
```

### Option 3: Run with Custom NVLS Settings
```bash
cd /home/johnson/benchmark
source jax-env/bin/activate

# Set NVLS configuration
export NCCL_NVLS_ENABLE=2
export NCCL_COLLNET_ENABLE=1
export NCCL_ALGO=Ring,Tree,CollNetDirect,CollNetChain,NVLS,NVLSTree
export NCCL_NET_GDR_LEVEL=5
export NCCL_P2P_LEVEL=NVL

python3 physical-intelligence-run-complete.py
```

## Expected Performance

With NVLS enabled on 8x H100 GPUs:
- **2GB buffer**: ~470 GB/s bus bandwidth
- **4GB buffer**: ~475 GB/s bus bandwidth
- **Target**: 460+ GB/s ✅

## Files

- `physical-intelligence-run-complete.py` - Main benchmark script
- `run-jax-benchmark.sh` - Run script with NVLS configuration
- `jax-env/` - Python virtual environment with JAX
- `jax-benchmark-output.txt` - Latest benchmark results

## Troubleshooting

### Out of Memory Error
If you get OOM for 8GB buffers, edit the script to test smaller sizes:
```python
buffer_sizes = [2.0, 4.0]  # Remove 8.0
```

### No GPUs Detected
Check GPU availability:
```bash
source jax-env/bin/activate
python3 -c "import jax; print(jax.devices())"
```

### NVLS Not Working
Enable debug output to verify NVLS is active:
```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV,NVLS
./run-jax-benchmark.sh
```

## Multi-Node Setup

For multi-node benchmarks, you'll need to:
1. Set up JAX distributed initialization
2. Configure SLURM or similar job scheduler
3. Ensure InfiniBand connectivity between nodes

Example multi-node configuration coming soon.

## Customizing Buffer Sizes

Edit `physical-intelligence-run-complete.py` line 138:
```python
buffer_sizes = [1.0, 2.0, 4.0, 6.0]  # GB per device
```

## Environment Details

- Python: 3.10.12
- JAX: 0.6.2
- CUDA: 12.9
- NCCL: 2.29.2
- Hardware: 8x NVIDIA H100 80GB HBM3

## Performance Comparison

| Framework | Buffer Size | Bus Bandwidth |
|-----------|-------------|---------------|
| NCCL (native) | 4GB | 476 GB/s |
| **JAX** | **4GB** | **475 GB/s** |
| Target | - | 460 GB/s |

JAX achieves native NCCL performance when properly configured!
