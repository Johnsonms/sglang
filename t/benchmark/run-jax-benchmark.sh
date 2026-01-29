#!/bin/bash
# JAX All-Reduce Benchmark Runner with NVLS Configuration

cd /home/johnson/benchmark

# Activate the JAX virtual environment
source jax-env/bin/activate

# Set NVLS configuration for optimal performance (460+ GB/s)
export NCCL_NVLS_ENABLE=2              # Force enable NVLS
export NCCL_COLLNET_ENABLE=1           # Enable collective network
export NCCL_ALGO=Ring,Tree,CollNetDirect,CollNetChain,NVLS,NVLSTree
export NCCL_NET_GDR_LEVEL=5            # GPU Direct RDMA
export NCCL_P2P_LEVEL=NVL              # NVLink for P2P
export NCCL_NVLS_NCHANNELS=16          # 16 NVLS channels
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=32
export NCCL_NTHREADS=1024
export NCCL_BUFFSIZE=16777216
export NCCL_PROTO=Simple

# Optional: Enable NCCL debug output
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=INIT,ENV,NVLS

echo "========================================"
echo "JAX All-Reduce Benchmark with NVLS"
echo "========================================"
echo "NCCL Configuration:"
env | grep NCCL_ | sort
echo "========================================"
echo ""

# Run the benchmark
python3 physical-intelligence-run-complete.py

echo ""
echo "========================================"
echo "Benchmark completed!"
echo "========================================"
