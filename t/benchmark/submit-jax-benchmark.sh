#!/bin/bash
#SBATCH -N 1                           # 1 node
#SBATCH --ntasks-per-node=1            # 1 task per node
#SBATCH --gpus-per-node=8              # 8 GPUs
#SBATCH --job-name=jax_benchmark       # Job name
#SBATCH --time=00:15:00                # 15 minutes
#SBATCH --output=jax_benchmark_%j.out  # Output file
#SBATCH --error=jax_benchmark_%j.out   # Merge stderr to stdout

# ============================================
# JAX All-Reduce Benchmark with NVLS
# ============================================

echo "============================================"
echo "JAX Benchmark Job Started"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"
echo "============================================"

# Change to benchmark directory
cd /home/johnson/benchmark

# Add uv to PATH
export PATH="$HOME/.local/bin:$PATH"

# Activate JAX environment
source jax-env/bin/activate

# Set NVLS configuration for optimal performance
export NCCL_NVLS_ENABLE=2              # Force enable NVLS
export NCCL_COLLNET_ENABLE=1           # Enable collective network
export NCCL_ALGO=Ring,Tree,CollNetDirect,CollNetChain,NVLS,NVLSTree
export NCCL_NET_GDR_LEVEL=5            # GPU Direct RDMA
export NCCL_P2P_LEVEL=NVL              # NVLink for P2P
export NCCL_NVLS_NCHANNELS=16          # 16 NVLS channels
export NCCL_NVLS_CHUNKSIZE=131072      # 128KB chunks
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=32
export NCCL_NTHREADS=1024
export NCCL_BUFFSIZE=16777216
export NCCL_PROTO=Simple

# InfiniBand settings
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5
export NCCL_IB_TC=106
export NCCL_IB_TIMEOUT=22
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_SPLIT_DATA_ON_QPS=1
export NCCL_NET_GDR_READ=1

# P2P settings
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_CROSS_NIC=2

# Optional: Enable debug output
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=INIT,ENV,NVLS

echo ""
echo "NCCL Configuration:"
env | grep NCCL_ | sort
echo ""
echo "============================================"
echo "Starting JAX Benchmark..."
echo "============================================"
echo ""

# Run the benchmark
python3 physical-intelligence-run-complete.py

echo ""
echo "============================================"
echo "Job Completed: $(date)"
echo "============================================"
