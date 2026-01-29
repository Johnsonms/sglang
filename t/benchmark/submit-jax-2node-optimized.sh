#!/bin/bash
#SBATCH -N 2
#SBATCH --nodelist=gpu-dp-x2vff-zrg5j,gpu-dp-x2vff-qfz7z
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --job-name=jax_2n_opt
#SBATCH --time=00:15:00
#SBATCH --output=jax_2node_optimized_%j.out
#SBATCH --error=jax_2node_optimized_%j.out

# ============================================
# JAX 2-Node Benchmark - Optimized (2GB & 4GB only)
# ============================================

echo "============================================"
echo "JAX 2-Node Optimized Benchmark"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"
echo "Time: $(date)"
echo "============================================"

cd /home/johnson/benchmark

# Activate environment
export PATH="$HOME/.local/bin:$PATH"
source jax-env/bin/activate

# NVLS Configuration
export NCCL_NVLS_ENABLE=2
export NCCL_COLLNET_ENABLE=1
export NCCL_ALGO=Ring,Tree,CollNetDirect,CollNetChain,NVLS,NVLSTree
export NCCL_NET_GDR_LEVEL=5
export NCCL_P2P_LEVEL=NVL
export NCCL_NVLS_NCHANNELS=16
export NCCL_NVLS_CHUNKSIZE=131072
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=32
export NCCL_NTHREADS=1024
export NCCL_BUFFSIZE=16777216
export NCCL_PROTO=Simple

# InfiniBand
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5
export NCCL_IB_TC=106
export NCCL_IB_TIMEOUT=22
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_SPLIT_DATA_ON_QPS=1
export NCCL_NET_GDR_READ=1

# P2P
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_CROSS_NIC=2

# Suppress warnings for cleaner output
export TF_CPP_MIN_LOG_LEVEL=2

# JAX multi-node
export JAX_COORDINATOR_ADDRESS=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export JAX_NUM_PROCESSES=$SLURM_NTASKS
export JAX_PROCESS_INDEX=$SLURM_PROCID

echo ""
echo "Running optimized benchmark (2GB & 4GB buffers only)..."
echo ""

# Run the optimized benchmark
srun --cpu-bind=none python3 physical-intelligence-run-2-4gb.py

echo ""
echo "============================================"
echo "Job Completed: $(date)"
echo "============================================"
