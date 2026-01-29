#!/bin/bash
#SBATCH -N 4                           # 4 nodes
#SBATCH --nodelist=gpu-dp-x2vff-zrg5j,gpu-dp-x2vff-qfz7z,gpu-dp-x2vff-xjx42,gpu-dp-x2vff-7gdx4
#SBATCH --ntasks-per-node=1            # 1 task per node
#SBATCH --gpus-per-node=8              # 8 GPUs per node
#SBATCH --job-name=jax_4node           # Job name
#SBATCH --time=00:20:00                # 20 minutes
#SBATCH --output=jax_benchmark_4node_%j.out
#SBATCH --error=jax_benchmark_4node_%j.out

# ============================================
# JAX All-Reduce Benchmark - 4 Nodes (32 GPUs)
# ============================================

echo "============================================"
echo "JAX 4-Node Benchmark Job Started"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"
echo "Time: $(date)"
echo "============================================"

# Change to benchmark directory
cd /home/johnson/benchmark

# Add uv to PATH
export PATH="$HOME/.local/bin:$PATH"

# Activate JAX environment
source jax-env/bin/activate

# Set NVLS configuration
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

# JAX multi-node settings
export JAX_COORDINATOR_ADDRESS=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export JAX_NUM_PROCESSES=$SLURM_NTASKS
export JAX_PROCESS_INDEX=$SLURM_PROCID

echo ""
echo "JAX Configuration:"
echo "Coordinator: $JAX_COORDINATOR_ADDRESS"
echo "Num processes: $JAX_NUM_PROCESSES"
echo "Process index: $JAX_PROCESS_INDEX"
echo ""
echo "NCCL Configuration:"
env | grep NCCL_ | sort
echo ""
echo "============================================"
echo "Starting JAX 4-Node Benchmark..."
echo "============================================"
echo ""

# Run with srun for multi-node
srun --cpu-bind=none python3 physical-intelligence-run-complete.py

echo ""
echo "============================================"
echo "Job Completed: $(date)"
echo "============================================"
