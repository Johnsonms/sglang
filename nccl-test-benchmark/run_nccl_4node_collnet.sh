#!/bin/bash
#SBATCH -N 4
#SBATCH --nodelist=gpu-dp-x2vff-zrg5j,gpu-dp-x2vff-qfz7z,gpu-dp-x2vff-xjx42,gpu-dp-x2vff-7gdx4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --job-name=nccl_4n_collnet
#SBATCH --time=00:15:00
#SBATCH --output=nccl_4node_collnet_%j.out

# ============================================
# CollNet-Enabled NCCL Configuration
# ============================================

# Network optimizations
export NCCL_IB_DISABLE=0              # Enable InfiniBand
export NCCL_IB_GID_INDEX=3            # RoCE v2
export NCCL_IB_HCA=mlx5               # Use all mlx5 adapters
export NCCL_IB_TC=106                 # Traffic class
export NCCL_IB_TIMEOUT=22             # Timeout value
export NCCL_NET_GDR_LEVEL=5           # Enable GPUDirect RDMA
export NCCL_NET_GDR_READ=1            # Enable GPUDirect RDMA reads

# CollNet configuration
export NCCL_COLLNET_ENABLE=1          # Enable CollNet algorithms
export NCCL_ALGO=Ring,Tree,CollNetDirect,CollNetChain  # Include CollNet algorithms

# Protocol selection
export NCCL_PROTO=Simple              # Simple protocol for large messages

# NVLink/P2P optimizations
export NCCL_P2P_LEVEL=NVL             # Force NVLink for intra-node
export NCCL_P2P_DISABLE=0             # Enable P2P
export NCCL_SHM_DISABLE=0             # Enable shared memory

# Buffer and channel settings
export NCCL_BUFFSIZE=8388608          # 8MB buffer size
export NCCL_NTHREADS=640              # Number of CUDA threads
export NCCL_CROSS_NIC=1               # Allow cross-NIC traffic
export NCCL_MIN_NCHANNELS=8           # Minimum channels
export NCCL_MAX_NCHANNELS=16          # Maximum channels

# Debug
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV

echo "============================================"
echo "4-Node NCCL Test with CollNet"
echo "Job started: $(date)"
echo "Nodes: $SLURM_NODELIST"
echo "============================================"
echo "NCCL Configuration:"
env | grep NCCL_ | sort
echo "============================================"
echo ""

# Test from 8B to 2GB for comprehensive analysis
srun --cpu-bind=none ./build/all_reduce_perf -b 8 -e 2G -f 2 -g 8

echo ""
echo "============================================"
echo "Job finished: $(date)"
echo "============================================"
