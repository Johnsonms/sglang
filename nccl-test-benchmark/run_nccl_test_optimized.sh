#!/bin/bash
#SBATCH -N 4                          # Number of nodes
#SBATCH --nodelist=gpu-dp-x2vff-zrg5j,gpu-dp-x2vff-qfz7z,gpu-dp-x2vff-xjx42,gpu-dp-x2vff-7gdx4
#SBATCH --ntasks-per-node=1           # 1 task per node (each uses 8 GPUs)
#SBATCH --gpus-per-node=8             # Request 8 GPUs per node
#SBATCH --job-name=nccl_test_opt
#SBATCH --time=00:30:00               # Time limit
#SBATCH --output=nccl_test_opt_%j.out # Output file

# ============================================
# NCCL Performance Optimization Settings
# ============================================

# Network optimizations
export NCCL_IB_DISABLE=0              # Enable InfiniBand
export NCCL_IB_GID_INDEX=3            # RoCE v2 (change to 0 for IB if needed)
export NCCL_IB_HCA=mlx5               # Use all mlx5 adapters
export NCCL_IB_TC=106                 # Traffic class for lossless traffic
export NCCL_IB_TIMEOUT=22             # Timeout value
export NCCL_NET_GDR_LEVEL=5           # Enable GPUDirect RDMA (5 = all)
export NCCL_NET_GDR_READ=1            # Enable GPUDirect RDMA reads

# Algorithm selection
export NCCL_ALGO=Ring,Tree            # Use both Ring and Tree algorithms
export NCCL_PROTO=Simple              # Simple protocol is fastest for large messages

# NVLink/P2P optimizations
export NCCL_P2P_LEVEL=NVL             # Force NVLink for intra-node
export NCCL_P2P_DISABLE=0             # Enable P2P
export NCCL_SHM_DISABLE=0             # Enable shared memory

# Buffer sizes
export NCCL_BUFFSIZE=8388608          # 8MB buffer size (good for large messages)
export NCCL_NTHREADS=640              # Number of CUDA threads (640 for H100)

# Topology and tuning
export NCCL_CROSS_NIC=1               # Allow cross-NIC traffic
export NCCL_MIN_NCHANNELS=8           # Minimum channels (match number of NICs)
export NCCL_MAX_NCHANNELS=16          # Maximum channels

# Debug (set to WARN for production)
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV

# Print configuration
echo "============================================"
echo "Job started at: $(date)"
echo "Nodes: $SLURM_NODELIST"
echo "Total tasks: $SLURM_NTASKS"
echo "============================================"
echo "NCCL Configuration:"
env | grep NCCL_ | sort
echo "============================================"
echo ""

# Run NCCL test - each task will use 8 GPUs (-g 8)
# Test from 8B to 2GB for comprehensive bandwidth analysis
srun --cpu-bind=none ./build/all_reduce_perf -b 8 -e 2G -f 2 -g 8

echo ""
echo "============================================"
echo "Job finished at: $(date)"
echo "============================================"
