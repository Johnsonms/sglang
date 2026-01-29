#!/bin/bash
#SBATCH -N 4
#SBATCH --nodelist=gpu-dp-x2vff-zrg5j,gpu-dp-x2vff-qfz7z,gpu-dp-x2vff-xjx42,gpu-dp-x2vff-7gdx4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --job-name=nccl_tune_420
#SBATCH --time=00:20:00
#SBATCH --output=nccl_tune_420_%j.out

# ============================================
# NCCL Tuning for 420 GB/s Target
# ============================================

# Network optimizations - More aggressive settings
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5
export NCCL_IB_TC=106
export NCCL_IB_TIMEOUT=22

# Try all protocols to find the fastest
export NCCL_PROTO=Simple,LL128     # Add LL128 for potentially better perf

# GPUDirect RDMA - maximum settings
export NCCL_NET_GDR_LEVEL=5
export NCCL_NET_GDR_READ=1

# CollNet with all available algorithms
export NCCL_COLLNET_ENABLE=1
export NCCL_ALGO=Ring,Tree,CollNetDirect,CollNetChain

# Aggressive channel and thread settings
export NCCL_MIN_NCHANNELS=16       # Increased from 8
export NCCL_MAX_NCHANNELS=32       # Increased from 16
export NCCL_NTHREADS=1024          # Increased from 640 for H100

# Larger buffer for better throughput
export NCCL_BUFFSIZE=16777216      # 16MB (doubled from 8MB)

# P2P and topology optimizations
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_CROSS_NIC=2            # More aggressive cross-NIC (was 1)

# InfiniBand specific optimizations
export NCCL_IB_QPS_PER_CONNECTION=4   # More QPs per connection
export NCCL_IB_SPLIT_DATA_ON_QPS=1    # Split data across QPs
export NCCL_IB_AR_THRESHOLD=8192      # Adaptive routing threshold

# NVLS (NVLink Sharp) if available on H100
export NCCL_NVLS_ENABLE=1

# Tuning for large messages
export NCCL_LL128_BUFFSIZE=16777216
export NCCL_LL128_NTHREADS=640

# Debug
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV,TUNING

echo "============================================"
echo "NCCL Tuning Test - Target: 420 GB/s"
echo "Job started: $(date)"
echo "Nodes: $SLURM_NODELIST"
echo "============================================"
echo "NCCL Configuration:"
env | grep NCCL_ | sort
echo "============================================"
echo ""

# Focus on large messages where we should see peak bandwidth
srun --cpu-bind=none ./build/all_reduce_perf -b 128M -e 4G -f 2 -g 8

echo ""
echo "============================================"
echo "Job finished: $(date)"
echo "============================================"
