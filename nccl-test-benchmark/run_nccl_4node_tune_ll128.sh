#!/bin/bash
#SBATCH -N 4
#SBATCH --nodelist=gpu-dp-x2vff-zrg5j,gpu-dp-x2vff-qfz7z,gpu-dp-x2vff-xjx42,gpu-dp-x2vff-7gdx4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --job-name=nccl_tune_ll128
#SBATCH --time=00:20:00
#SBATCH --output=nccl_tune_ll128_%j.out

# ============================================
# NCCL with LL128 Protocol Focus
# ============================================

# Network optimizations
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5
export NCCL_IB_TC=106
export NCCL_IB_TIMEOUT=22
export NCCL_NET_GDR_LEVEL=5
export NCCL_NET_GDR_READ=1

# Try LL128 protocol which can be faster for certain sizes
export NCCL_PROTO=LL128,Simple

# LL128 specific tuning
export NCCL_LL128_BUFFSIZE=33554432   # 32MB for LL128
export NCCL_LL128_NTHREADS=1024       # More threads for LL128

# CollNet
export NCCL_COLLNET_ENABLE=1
export NCCL_ALGO=Ring,Tree,CollNetDirect,CollNetChain

# Maximum channels
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=32
export NCCL_NTHREADS=1024
export NCCL_BUFFSIZE=33554432          # 32MB

# P2P and topology
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_CROSS_NIC=2

# InfiniBand optimizations
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_SPLIT_DATA_ON_QPS=1
export NCCL_IB_AR_THRESHOLD=8192

# Debug
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV,TUNING

echo "============================================"
echo "NCCL LL128 Protocol Test - Target: 420 GB/s"
echo "Job started: $(date)"
echo "Nodes: $SLURM_NODELIST"
echo "============================================"
echo "NCCL Configuration:"
env | grep NCCL_ | sort
echo "============================================"
echo ""

# Test large messages where LL128 should excel
srun --cpu-bind=none ./build/all_reduce_perf -b 128M -e 4G -f 2 -g 8

echo ""
echo "============================================"
echo "Job finished: $(date)"
echo "============================================"
