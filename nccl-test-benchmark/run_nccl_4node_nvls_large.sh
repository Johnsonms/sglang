#!/bin/bash
#SBATCH -N 4
#SBATCH --nodelist=gpu-dp-x2vff-zrg5j,gpu-dp-x2vff-qfz7z,gpu-dp-x2vff-xjx42,gpu-dp-x2vff-7gdx4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --job-name=nccl_4n_large
#SBATCH --time=00:15:00
#SBATCH --output=nccl_4node_nvls_large_%j.out

# ============================================
# 4-Node NVLS - Large Messages (2G-8G)
# ============================================

# Network optimizations
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5
export NCCL_IB_TC=106
export NCCL_IB_TIMEOUT=22
export NCCL_NET_GDR_LEVEL=5
export NCCL_NET_GDR_READ=1

# NVLS - NVLink Sharp
export NCCL_NVLS_ENABLE=2
export NCCL_NVLS_NCHANNELS=16
export NCCL_NVLS_CHUNKSIZE=131072

# CollNet
export NCCL_COLLNET_ENABLE=1
export NCCL_ALGO=Ring,Tree,CollNetDirect,CollNetChain,NVLS,NVLSTree

# Protocol
export NCCL_PROTO=Simple

# Aggressive settings
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=32
export NCCL_NTHREADS=1024
export NCCL_BUFFSIZE=16777216

# P2P and topology
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_CROSS_NIC=2

# InfiniBand tuning
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_SPLIT_DATA_ON_QPS=1

# Debug
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV

echo "============================================"
echo "4-Node NVLS - Large Messages (2G-8G)"
echo "Job started: $(date)"
echo "Nodes: $SLURM_NODELIST"
echo "============================================"

# Test large messages: 2G to 8G
srun --cpu-bind=none ./build/all_reduce_perf -b 2G -e 8G -f 2 -g 8

echo ""
echo "============================================"
echo "Job finished: $(date)"
echo "============================================"
