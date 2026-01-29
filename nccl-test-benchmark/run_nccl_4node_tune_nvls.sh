#!/bin/bash
#SBATCH -N 4
#SBATCH --nodelist=gpu-dp-x2vff-zrg5j,gpu-dp-x2vff-qfz7z,gpu-dp-x2vff-xjx42,gpu-dp-x2vff-7gdx4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --job-name=nccl_tune_nvls
#SBATCH --time=00:20:00
#SBATCH --output=nccl_tune_nvls_%j.out

# ============================================
# NCCL with NVLS (NVLink Sharp) Focus
# ============================================

# Network optimizations
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5
export NCCL_IB_TC=106
export NCCL_IB_TIMEOUT=22
export NCCL_NET_GDR_LEVEL=5
export NCCL_NET_GDR_READ=1

# NVLS - NVLink Sharp (H100 feature)
export NCCL_NVLS_ENABLE=2              # Force enable
export NCCL_NVLS_NCHANNELS=16          # NVLS channels
export NCCL_NVLS_CHUNKSIZE=131072      # NVLS chunk size

# CollNet
export NCCL_COLLNET_ENABLE=1
export NCCL_ALGO=Ring,Tree,CollNetDirect,CollNetChain,NVLS,NVLSTree

# Protocol selection
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
export NCCL_DEBUG_SUBSYS=INIT,ENV,TUNING,NVLS

echo "============================================"
echo "NCCL NVLS Test - Target: 420 GB/s"
echo "Job started: $(date)"
echo "Nodes: $SLURM_NODELIST"
echo "============================================"
echo "NCCL Configuration:"
env | grep NCCL_ | sort
echo "============================================"
echo ""

# Test large messages
srun --cpu-bind=none ./build/all_reduce_perf -b 128M -e 4G -f 2 -g 8

echo ""
echo "============================================"
echo "Job finished: $(date)"
echo "============================================"
