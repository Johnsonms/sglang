#!/bin/bash
#SBATCH -N 4
#SBATCH --nodelist=gpu-dp-x2vff-zrg5j,gpu-dp-x2vff-qfz7z,gpu-dp-x2vff-xjx42,gpu-dp-x2vff-7gdx4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --job-name=nccl_minimal
#SBATCH --time=00:30:00
#SBATCH --output=nccl_minimal_%j.out

# Minimal optimizations - let NCCL auto-tune most settings
export NCCL_IB_DISABLE=0              # Enable InfiniBand
export NCCL_NET_GDR_LEVEL=5           # Enable GPUDirect RDMA
export NCCL_CROSS_NIC=2               # Enable aggressive cross-NIC (2 for H100)
export NCCL_DEBUG=WARN                # Less verbose

echo "Job started: $(date)"
echo "Nodes: $SLURM_NODELIST"

# Test at larger sizes where performance is best
srun --cpu-bind=none ./build/all_reduce_perf -b 64M -e 2G -f 2 -g 8

echo "Job finished: $(date)"
