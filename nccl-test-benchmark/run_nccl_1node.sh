#!/bin/bash
#SBATCH -N 1
#SBATCH --nodelist=gpu-dp-x2vff-zrg5j
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --job-name=nccl_1n
#SBATCH --time=00:10:00
#SBATCH --output=nccl_1node_%j.out

export NCCL_DEBUG=WARN

echo "============================================"
echo "1-Node NCCL Test"
echo "Job started: $(date)"
echo "Nodes: $SLURM_NODELIST"
echo "============================================"

srun --cpu-bind=none ./build/all_reduce_perf -b 64M -e 128M -f 2 -g 8

echo "============================================"
echo "Job finished: $(date)"
echo "============================================"
