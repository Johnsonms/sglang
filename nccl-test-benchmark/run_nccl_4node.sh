#!/bin/bash
#SBATCH -N 4
#SBATCH --nodelist=gpu-dp-x2vff-zrg5j,gpu-dp-x2vff-qfz7z,gpu-dp-x2vff-xjx42,gpu-dp-x2vff-7gdx4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --job-name=nccl_4n
#SBATCH --time=00:10:00
#SBATCH --output=nccl_4node_%j.out

export NCCL_DEBUG=WARN

echo "============================================"
echo "4-Node NCCL Test"
echo "Job started: $(date)"
echo "Nodes: $SLURM_NODELIST"
echo "============================================"

srun --cpu-bind=none ./build/all_reduce_perf -b 64M -e 128M -f 2 -g 8

echo "============================================"
echo "Job finished: $(date)"
echo "============================================"
