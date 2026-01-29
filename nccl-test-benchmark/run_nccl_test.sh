#!/bin/bash
#SBATCH -N 4                          # Number of nodes
#SBATCH --nodelist=gpu-dp-x2vff-zrg5j,gpu-dp-x2vff-qfz7z,gpu-dp-x2vff-xjx42,gpu-dp-x2vff-7gdx4
#SBATCH --ntasks-per-node=1           # 1 task per node (each uses 8 GPUs)
#SBATCH --gpus-per-node=8             # Request 8 GPUs per node
#SBATCH --job-name=nccl_test
#SBATCH --time=00:30:00               # Time limit
#SBATCH --output=nccl_test_%j.out     # Output file

# NCCL debug settings
export NCCL_DEBUG=INFO

# Print job info
echo "Job started at: $(date)"
echo "Nodes: $SLURM_NODELIST"
echo "Total tasks: $SLURM_NTASKS"
echo ""

# Run NCCL test - each task will use 8 GPUs (-g 8)
# Disable CPU binding since we have limited CPUs but many GPUs
srun --cpu-bind=none ./build/all_reduce_perf -b 64M -e 128M -f 2 -g 8

echo ""
echo "Job finished at: $(date)"
