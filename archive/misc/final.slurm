#!/bin/bash
#SBATCH --job-name="CUDAConvFinal"
#SBATCH --output="final_job.out"
#SBATCH --error="final_job.err"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint="projects"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bche-delta-gpu
#SBATCH -t 00:25:00

module reset
module load cuda

echo -e "job $SLURM_JOBID is starting on `hostname`\n\n"

# check mem leak
# srun compute-sanitizer ./final 100

srun ./final 10000 > final.out

# Profiling examples
# srun nsys profile --stats=true ./final > profile.out
# ncu --set full -f -o analysis_file ./final
