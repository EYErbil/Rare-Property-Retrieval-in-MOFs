#!/bin/bash
#SBATCH --job-name=vasp_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --time=180:00:00
#SBATCH --output=vasp-%J.log


# 1. Clean environment
module purge
module load vasp/6.5.1

# 2. Force MPI to ignore GPU-aware features
export OMPI_MCA_opal_cuda_support=0
export MPICH_GPU_SUPPORT_ENABLED=0


mpirun -np 32 vasp_std

# =================== Notes ===================
# - MPI-based job; adjust total cores in mpirun.
# - GPU usage optional but accelerates some calculations.
# - Memory per node depends on system size.
