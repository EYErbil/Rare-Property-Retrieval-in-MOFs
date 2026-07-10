#!/bin/bash
#SBATCH --job-name=soap_compare
#SBATCH --output=logs/soap_compare_%j.out
#SBATCH --error=logs/soap_compare_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --partition=mid
#SBATCH --time=24:00:00

set -euo pipefail

module purge
module load cuda/12.3 cudnn/8.9.5/cuda-12.x python/3.9.5
source /path/to/your/venv/bin/activate   # edit to your Python environment



# Run from REPO_ROOT (the folder containing scripts/). Edit paths to match your system.
python scripts/soap_analysis/compare_generated_vs_qmof.py \
    --qmof-cache  soap_analysis/qmof_soap_descriptors.npz \
    --generated-cif-dir  generated_cifs/small_30A_200atom \
    --output_dir  soap_analysis/generated_vs_qmof