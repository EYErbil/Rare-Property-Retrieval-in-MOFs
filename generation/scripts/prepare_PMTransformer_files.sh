#!/bin/bash
#SBATCH --job-name=MOFT-prepare
#SBATCH --output=logs/MT_%j.out
#SBATCH --error=logs/MT_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --partition=mid
#SBATCH --time=24:00:00

set -euo pipefail

module purge
module load cuda/12.3 cudnn/8.9.5/cuda-12.x python/3.9.5
source /path/to/your/venv/bin/activate   # edit to your Python environment



python prepare_moftransformer_test_only.py \
  --cif-dir generated_cifs/small_30A_200atom \
  --output-dataset-dir generated_cifs/PMtransformer_Files \
  --downstream bandgaps \
  --default-target-value 0.0 \
  --overwrite-raw-json