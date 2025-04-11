#!/bin/bash
#SBATCH --job-name=hplt-registers
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=06:00:00
#SBATCH --output=logs/output_%j.log
#SBATCH --error=logs/error_%j.log
#SBATCH --account=project_2011770
#SBATCH --partition=gpumedium

# If run without sbatch, invoke here
if [ -z $SLURM_JOB_ID ]; then
    sbatch "$0" "$@"
    exit
fi

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 PACKAGE" >&2
    echo >&2
    echo "example: $0 cleaned/eng_Latn/1" >&2
    exit 1
fi

module load pytorch/2.4

# Launch each execution of predict.py using one specific GPU
srun --ntasks=1 --gres=gpu:mi250:1 python3 predict.py /scratch/project_2011770/webscale-registers/splits/deduplicated/fin_Latn/1/00.jsonl &
srun python3 predict.py /scratch/project_2011770/webscale-registers/splits/deduplicated/fin_Latn/1/01.jsonl &
srun python3 predict.py /scratch/project_2011770/webscale-registers/splits/deduplicated/fin_Latn/1/02.jsonl &
srun python3 predict.py /scratch/project_2011770/webscale-registers/splits/deduplicated/fin_Latn/1/03.jsonl &

wait  # Wait for all background tasks to complete