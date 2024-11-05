#!/bin/bash

# Invoke predict.py on data preprocessed by prepare.sh

#SBATCH --job-name=hplt-registers
#SBATCH --nodes=1
#SBATCH --gres=gpu:mi250:8
#SBATCH --ntasks=8
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=slurm-logs/%j.out
#SBATCH --error=slurm-logs/%j.err
#SBATCH --account=project_462000353
#SBATCH --partition=small-g

# If run without sbatch, invoke here
if [ -z $SLURM_JOB_ID ]; then
    sbatch "$0" "$@"
    exit
fi

set -euo pipefail

source common_lumi.sh

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 PACKAGE" >&2
    echo >&2
    echo "example: $0 cleaned/eng_Latn/1" >&2
    exit 1
fi

PACKAGE="$1"
get_lock "$PACKAGE"

LOG_PATH="$LOG_BASE_DIR/${PACKAGE%%.*}.txt"
mkdir -p $(dirname "$LOG_PATH")

module use /appl/local/csc/modulefiles
module load pytorch/2.4

rocm-smi

echo "$(date): START RUNNING predict.sh" >> "$LOG_PATH"

cat <<EOF

------------------------------------------------------------------------------
Predict labels
------------------------------------------------------------------------------
EOF

SPLIT_DIR="$SPLIT_BASE_DIR/$PACKAGE"

PREDICT_DIR="$PREDICT_BASE_DIR/$PACKAGE"
mkdir -p "$PREDICT_DIR"

for i in `seq 0 $((SPLIT_PARTS-1))`; do
    srun \
	--ntasks=1 \
	--gres=gpu:mi250:1 \
    --mem=16G \ 
	python3 predict.py \
	"$SPLIT_DIR/0$i.jsonl" \
	"$PREDICT_DIR/0$i.jsonl" \
	&
done
    
wait

cat <<EOF

------------------------------------------------------------------------------
Predictions DONE, files in $PREDICT_DIR:
------------------------------------------------------------------------------
EOF
find "$PREDICT_DIR" -name '*.jsonl'

echo "$(date): END RUNNING predict.sh" >> "$LOG_PATH"