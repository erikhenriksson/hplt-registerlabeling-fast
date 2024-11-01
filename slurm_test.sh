#!/bin/bash
#SBATCH --job-name=hplt-registers          
#SBATCH --nodes=1                       
#SBATCH --gres=gpu:a100:4                 
#SBATCH --ntasks=4            
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G        
#SBATCH --cpus-per-task=16         
#SBATCH --time=00:15:00  
#SBATCH --output=logs/output_%j.lo
#SBATCH --error=logs/error_%j.log  
#SBATCH --account=project_2011770
#SBATCH --partition=gputest

module load pytorch/2.4

# Launch each execution of predict.py using one specific GPU
srun --ntasks=1 --gres=gpu:1 --exclusive python3 predict.py /scratch/project_2011770/webscale-registers/splits/deduplicated/fin_Latn/1/00.jsonl & 
srun --ntasks=1 --gres=gpu:1 --exclusive python3 predict.py /scratch/project_2011770/webscale-registers/splits/deduplicated/fin_Latn/1/01.jsonl & 
srun --ntasks=1 --gres=gpu:1 --exclusive python3 predict.py /scratch/project_2011770/webscale-registers/splits/deduplicated/fin_Latn/1/02.jsonl & 
srun --ntasks=1 --gres=gpu:1 --exclusive python3 predict.py /scratch/project_2011770/webscale-registers/splits/deduplicated/fin_Latn/1/03.jsonl & 

wait  # Wait for all background tasks to complete