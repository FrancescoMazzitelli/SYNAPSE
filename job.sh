#!/bin/bash
#SBATCH --job-name=synth_survey
#SBATCH --partition=batch_gpu
#SBATCH --output=run/output_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --gres=gpu:1
#SBATCH --mem=240G
#SBATCH --time=03:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=isalva2@uic.edu


module load Python/3.11.3-GCCcore-12.3.0
source ~/venv/bin/activate

module load ollama
ollama serve > /dev/null 2>&1 &

python main.py configs/Chicago