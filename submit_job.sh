#!/bin/bash
#SBATCH --partition=frink
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --export=1
#SBATCH --mem=40Gb
#SBATCH --cpus-per-task=12
#SBATCH --time=96:00:00
#SBATCH --job-name=train_gloria
module load anaconda3/3.7
module load cuda/10.2
module load discovery/2019-02-21
source activate gloria
python run.py -c configs/imagenome_pretrain_config.yaml --train
