#!/bin/bash
#SBATCH --partition=frink
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --export=1
#SBATCH --mem=50Gb
#SBATCH --cpus-per-task=12
#SBATCH --time=36:00:00
#SBATCH --job-name=gloria_train
module load anaconda3/3.7
module load cuda/10.2
module load discovery/2019-02-21
source activate gloria
python run.py -c configs/imagenome_pretrain_config.yaml --train --random_seed 1
#python run.py -c configs/imagenome_pretrain_config.yaml --train --mask_mode clinical --mask_prob 0.5 --random_seed 1
#python run.py -c configs/imagenome_pretrain_config.yaml --train --mask_mode clinical --mask_prob 0.5
#python run.py -c configs/imagenome_pretrain_config.yaml --train --mask_mode word --mask_prob 0.3
#python run.py -c configs/imagenome_pretrain_config.yaml --train
#python test_data.py
