#!/bin/bash
#SBATCH --partition=frink
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --export=1
#SBATCH --mem=50Gb
#SBATCH --cpus-per-task=12
#SBATCH --time=8:00:00
#SBATCH --job-name=gloria_train
module load anaconda3/3.7
module load cuda/10.2
module load discovery/2019-02-21
source activate gloria
python run.py -c configs/imagenome_attn_finetune_config.yaml --train --random_seed=0 \
    --ckpt_path=pretrained/baseline_2022_04_20_09_25_42_epoch14.ckpt
