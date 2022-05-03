#!/bin/bash
#SBATCH --partition=frink
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --export=1
#SBATCH --mem=50Gb
#SBATCH --cpus-per-task=12
#SBATCH --time=48:00:00
#SBATCH --job-name=gloria_train
module load anaconda3/3.7
module load cuda/10.2
module load discovery/2019-02-21
source activate gloria
python run.py -c configs/imagenome_pretrain_config.yaml --train --no_attn_vec --attention_entropy_loss_weight 1.0 --attention_divergence_loss_weight 1.0 --mask_mode word --mask_prob .3 --random_seed=0


#python test_data.py


# random seed 0
# baseline
#python run.py -c configs/imagenome_pretrain_config.yaml --train
#python run.py -c configs/imagenome_pretrain_config.yaml --train --ckpt_path=/scratch/mcinerney.de/gloria_outputs7/ckpt/gloria_pretrain_1.0/2022_04_18_02_22_57/last.ckpt

# clinical entity masking
#python run.py -c configs/imagenome_pretrain_config.yaml --train --mask_mode clinical --mask_prob 0.5 --random_seed=0
#python run.py -c configs/imagenome_pretrain_config.yaml --train --mask_mode clinical --mask_prob 0.5 --random_seed=0 \
#    --ckpt_path=/scratch/mcinerney.de/gloria_outputs7/ckpt/gloria_pretrain_1.0/2022_04_18_02_23_47/last.ckpt
#python run.py -c configs/imagenome_pretrain_config.yaml --train --mask_mode clinical --mask_prob 0.5 --random_seed=0 \
#    --ckpt_path=/scratch/mcinerney.de/gloria_outputs7/ckpt/gloria_pretrain_1.0/2022_04_22_01_32_55/last.ckpt

# word masking
#python run.py -c configs/imagenome_pretrain_config.yaml --train --mask_mode word --mask_prob 0.3 --random_seed=0
#python run.py -c configs/imagenome_pretrain_config.yaml --train --mask_mode word --mask_prob 0.3 --random_seed=0 \
#    --ckpt_path=/scratch/mcinerney.de/gloria_outputs7/ckpt/gloria_pretrain_1.0/2022_04_18_02_23_20/last.ckpt
#python run.py -c configs/imagenome_pretrain_config.yaml --train --mask_mode word --mask_prob 0.3 --random_seed=0 \
#    --ckpt_path=/scratch/mcinerney.de/gloria_outputs7/ckpt/gloria_pretrain_1.0/2022_04_19_15_45_33/last.ckpt

# no global loss
#python run.py -c configs/imagenome_pretrain_config.yaml --train --global_loss_weight 0 --random_seed=0

# entropy
#python run.py -c configs/imagenome_pretrain_config.yaml --train --attention_entropy_loss_weight 1.0 --random_seed=0

# no attn vec w/ loss
#python run.py -c configs/imagenome_pretrain_config.yaml --train --no_attn_vec --no_attn_loss_weight 1.0 --random_seed=0

# no attn vec w/ loss and entropy
#python run.py -c configs/imagenome_pretrain_config.yaml --train --no_attn_vec --no_attn_loss_weight 1.0 --attention_entropy_loss_weight 1.0 --random_seed=0

# no attn vec and entropy
#python run.py -c configs/imagenome_pretrain_config.yaml --train --no_attn_vec --attention_entropy_loss_weight 1.0 --random_seed=0

# kl divergence
#python run.py -c configs/imagenome_pretrain_config.yaml --train --attention_divergence_loss_weight 1.0 --random_seed=0


# random seed 23
# baseline
#python run.py -c configs/imagenome_pretrain_config.yaml --train --random_seed=23
#python run.py -c configs/imagenome_pretrain_config.yaml --train --random_seed=23 \
#    --ckpt_path=/scratch/mcinerney.de/gloria_outputs7/ckpt/gloria_pretrain_1.0/2022_04_23_13_33_29/last.ckpt

# clinical entity masking
#python run.py -c configs/imagenome_pretrain_config.yaml --train --mask_mode clinical --mask_prob 0.5 --random_seed=23
#python run.py -c configs/imagenome_pretrain_config.yaml --train --mask_mode clinical --mask_prob 0.5 --random_seed=23 \
#    --ckpt_path=/scratch/mcinerney.de/gloria_outputs7/ckpt/gloria_pretrain_1.0/2022_04_23_13_33_52/last.ckpt

# word masking
#python run.py -c configs/imagenome_pretrain_config.yaml --train --mask_mode word --mask_prob 0.3 --random_seed=23
#python run.py -c configs/imagenome_pretrain_config.yaml --train --mask_mode word --mask_prob 0.3 --random_seed=23 \
#    --ckpt_path=/scratch/mcinerney.de/gloria_outputs7/ckpt/gloria_pretrain_1.0/2022_04_25_17_19_40/last.ckpt
