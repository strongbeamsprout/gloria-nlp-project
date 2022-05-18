#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
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
CKPT=pretrained/clinicalmask_2022_04_24_10_17_44_epoch16.ckpt
python run.py -c configs/imagenome_pretrain_val_config.yaml --test --random_seed 0 \
    --ckpt_path=$CKPT
#python run.py -c configs/imagenome_pretrain_val_config.yaml --test --random_seed 0 --swap_left_right \
#    --ckpt_path=$CKPT
#python run.py -c configs/imagenome_pretrain_val_config.yaml --test --random_seed 0 --randomize_objects_mode shuffle_bboxes_sentences \
#    --ckpt_path=$CKPT
#python run.py -c configs/imagenome_pretrain_val_config.yaml --test --random_seed 0 --randomize_objects_mode random_sentences \
#    --ckpt_path=$CKPT
#python run.py -c configs/imagenome_pretrain_val_config.yaml --test --random_seed 0 --randomize_objects_mode random_bboxes \
#    --ckpt_path=$CKPT
#python run.py -c configs/imagenome_pretrain_val_config.yaml --test --random_seed 0 --generate_sent \
#    --ckpt_path=$CKPT
#python run.py -c configs/imagenome_pretrain_val_config.yaml --test --random_seed 0 --generate_sent --swap_conditions \
#    --ckpt_path=$CKPT

#CKPT=pretrained/pretrained.ckpt
#CKPT=pretrained/baseline_2022_04_20_09_25_42_epoch14.ckpt
#CKPT=pretrained/wordmask_2022_04_19_15_45_33_epoch17.ckpt
#CKPT=pretrained/clinicalmask_2022_04_24_10_17_44_epoch16.ckpt

#CKPT=pretrained/abnormal_2022_05_13_19_21_55_epoch20.ckpt
#CKPT='pretrained/noattn_2022_05_16_09_57_38_epoch14.ckpt --no_attn_vec'
#CKPT=pretrained/baseline_supervised_2022_05_17_01_34_49_epoch_last.ckpt

#CKPT=pretrained/pretrained_supervised_2022_05_16_19_55_10_epoch_last.ckpt
#CKPT=pretrained/noattn_entropy_2022_05_06_00_57_04_epoch18.ckpt
#CKPT=pretrained/noattn_entropy_kl_2022_05_06_11_34_25_epoch13.ckpt
#CKPT=pretrained/onlylocal_2022_05_10_19_27_18_epoch11.ckpt
#CKPT=pretrained/
