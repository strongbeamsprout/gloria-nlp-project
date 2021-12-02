module load anaconda3/3.7
module load cuda/10.2
module load discovery/2019-02-21
source activate gloria_new
python run.py -c configs/imagenome_pretrain_val_config.yaml --val
