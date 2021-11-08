module load anaconda3/3.7
module load cuda/10.2
module load discovery/2019-02-21
source activate gloria
python -m pdb run.py -c configs/imagenome_pretrain_config.yaml
