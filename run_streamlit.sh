module load anaconda3/3.7
module load cuda/10.2
module load discovery/2019-02-21
source activate gloria
ssh login-00 -f -N -T -R 8501:localhost:8501
streamlit run interface.py
#python -m pdb interface.py
