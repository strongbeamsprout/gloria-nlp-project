import streamlit as st
import gloria
from gloria.lightning.pretrain_model import PretrainModel
from gloria.datasets.mimic_for_gloria import *
from omegaconf import OmegaConf
import os
import pandas as pd


@st.cache(allow_output_mutation=True)
def load_model(ckpt):
    print("loading model")
    m = PretrainModel.load_from_checkpoint(ckpt)
    print("done")
    return m


@st.cache(allow_output_mutation=True)
def get_config(config_file):
    print("loading config")
    cfg = OmegaConf.load(config_file)
    print("done")
    return cfg


@st.cache(allow_output_mutation=True)
def load_data(cfg):
    print("loading data module")
    dm = gloria.builder.build_data_module(cfg)
    print("done")
    return dm.dm


@st.cache(allow_output_mutation=True)
def get_collate_fn(cfg):
    print("loading collate function")
    cf = GloriaCollateFn(cfg, "test")
    print("done")
    return cf


@st.cache(allow_output_mutation=True)
def get_annotations(ann_file):
    print("loading annotations")
    df = pd.read_csv(ann_file)
    print("done")
    return df


checkpoints = {
#    'gloria_pretrained': 'pretrained/retrained_masked_last_epoch25.ckpt',
#    'gloria_retained': 'pretrained/retrained_masked_last_epoch25.ckpt',
    'clinical_masking': 'pretrained/retrained_masked_last_epoch25.ckpt'
}


st.title('Exploring & Annotating GLoRIA Attention')
checkpoint_name = st.selectbox('Model', checkpoints.keys())
checkpoint = checkpoints[checkpoint_name]
model = load_model(checkpoint)
config = get_config('configs/imagenome_pretrain_val_config.yaml')
datamodule = load_data(config)
collate_fn = get_collate_fn(config)
split = st.selectbox('Dataset Split', ['val', 'test'])
dataset = getattr(datamodule, split)
data_size = len(dataset)
instance_number = st.slider('instance', 1, data_size, 1)
instance = dataset[instance_number]
patient_id = next(iter(instance.keys()))
study_id = next(iter(instance[patient_id].keys()))
dicom_id = next(iter(instance[patient_id][study_id]['images'].keys()))
#instance_batch = collate_fn([instance])
#import pdb; pdb.set_trace()
show_caption = st.checkbox('Show caption')
original_image = instance[patient_id][study_id]['images'][dicom_id]
image = collate_fn.process_img([original_tensor_to_numpy_image(original_image)], 'cpu')[0, 0]
st.image(
    original_tensor_to_numpy_image(image),
    caption=instance[patient_id][study_id]['sentence'] if show_caption else None)
custom_prompt = st.checkbox('Custom Prompt')
st.write('Prompt:')
if custom_prompt:
    prompt = st.text_input('Enter text prompt here.')
else:
    prompt = instance[patient_id][study_id]['sentence']
    st.write(prompt)
if not os.path.exists('annotations'):
    os.mkdir('annotations')
current_annotations = [x[:-4] for x in os.listdir('annotations') if x.endswith('.csv')]
annotations_name = st.selectbox('Set of annotations to update', ['New set of annotations'] + current_annotations)
if annotations_name == 'New set of annotations':
    annotations_name = st.text_input('Type out a name for this set of annotations')
    assert '/' not in annotations_name
if annotations_name != "":
    st.write('Annotations: ' + annotations_name)
    rating = st.slider('Rating of the usefulness of the attention', 1, 5, 1)
    file = 'annotations/' + annotations_name + '.csv'
    if os.path.exists(file):
        df = get_annotations(file)
    else:
        df = pd.DataFrame([])
    # need to implement writing to this dataframe here
    df.to_csv(file)
    print("done")
