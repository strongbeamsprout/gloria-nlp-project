import streamlit as st
import gloria
from gloria.lightning.pretrain_model import PretrainModel
from gloria.datasets.mimic_for_gloria import *
from gloria.datasets.visualization_utils import *
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


class OnSubmit:
    def __init__(self, df, dicom_id, sent_id, new_row, file):
        self.df = df
        self.dicom_id = dicom_id
        self.sent_id = sent_id
        self.new_row = new_row
        self.file = file

    def __call__(self):
        self.df = self.df[(self.df.dicom_id != self.dicom_id) | (self.df.sent_id != self.sent_id)]
        self.df = self.df.append(pd.DataFrame([self.new_row]))
        self.df.to_csv(self.file, index=False)


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
instance = dataset[instance_number-1]
patient_id = next(iter(instance.keys()))
study_id = next(iter(instance[patient_id].keys()))
dicom_id = next(iter(instance[patient_id][study_id]['images'].keys()))
#instance_batch = collate_fn([instance])
#import pdb; pdb.set_trace()
show_caption = st.checkbox('Show caption')
original_image = instance[patient_id][study_id]['images'][dicom_id]
image = collate_fn.process_img([original_tensor_to_numpy_image(original_image)], 'cpu')[0, 0]
custom_prompt = st.checkbox('Custom Prompt')
st.write('Prompt:')
if custom_prompt:
    prompt = st.text_input('Enter text prompt here.')
else:
    prompt = instance[patient_id][study_id]['sentence']
    st.write(prompt)
@st.cache(allow_output_mutation=True)
def get_attention(image_id, prmpt):
    if len(prmpt) == 0:
        return torch.zeros_like(image)
    batch = collate_fn.get_batch(
        [original_tensor_to_numpy_image(original_image)],
        [prmpt],
    )
    img_emb_l, img_emb_g, text_emb_l, text_emb_g, sents = model.gloria(batch)
    ams = model.gloria.get_attn_maps(img_emb_l, text_emb_l, sents)
    am = ams[0][0].mean(0).detach().cpu().numpy()
    attn_img = pyramid_attn_overlay(am, (224, 224))
    return attn_img
attn_img = get_attention(dicom_id, prompt)
st.image(
    original_tensor_to_numpy_image(image + 1000 * attn_img),
    caption=instance[patient_id][study_id]['report'] if show_caption else None)
if not os.path.exists('annotations'):
    os.mkdir('annotations')
current_annotations = [x[:-4] for x in os.listdir('annotations') if x.endswith('.csv')]
annotations_name = st.selectbox('Set of annotations to update', ['New set of annotations'] + current_annotations)
if annotations_name == 'New set of annotations':
    annotations_name = st.text_input('Type out a name for this set of annotations')
    assert '/' not in annotations_name
#annotations_name = 'anns1'
if annotations_name != "":
    st.write('Annotations: ' + annotations_name)
    file = 'annotations/' + annotations_name + '.csv'
    if os.path.exists(file):
        df = get_annotations(file)
    else:
        df = pd.DataFrame([], columns=['dicom_id', 'sent_id', 'prompt', 'rating', 'is_custom_prompt'])
    st.write(df)
    if not custom_prompt:
        sent_id = instance[patient_id][study_id]['sent_id']
    else:
        current_custom_prompt_rows = df[(df.dicom_id == dicom_id) & df.is_custom_prompt]
        same_rows = current_custom_prompt_rows[current_custom_prompt_rows.prompt == prompt]
        if len(same_rows) > 0:
            sent_id = same_rows.iloc[0].sent_id
        else:
            current_custom_id_ints = [int(custom_id[6:]) for custom_id in current_custom_prompt_rows.sent_id]
            index = max(current_custom_id_ints) if len(current_custom_id_ints) > 0 else -1
            sent_id = 'custom' + str(index + 1)
    relevant_rows = df[(df.dicom_id == dicom_id) & (df.sent_id == sent_id)]
    if len(relevant_rows) > 0:
        st.write('Current rating: %i' % int(relevant_rows.iloc[0].rating))
    rating = st.slider('Rating of the usefulness of the attention', 1, 5, 3, key='rating %s %s' % (dicom_id, sent_id))
    new_row = {'dicom_id': dicom_id, 'sent_id': sent_id, 'prompt': prompt, 'rating': rating, 'is_custom_prompt': custom_prompt, 'checkpoint_name': checkpoint_name}
    onsubmit = OnSubmit(df, dicom_id, sent_id, new_row, file)
    st.button('submit', on_click=onsubmit)
    show_anns = st.checkbox('Show annotations')
    if show_anns:
        st.write(df)
print("done")
