import streamlit as st
import gloria
from gloria.lightning.pretrain_model import PretrainModel
from gloria.datasets.mimic_for_gloria import *
from gloria.datasets.visualization_utils import *
from omegaconf import OmegaConf
import os
import pandas as pd


checkpoints = {
    'gloria_pretrained': 'pretrained/chexpert_resnet50.ckpt',
    'gloria_retrained': 'pretrained/retrained_last_epoch16.ckpt',
    'clinical_masking': 'pretrained/retrained_masked_last_epoch25.ckpt'
}


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


class OnDelete:
    def __init__(self, df, dicom_id, sent_id, ckpt_name, file):
        self.df = df
        self.dicom_id = dicom_id
        self.sent_id = sent_id
        self.ckpt_name = ckpt_name
        self.file = file
    def __call__(self):
        self.df = self.df[(self.df.dicom_id != self.dicom_id) | (self.df.sent_id != self.sent_id) | (self.df.checkpoint_name != self.ckpt_name)]
        self.df.to_csv(self.file, index=False)


class OnSubmit:
    def __init__(self, df, dicom_id, sent_id, ckpt_name, new_row, file):
        self.df = df
        self.dicom_id = dicom_id
        self.sent_id = sent_id
        self.ckpt_name = ckpt_name
        self.new_row = new_row
        self.file = file

    def __call__(self):
        self.df = self.df[(self.df.dicom_id != self.dicom_id) | (self.df.sent_id != self.sent_id) | (self.df.checkpoint_name != self.ckpt_name)]
        self.df = self.df.append(pd.DataFrame([self.new_row]))
        self.df.to_csv(self.file, index=False)


st.set_page_config(layout="wide")
with st.sidebar:
    st.title('Exploring & Annotating GLoRIA Attention')
    checkpoint_name = st.selectbox('Model', checkpoints.keys())
    checkpoint = checkpoints[checkpoint_name]
    model = load_model(checkpoint)
    config = get_config('configs/imagenome_pretrain_val_config.yaml')
    datamodule = load_data(config)
    collate_fn = get_collate_fn(config)
    split = st.selectbox('Dataset Split', ['val', 'test'])
    dataset = getattr(datamodule, split)
    dataset.sentences_df = dataset.sentences_df.drop_duplicates('dicom_id')
    data_size = len(dataset)

    if not os.path.exists('annotations'):
        os.mkdir('annotations')
    current_annotations = [x[:-4] for x in os.listdir('annotations') if x.endswith('.csv')]
    annotations_name = st.selectbox('Set of annotations to update', ['New set of annotations'] + current_annotations)
    if annotations_name == 'New set of annotations':
        annotations_name = st.text_input('Type out a name for this set of annotations')
        assert '/' not in annotations_name
    instance_number = st.number_input('Instance', min_value=1, max_value=data_size, value=1, step=1)
    st.slider('', 1, data_size, instance_number, disabled=True)
    instance = dataset[instance_number - 1]
    patient_id = next(iter(instance.keys()))
    study_id = next(iter(instance[patient_id].keys()))
    dicom_id = next(iter(instance[patient_id][study_id]['images'].keys()))
    #sent_id = instance[patient_id][study_id]['sent_id']
col1, col2 = st.columns(2)
with col1:
    with st.expander('Full Report', expanded=False):
        st.write(instance[patient_id][study_id]['report'])
    with st.expander('Prompt', expanded=True):
        file = 'annotations/' + annotations_name + '.csv'
        if os.path.exists(file):
            df = get_annotations(file)
        else:
            df = pd.DataFrame([], columns=['dicom_sent_id', 'dicom_id', 'sent_id', 'checkpoint_name', 'prompt', 'rating', 'is_custom_prompt'])
        sent_info = instance[patient_id][study_id]['objects'][dicom_id]['sent_to_bboxes']
        relevant_rows = df[(df.dicom_id == dicom_id) & (df.checkpoint_name == checkpoint_name)]
        sent_id_is_annotated = {k: len(relevant_rows[relevant_rows.sent_id == k]) > 0 for k in sent_info.keys()}
        custom_prompt = st.checkbox('Custom Prompt')
        st.write('Report Sentences')
        format_func = lambda k: sent_info[k]['sentence'] + (' (annotated)' if sent_id_is_annotated[k] else '')
        sent_id = st.radio('Sentences', list(sent_info.keys()), disabled=custom_prompt, format_func=format_func)
        sentence = sent_info[sent_id]['sentence']
        st.write('Prompt:')
        if custom_prompt:
            prompt = st.text_input('Enter text prompt here.')
        else:
            prompt = sentence
            st.write(prompt)
    with st.expander('Annotate', expanded=True):
        if annotations_name != "":
            st.write('Annotations: ' + annotations_name)
            if custom_prompt:
                current_custom_prompt_rows = df[(df.dicom_id == dicom_id) & df.is_custom_prompt]
                same_rows = current_custom_prompt_rows[current_custom_prompt_rows.prompt == prompt]
                if len(same_rows) > 0:
                    sent_id = same_rows.iloc[0].sent_id
                else:
                    current_custom_id_ints = [int(custom_id[6:]) for custom_id in current_custom_prompt_rows.sent_id]
                    index = max(current_custom_id_ints) if len(current_custom_id_ints) > 0 else -1
                    sent_id = 'custom' + str(index + 1)
            relevant_rows = df[(df.dicom_id == dicom_id) & (df.sent_id == sent_id) & (df.checkpoint_name == checkpoint_name)]
            if len(relevant_rows) > 0:
                st.write('Current rating: %i' % int(relevant_rows.iloc[0].rating))
                ondelete = OnDelete(df, dicom_id, sent_id, checkpoint_name, file)
                st.button('delete', on_click=ondelete)
            rating = st.slider('Rating of the usefulness of the attention', 1, 5, 3, key='rating %s %s %s' % (dicom_id, sent_id, checkpoint_name))
            new_row = {'dicom_sent_id': 'dicom_%s_sent_%s' % (dicom_id, sent_id), 'dicom_id': dicom_id, 'sent_id': sent_id,
                       'checkpoint_name': checkpoint_name, 'prompt': prompt, 'rating': rating, 'is_custom_prompt': custom_prompt}
            onsubmit = OnSubmit(df, dicom_id, sent_id, checkpoint_name, new_row, file)
            st.button('submit', on_click=onsubmit)
    if annotations_name != "":
        with st.expander('All Annotations', expanded=False):
            st.write(df)
with col2:
    original_image = instance[patient_id][study_id]['images'][dicom_id]
    image = collate_fn.process_img([original_tensor_to_numpy_image(original_image)], 'cpu')[0, 0]
    @st.cache(allow_output_mutation=True)
    def get_attention(image_id, prmpt, ckpt_name):
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
    attn_img = get_attention(dicom_id, prompt, checkpoint_name)
    attn_strength = st.slider('Display attention coefficient', 0, 10000, value=5000, step=100)
    st.image(
        original_tensor_to_numpy_image(image + attn_strength * attn_img),
        use_column_width='always')
print("done")
