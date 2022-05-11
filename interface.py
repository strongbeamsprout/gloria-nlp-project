import streamlit as st
import gloria
from gloria.lightning.pretrain_model import PretrainModel
from gloria.datasets.mimic_for_gloria import *
from gloria.datasets.visualization_utils import *
from omegaconf import OmegaConf
import os
import pandas as pd
import tokenizers
from gloria.datasets.mimic_data import RowLabelAndContextSelector
import copy


def process_bboxes(image_shapes, bboxes, collatefn):
    new_bboxes = []
    box_masks = []
    for shape, bbox in zip(image_shapes, bboxes):
        box_mask = bbox_to_mask(bbox, shape)
        box_masks.append(original_tensor_to_numpy_image(box_mask))
    new_box_masks = collatefn.process_img(box_masks, 'cpu')
    new_box_masks = new_box_masks > 0
    new_bboxes = [mask_to_bbox(new_box_mask[0]) for new_box_mask in new_box_masks]
    return new_bboxes


checkpoints = {
    'pretrained': 'pretrained/pretrained.ckpt',
    'baseline': 'pretrained/baseline_2022_04_20_09_25_42_epoch14.ckpt',
#    'noattn_entropy': 'pretrained/noattn_entropy_2022_05_06_00_57_04_epoch18.ckpt',
#    'noattn_noattnloss_entropy': 'pretrained/noattn_noattnloss_entropy_2022_05_06_00_58_48_epoch17.ckpt',
#    'noattn_entropy_kl': 'pretrained/noattn_entropy_kl_2022_05_06_11_34_25_epoch13.ckpt',
    'wordmask': 'pretrained/wordmask_2022_04_19_15_45_33_epoch17.ckpt',
    'clinicalmask': 'pretrained/clinicalmask_2022_04_24_10_17_44_epoch16.ckpt',
#    'pretrained_finetuned': 'pretrained/pretrained_segfinetune30_2022_05_10_02_07_19_last.ckpt',
#    'noattn_entropy_kl.1': '/scratch/mcinerney.de/gloria_outputs7/ckpt/gloria_pretrain_1.0/2022_05_09_00_03_47/last.ckpt',
#    'onlylocal': '/scratch/mcinerney.de/gloria_outputs7/ckpt/gloria_pretrain_1.0/2022_05_09_00_04_46/last.ckpt',

#    'no_attn': '/scratch/mcinerney.de/gloria_outputs7/ckpt/gloria_pretrain_1.0/2022_04_29_10_46_53/epoch=0-step=3650.ckpt',
#    'no_attn_low_entropy': '/scratch/mcinerney.de/gloria_outputs7/ckpt/gloria_pretrain_1.0/2022_04_27_13_33_00/last.ckpt',
#    'gloria_pretrained': 'pretrained/chexpert_resnet50.ckpt',
#    'gloria_retrained': 'pretrained/retrained_last_epoch16.ckpt',
#    'clinical_masking': 'pretrained/retrained_masked_last_epoch25.ckpt'
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


anonymize_models = True
st.set_page_config(layout="wide")
with st.sidebar:
    st.title('Exploring & Annotating GLoRIA Attention')

    # get data
    config = get_config('configs/imagenome_pretrain_val_config.yaml')
    datamodule = load_data(config)
    collate_fn = get_collate_fn(config)
    split = st.selectbox('Dataset Split', ['valid', 'gold'])
    dataset = datamodule.get_dataset(split)
    subset = st.selectbox('Subset', ['all', 'abnormal', 'one_lung'])
    @st.cache(allow_output_mutation=True, hash_funcs={pd.DataFrame: lambda x: 0})
    def get_sentences(split, subset, sentences_df):
        if subset == 'abnormal':
            selector = RowLabelAndContextSelector(contains={('abnormal', 'yes')})
            sentences_df = sentences_df[sentences_df.apply(selector, axis=1)]
        elif subset == 'one_lung':
            selector1 = RowBBoxSelector(contains={'right lung'}, does_not_contain={'left lung'})
            selector2 = RowBBoxSelector(contains={'left lung'}, does_not_contain={'right lung'})
            selector = lambda r: selector1(r) or selector2(r)
            sentences_df = sentences_df[sentences_df.apply(selector, axis=1)]
        sentences_df = sentences_df.drop_duplicates('dicom_id')        
        return sentences_df
    dataset.sentences_df = get_sentences(split, subset, dataset.sentences_df)
    data_size = len(dataset)

    # get instance
    instance_number = st.number_input('Instance', min_value=1, max_value=data_size, value=1, step=1)
    st.slider('', 1, data_size, instance_number, disabled=True)
    instance = dataset[instance_number - 1]
    patient_id = next(iter(instance.keys()))
    study_id = next(iter(instance[patient_id].keys()))
    dicom_id = next(iter(instance[patient_id][study_id]['images'].keys()))
    #sent_id = instance[patient_id][study_id]['sent_id']

    # get model
    model_names = sorted(list(checkpoints.keys()))
    if anonymize_models:
        @st.cache(allow_output_mutation=True)
        def get_model_aliases(dicomid):
            aliases = ['model_%i' % i for i in range(len(checkpoints))]
            random.shuffle(aliases)
            return aliases
        aliases = get_model_aliases(dicom_id)
    else:
        aliases = model_names
    alias_to_model = {alias: model for alias, model in zip(aliases, model_names)}
    alias = st.selectbox('Model', sorted(list(aliases)))
    checkpoint_name = alias_to_model[alias]
    checkpoint = checkpoints[checkpoint_name]
    model = load_model(checkpoint)
    has_no_attn = model.hparams.model.gloria.no_attn_vec

    # get annotations
    if not os.path.exists('annotations'):
        os.mkdir('annotations')
    current_annotations = [x[:-4] for x in os.listdir('annotations') if x.endswith('.csv')]
    annotations_name = st.selectbox('Set of annotations to update', ['New set of annotations'] + current_annotations)
    if annotations_name == 'New set of annotations':
        annotations_name = st.text_input('Type out a name for this set of annotations')
        assert '/' not in annotations_name
with st.expander('Annotation Instructions', expanded=True):
    with open('annotation_instructions.txt', 'r') as f:
        raw_sections = f.read().split('\n-\n')
        sections = {}
        for sec in raw_sections:
            splitsec = sec.split('\n\n')
            key = splitsec[0].strip()
            text = '\n\n'.join(splitsec[1:]).strip()
            sections[key] = text
    right_broad_cap = sections['right_broad_cap']
    del sections['right_broad_cap']
    wrong_precise_cap = sections['wrong_precise_cap']
    del sections['wrong_precise_cap']
    for key, text in sections.items():
        st.markdown(text)
    col1, col2 = st.columns(2)
    with col1:
        st.write(right_broad_cap)
    with col2:
        st.write(wrong_precise_cap)
with st.expander('Full Report', expanded=False):
    st.write(instance[patient_id][study_id]['report'])
col1, col2 = st.columns(2)
with col1:
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
        format_func = lambda k: sent_info[k]['sentence'] + (' (annotated)' if sent_id_is_annotated[k] else '')
        sent_id = st.radio('Report Sentences', list(sent_info.keys()), disabled=custom_prompt, format_func=format_func)
        sentence = sent_info[sent_id]['sentence']
        if custom_prompt:
            prompt = st.text_area('Enter text prompt here.')
        else:
            prompt = sentence
    st.markdown('**Prompt**: ' + prompt)
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
                st.write('Current annotation:')
                st.write('Has good recall? %s' % relevant_rows.iloc[0].has_good_recall)
                st.write('Has bad precision? %s' % relevant_rows.iloc[0].has_bad_precision)
                st.write('Is intuitive? %s' % relevant_rows.iloc[0].is_intuitive)
                ondelete = OnDelete(df, dicom_id, sent_id, checkpoint_name, file)
                st.button('delete', on_click=ondelete)
            has_good_recall = st.select_slider('Is the region of interest from the prompt in the heatmap?',
                options=['no', 'partially', 'yes'],
                key='good recall %s %s %s' % (dicom_id, sent_id, checkpoint_name))
            has_bad_precision = st.select_slider('Does the heatmap contain irrelevant regions?',
                options=['no', 'partially', 'yes'],
                key='bad precision %s %s %s' % (dicom_id, sent_id, checkpoint_name))
            is_intuitive = st.select_slider('Is this heatmap intuitive?',
                options=['no', 'partially', 'yes'],
                key='intuitive %s %s %s' % (dicom_id, sent_id, checkpoint_name))
            new_row = {'dicom_sent_id': 'dicom_%s_sent_%s' % (dicom_id, sent_id), 'dicom_id': dicom_id, 'sent_id': sent_id,
                       'checkpoint_name': checkpoint_name, 'prompt': prompt, 'has_good_recall': has_good_recall,
                       'has_bad_precision': has_bad_precision,
                       'is_intuitive': is_intuitive, 'is_custom_prompt': custom_prompt}
            onsubmit = OnSubmit(df, dicom_id, sent_id, checkpoint_name, new_row, file)
            st.button('submit', on_click=onsubmit, disabled=prompt == "")
    if annotations_name != "":
        with st.expander('All Annotations', expanded=False):
            if anonymize_models:
                df_without_checkpoint = df.copy()
                del df_without_checkpoint['checkpoint_name']
                st.write(df_without_checkpoint)
            else:
                st.write(df)
with col2:
    original_image = instance[patient_id][study_id]['images'][dicom_id]
    original_image = original_tensor_to_numpy_image(original_image)
    image = collate_fn.process_img([original_image], 'cpu')[0, 0]
    show_bboxes = st.checkbox('Show Bounding Boxes', disabled=custom_prompt)
    display_attn = st.checkbox('Display Attention')
    if display_attn:
        @st.cache(allow_output_mutation=True, hash_funcs={tokenizers.Tokenizer: lambda x: 0})
        def get_attention(image_id, prmpt, ckpt_name):
            if len(prmpt) == 0:
                return torch.zeros_like(image), 0
            batch = collate_fn.get_batch(
                [original_image],
                [prmpt],
            )
            img_emb_l, img_emb_g, text_emb_l, text_emb_g, sents = model.gloria(batch)
            ams = model.gloria.get_attn_maps(img_emb_l, text_emb_l, sents)
            am = ams[0][0].mean(0).detach().cpu().numpy()
            no_attn = 1 - am.sum()
            attn_img = pyramid_attn_overlay(am, (224, 224))
            return attn_img, no_attn
        attn_img, no_attn_score = get_attention(dicom_id, prompt, checkpoint_name)
        if has_no_attn:
            attn_img[-10:, -10:] = no_attn_score
        attn_strength = st.select_slider('Display attention coefficient', options=[0., 1e1, 3e1, 1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7, 3e7, 1e8, 3e8, 1e9, 3e9, 1e10], value=1e5)
        numpy_image = original_tensor_to_numpy_image(image + attn_strength * attn_img)
    else:
        numpy_image = original_tensor_to_numpy_image(image)
    if not custom_prompt and show_bboxes:
        def get_bboxes(image_id, sent_id):
            original_bboxes = sent_info[sent_id]['coords_original']
            new_bboxes = process_bboxes([original_image.shape] * len(original_bboxes), original_bboxes, collate_fn)
            return new_bboxes
        bboxes = get_bboxes(dicom_id, sent_id)
        numpy_image = draw_bounding_boxes(to_rgb(torch.tensor(numpy_image)), bboxes)
    st.image(
        numpy_image,
        use_column_width='always')
print("done")
