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
from PIL import Image


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
    'wordmask': 'pretrained/wordmask_2022_04_19_15_45_33_epoch17.ckpt',
    'clinicalmask': 'pretrained/clinicalmask_2022_04_24_10_17_44_epoch16.ckpt',
    'abnormal': 'pretrained/abnormal_2022_05_13_19_21_55_epoch20.ckpt',
    'noattn': 'pretrained/noattn_2022_05_16_09_57_38_epoch14.ckpt',
    'baseline_supervised': 'pretrained/baseline_supervised_2022_05_17_01_34_49_epoch_last.ckpt',

#    'pretrained_supervised': 'pretrained/pretrained_supervised_2022_05_16_19_55_10_epoch_last.ckpt',
#    'noattn_entropy': 'pretrained/noattn_entropy_2022_05_06_00_57_04_epoch18.ckpt',
#    'noattn_noattnloss_entropy': 'pretrained/noattn_noattnloss_entropy_2022_05_06_00_58_48_epoch17.ckpt',
#    'noattn_entropy_kl': 'pretrained/noattn_entropy_kl_2022_05_06_11_34_25_epoch13.ckpt',
#    'onlylocal': 'pretrained/onlylocal_2022_05_10_19_27_18_epoch11.ckpt',

#    'noattn_entropy_kl.1': '/scratch/mcinerney.de/gloria_outputs7/ckpt/gloria_pretrain_1.0/2022_05_16_10_01_01/last.ckpt',
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
    yes_yes_yes = sections['yes_yes_yes']
    del sections['yes_yes_yes']
    no_no_yes = sections['no_no_yes']
    del sections['no_no_yes']
    partially_partially_partially = sections['partially_partially_partially']
    del sections['partially_partially_partially']
    no_partially_partially = sections['no_partially_partially']
    del sections['no_partially_partially']
    for key, text in sections.items():
        st.markdown(text)
with st.expander('Examples', expanded=True):
    instructions_col1, instructions_col2 = st.columns(2)
    with instructions_col1:
        st.image(Image.open('yes_yes_yes.png'))
        st.write(yes_yes_yes)
    with instructions_col2:
        st.image(Image.open('no_no_yes.png'))
        st.write(no_no_yes)
    instructions_col1, instructions_col2 = st.columns(2)
    with instructions_col1:
        st.image(Image.open('partially_partially_partially.png'))
        st.write(partially_partially_partially)
    with instructions_col2:
        st.image(Image.open('no_partially_partially.png'))
        st.write(no_partially_partially)
with st.expander('Full Report', expanded=False):
    st.write(instance[patient_id][study_id]['report'])
file = 'annotations/' + annotations_name + '.csv'
if os.path.exists(file):
    df = get_annotations(file)
else:
    df = pd.DataFrame([], columns=['dicom_sent_id', 'dicom_id', 'sent_id', 'checkpoint_name', 'prompt',
                                   'has_good_recall', 'has_good_precision', 'is_intuitive', 'is_custom_prompt',
                                   'no_attn_score'])
if annotations_name != "":
    with st.expander('Current Annotations', expanded=False):
        current_annotations = []
        rows = df[df.dicom_id == dicom_id]
        prompts = set(rows.prompt)
        for prompt in prompts:
            current_annotations.append({
                'prompt': prompt
            })
            for alias in sorted(list(aliases)):
                m = alias_to_model[alias]
                rs = rows[(rows.checkpoint_name == m) & (rows.prompt == prompt)]
                if len(rs) == 1:
                    current_annotations[-1][alias] = ', '.join(
                        [str(rs.iloc[0].has_good_recall),
                         str(rs.iloc[0].has_good_precision),
                         str(rs.iloc[0].is_intuitive)])
                else:
                    current_annotations[-1][alias] = ''
        st.write(pd.DataFrame(current_annotations))
with st.expander('Prompt', expanded=True):
    sent_info = instance[patient_id][study_id]['objects'][dicom_id]['sent_to_bboxes']
    relevant_rows = df[(df.dicom_id == dicom_id) & (df.checkpoint_name == checkpoint_name)]
    sent_id_is_annotated = {k: len(relevant_rows[relevant_rows.sent_id == k]) > 0 for k in sent_info.keys()}
    custom_prompt = st.checkbox('Custom Prompt')
#    format_func = lambda k: sent_info[k]['sentence'] + (' (annotated)' if sent_id_is_annotated[k] else '')
    format_func = lambda k: sent_info[k]['sentence']
    sent_id = st.radio('Report Sentences', list(sent_info.keys()), disabled=custom_prompt, format_func=format_func,
                       key='report sentences %s' % dicom_id)
    sentence = sent_info[sent_id]['sentence']
    if custom_prompt:
        prompt = st.text_area('Enter text prompt here.')
    else:
        prompt = sentence
#annotations_container = st.container()
#image_container, attn_container = st.columns(2)
annotations_container, image_container, attn_container = st.columns([2, 1, 1])
with annotations_container:
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
                st.write('Has good recall? %s' % str(relevant_rows.iloc[0].has_good_recall))
                st.write('Has good precision? %s' % str(relevant_rows.iloc[0].has_good_precision))
                st.write('Is intuitive? %s' % str(relevant_rows.iloc[0].is_intuitive))
                ondelete = OnDelete(df, dicom_id, sent_id, checkpoint_name, file)
                st.button('delete', on_click=ondelete)
            choices = {1: '0-20', 2: '20-40', 3: '40-60', 4: '60-80', 5: '80-100'}
            has_good_recall = st.radio('The heatmap includes what percentage of the region of interest from the prompt?',
#                options=['no', 'partially', 'yes'],
                options=[1, 2, 3, 4, 5],
                format_func=lambda x: choices[x],
                key='good recall %s %s %s' % (dicom_id, sent_id, checkpoint_name))
            has_good_precision = st.radio('What percentage of the heatmap represents an area of interest?',
#                options=['no', 'partially', 'yes'],
                options=[1, 2, 3, 4, 5],
                format_func=lambda x: choices[x],
                key='bad precision %s %s %s' % (dicom_id, sent_id, checkpoint_name))
            is_intuitive = st.radio('Rate how intuitive the heatmap is on a scale from 1-5 (1 being the worst, 5 being the best).',
#                options=['no', 'partially', 'yes'],
                options=[1, 2, 3, 4, 5],
                key='intuitive %s %s %s' % (dicom_id, sent_id, checkpoint_name))
            new_row = {'dicom_sent_id': 'dicom_%s_sent_%s' % (dicom_id, sent_id), 'dicom_id': dicom_id, 'sent_id': sent_id,
                       'checkpoint_name': checkpoint_name, 'prompt': prompt, 'has_good_recall': has_good_recall,
                       'has_good_precision': has_good_precision,
                       'is_intuitive': is_intuitive, 'is_custom_prompt': custom_prompt}
            if not has_no_attn:
                onsubmit = OnSubmit(df, dicom_id, sent_id, checkpoint_name, new_row, file)
                st.button('submit', on_click=onsubmit, disabled=prompt == "")
            else:
                submit_button = st.empty()
with image_container:
    image_placeholder = st.empty()
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
            new_row['no_attn_score'] = no_attn_score
            onsubmit = OnSubmit(df, dicom_id, sent_id, checkpoint_name, new_row, file)
            submit_button.button('submit', on_click=onsubmit, disabled=prompt == "")
        #attn_strength = st.select_slider('Display attention coefficient', options=[0., 1e1, 3e1, 1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7, 3e7, 1e8, 3e8, 1e9, 3e9, 1e10], value=1e5)
        #numpy_image = original_tensor_to_numpy_image(image + attn_strength * attn_img)
        numpy_image = original_tensor_to_numpy_image(image)
        attn_numpy_image = original_tensor_to_numpy_image(attn_img)
    else:
        numpy_image = original_tensor_to_numpy_image(image)
    if not custom_prompt and show_bboxes:
        def get_bboxes(image_id, sent_id):
            original_bboxes = sent_info[sent_id]['coords_original']
            new_bboxes = process_bboxes([original_image.shape] * len(original_bboxes), original_bboxes, collate_fn)
            return new_bboxes
        bboxes = get_bboxes(dicom_id, sent_id)
        numpy_image = draw_bounding_boxes(to_rgb(torch.tensor(numpy_image)), bboxes)
        if display_attn:
            attn_numpy_image = draw_bounding_boxes(to_rgb(torch.tensor(attn_numpy_image)), bboxes)
    image_placeholder.image(
        numpy_image,
        use_column_width='always')
    st.markdown('**Prompt**: ' + prompt)
if display_attn:
    with attn_container:
        st.image(attn_numpy_image, use_column_width='always')
if annotations_name != "":
    with st.expander('All Annotations', expanded=False):
        if anonymize_models:
            df_without_checkpoint = df.copy()
            del df_without_checkpoint['checkpoint_name']
            st.write(df_without_checkpoint)
        else:
            st.write(df)
print("done")
