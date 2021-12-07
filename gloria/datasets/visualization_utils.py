# import torch
# import gloria
# import pandas as pd
# import matplotlib.pyplot as plt
# from PIL import Image
# import numpy as np
# import os
# import cv2
# from torch import nn
# from jupyter_innotater import *
# import pickle as pkl
# from tqdm import tqdm
# from torchmetrics import AUROC, AveragePrecision, ROC, PrecisionRecallCurve
# from ..gloria import builder, utils
# from .mimic_data import ImaGenomeDataModule, MimicCxrFiler, ImaGenomeFiler, bbox_to_mask, mask_to_bbox
# from .mimic_for_gloria import GloriaCollateFn, normalize, original_tensor_to_numpy_image


# def process_bboxes(model, image_shape, bboxes):
#     gloria_collate_fn = GloriaCollateFn(model.cfg, 'test')
#     new_bboxes = []
#     for bbox in bboxes:
#         box_mask = bbox_to_mask(bbox, image_shape)
# #         box_mask = np.array((normalize(box_mask) * 2 - 1) * 255, dtype=np.uint8)
#         box_mask = original_tensor_to_numpy_image(box_mask)
#         new_box_mask = gloria_collate_fn.process_img([box_mask], 'cpu')
#         new_box_mask = new_box_mask > 0
#         coords = mask_to_bbox(new_box_mask[0, 0])
#         new_bboxes.append(coords)
#     return new_bboxes


# def get_batch(cfg, texts, imgs, device):
#     gloria_collate_fn = GloriaCollateFn(cfg, 'test', device=device)
#     imgs = [original_tensor_to_numpy_image(img) for img in imgs]
#     return gloria_collate_fn.get_batch(imgs, texts)


# def plot_attn_maps(attn_maps, imgs, sents, epoch_idx=0, batch_idx=0, nvis=1):

#     img_set, _ = utils.build_attention_images(
#         imgs,
#         attn_maps,
# #         max_word_num=self.cfg.data.text.word_num,
#         nvis=nvis,
# #         rand_vis=self.cfg.train.rand_vis,
#         sentences=sents,
#     )

#     if img_set is not None:
#         return Image.fromarray(img_set)


# def plot_attention_from_raw(images, reports, model, filename='attention.jpg'):
#     reports = [report[report.index('FINDINGS:'):] if 'FINDINGS:' in report else report for report in reports]
#     batch_size = len(images)
#     batch = get_batch(model.cfg, reports, images, 'cuda')
#     img_emb_l, img_emb_g, text_emb_l, text_emb_g, sents = model(batch)
#     attn_maps = model.get_attn_maps(img_emb_l, text_emb_l, sents)
#     im = plot_attn_maps(attn_maps, batch['imgs'].cpu(), sents, nvis=batch_size)
#     im.save(filename)


# def draw_bounding_boxes(image, bboxes, color=(255, 0, 0)):
#     thickness = image.shape[0] // 100
#     for bbox in bboxes:
#         image = cv2.rectangle(image, bbox[:2], bbox[2:], color, thickness)
#     return image


# def get_bounding_boxes_mask(image_shape, bboxes):
#     image = np.zeros(image_shape)
#     image = draw_bounding_boxes(image, bboxes, color=1)
#     return image == 1


# def show_attention_from_raw(batch, model):
#     batch_size = len(batch['imgs'])
#     img_emb_l, img_emb_g, text_emb_l, text_emb_g, sents = model(batch)
#     attn_maps = model.get_attn_maps(img_emb_l, text_emb_l, sents)
#     im = attn_maps[0][0].sum(0).cpu().detach().numpy()
#     return im


# def to_rgb(image):
#     return np.array((normalize(image) * 255).int().unsqueeze(-1).expand(*image.shape, 3).cpu(), dtype=np.uint8)


# def get_ent_to_bbox(sent_labels, sent_contexts, sent_bbox_names):
#     ent_to_bbox = {}
#     for label, context, bbox in zip(sent_labels, sent_contexts, sent_bbox_names):
#         if (label, context) not in ent_to_bbox.keys():
#             ent_to_bbox[(label, context)] = set()
#         ent_to_bbox[(label, context)].add(bbox)
#     return ent_to_bbox


def get_ent_to_bbox_from_row(row):
    return get_ent_to_bbox(eval(row['sent_labels']), eval(row['sent_contexts']), eval(row['bbox_names']))


# def sent_bboxes_to_segmentation_label(shape, sent_bboxes):
#     segmentation_label = torch.zeros(shape, dtype=torch.bool)
#     for bbox in sent_bboxes:
#         segmentation_label = segmentation_label | bbox_to_mask(bbox, shape)
#     return segmentation_label


# def process_instance(instance, model, replace_sentences_with=None):
#     patient_id = next(iter(instance.keys()))
#     study_id = next(iter(instance[patient_id].keys()))
#     instance = instance[patient_id][study_id]
#     dicom_id = next(iter(instance['images'].keys()))
#     image = instance['images'][dicom_id]
#     sent_ids = sorted(list(instance['objects'][dicom_id]['sent_to_bboxes'].keys()))
#     sents, bbox_names, new_bboxes, attentions, images, labels, contexts = [], [], [], [], [], [], []
#     aurocs, avg_precisions, roc_curves, pr_curves = [], [], [], []
#     auroc, avg_precision, roc, pr_curve = AUROC(), AveragePrecision(), ROC(), PrecisionRecallCurve()
#     for i, sent_id in enumerate(sent_ids):
#         sent_info = instance['objects'][dicom_id]['sent_to_bboxes'][sent_id]
#         sent = sent_info['sentence'] if replace_sentences_with is None else replace_sentences_with
#         sents.append(sent)
#         bbox_names.append(sent_info['bboxes'])
#         sent_bboxes = sent_info['coords_original']
#         labels.append(sent_info['labels'])
#         contexts.append(sent_info['contexts'])
#         sent_images = []
#         image1 = draw_bounding_boxes(to_rgb(image), sent_bboxes)
#         sent_images.append(image1)
#         batch = get_batch(model.cfg, [sents[-1]], [image], 'cuda')
#         new_sent_bboxes = process_bboxes(model, image.shape, sent_bboxes)
#         new_bboxes.append(new_sent_bboxes)
#         image2 = batch['imgs'][0, 0]
#         image2 = draw_bounding_boxes(to_rgb(image2), new_sent_bboxes)
#         sent_images.append(image2)
#         attn = torch.tensor(show_attention_from_raw(batch, model))
#         attn = attn.reshape(1, 1, *attn.shape)
#         new_attn = nn.Upsample(size=image2.shape[:2], mode="bilinear")(attn)
#         attentions.append(new_attn[0, 0])
#         new_attn = draw_bounding_boxes(to_rgb(new_attn[0, 0]), new_sent_bboxes)
#         sent_images.append(new_attn)
#         segmentation_label = sent_bboxes_to_segmentation_label(attentions[-1].shape, new_sent_bboxes)
#         if segmentation_label.sum() > 0:
#             pred, target = attentions[-1].reshape(-1), segmentation_label.reshape(-1).long()
#             aurocs.append(auroc(pred, target).item())
#             avg_precisions.append(avg_precision(pred, target).item())
#             roc_curves.append(roc(pred, target))
#             pr_curves.append(pr_curve(pred, target))
#         else:
#             aurocs.append(None)
#             avg_precisions.append(None)
#             roc_curves.append(None)
#             pr_curves.append(None)
#         images.append(sent_images)
#     return dict(
#         bbox_names=bbox_names,
#         new_bboxes=new_bboxes,
#         attentions=attentions,
#         images=images,
#         sents=sents,
#         sent_ids=sent_ids,
#         labels=labels,
#         contexts=contexts,
#         aurocs=aurocs,
#         avg_precisions=avg_precisions,
#         roc_curves=roc_curves,
#         pr_curves=pr_curves
#     )


# def get_and_process_row(path, patient_id, study_id, dicom_id, sent_id, sent, bbox_names, bboxes, sent_labels,
#                         sent_contexts, auroc, avg_precision, roc_curve, pr_curve, sent_images, attention,
#                         plot=False):
#     dicom_sent_id = 'dicom_%s_sent_%s' % (dicom_id, sent_id)
#     row = [
#         dicom_sent_id,
#         patient_id,
#         study_id,
#         dicom_id,
#         sent_id,
#         sent,
#         str(bbox_names),
#         str(bboxes),
#         str(sent_labels),
#         str(sent_contexts),
#         auroc,
#         avg_precision
#     ]
#     if auroc is not None:
#         np.save(os.path.join(path, 'roc_curves', dicom_sent_id), roc_curve)
#         np.save(os.path.join(path, 'pr_curves', dicom_sent_id), pr_curve)
#     Image.fromarray(sent_images[0]).save(
#         os.path.join(path, 'bbox_images0', dicom_sent_id + '.jpg'))
#     Image.fromarray(sent_images[1]).save(
#         os.path.join(path, 'bbox_images1', dicom_sent_id + '.jpg'))
#     Image.fromarray(sent_images[2]).save(
#         os.path.join(path, 'bbox_images2', dicom_sent_id + '.jpg'))
#     np.save(os.path.join(path, 'attentions', dicom_sent_id), attention)
#     if plot:
#         fig = plt.figure(figsize=(15, 5), tight_layout=True)
#         a1 = plt.subplot2grid((2, 5), (1, 0), rowspan = 1, colspan = 1)
#         a2 = plt.subplot2grid((2, 5), (1, 1), rowspan = 1, colspan = 1)
#         a3 = plt.subplot2grid((2, 5), (1, 2), rowspan = 1, colspan = 1)
#         a4 = plt.subplot2grid((2, 5), (1, 3), rowspan = 1, colspan = 1)
#         a5 = plt.subplot2grid((2, 5), (1, 4), rowspan = 1, colspan = 1)
#         text_a1 = plt.subplot2grid((2, 5), (0, 0), rowspan = 1, colspan = 3)
#         text_a2 = plt.subplot2grid((2, 5), (0, 3), rowspan = 1, colspan = 1)
#         text_a3 = plt.subplot2grid((2, 5), (0, 4), rowspan = 1, colspan = 1)
#         a1.imshow(sent_images[0])
#         a2.imshow(sent_images[1])
#         a3.imshow(sent_images[2])
#         if roc_curve is not None:
#             a4.plot(roc_curve[0], roc_curve[1])
#             a4.set_xlabel('1-Specificity')
#             a4.set_ylabel('Sensitivity/Recall')
#         if pr_curve is not None:
#             a5.plot(pr_curve[1], pr_curve[0])
#             a5.set(xlim=(0, 1), ylim=(0, 1))
#             a5.set_xlabel('Sensitivity/Recall')
#             a5.set_ylabel('Precision')
#         text = 'sentence: %s' % sent
#         ent_to_bbox = get_ent_to_bbox(sent_labels, sent_contexts, bbox_names)
#         for k, v in ent_to_bbox.items():
#             text += '\n' + str(k) + ': ' + str(v)
#         text_a1.text(.0, .5, text, horizontalalignment='left', verticalalignment='bottom')
#         text_a1.set_axis_off()
#         if auroc is not None:
#             text = 'auroc: %f' % auroc
#             text_a2.text(.5, .5, text, horizontalalignment='center', verticalalignment='bottom')
#         text_a2.set_axis_off()
#         if avg_precision is not None:
#             text = 'avg_precision: %f' % avg_precision
#             text_a3.text(.5, .5, text, horizontalalignment='center', verticalalignment='bottom')
#         text_a3.set_axis_off()
#         plt.savefig(os.path.join(path, 'sentence_figures', dicom_sent_id + '.jpg'))
#     return row


# def get_and_save_instance_results(path, dataset=None, model=None, example_indices=None, dicom_ids=None,
#                                   modification_mode='read', plot=False,
#                                   replace_sentences_with=None, save_every=10):
#     assert modification_mode in {'read', 'append', 'overwrite'}
#     if os.path.exists(os.path.join(path, 'sentences.csv')):
#         df_from_file = pd.read_csv(os.path.join(path, 'sentences.csv'))
#         if modification_mode == 'read':
#             return df_from_file
#     else:
#         df_from_file = None
#     assert dataset is not None
#     if modification_mode in {'read', 'overwrite'}:
#         assert model is not None
#     def save_rows(new_rows, df=None):
#         new_df = pd.DataFrame(new_rows, columns=[
#             'dicom_sent_id',
#             'patient_id',
#             'study_id',
#             'dicom_id',
#             'sent_id',
#             'sentence',
#             'bbox_names',
#             'bboxes',
#             'sent_labels',
#             'sent_contexts',
#             'auroc',
#             'avg_precision',
#         ])
#         if df is not None:
#             new_df = pd.concat([df, new_df])
#         new_df.to_csv(os.path.join(path, 'sentences.csv'), index=False)
#         return new_df
#     if example_indices is None:
#         example_indices = range(len(dataset))
#     example_indices = list(example_indices)
#     if not os.path.exists(path):
#         os.mkdir(path)
#     if not os.path.exists(os.path.join(path, 'roc_curves')):
#         os.mkdir(os.path.join(path, 'roc_curves'))
#     if not os.path.exists(os.path.join(path, 'pr_curves')):
#         os.mkdir(os.path.join(path, 'pr_curves'))
#     if not os.path.exists(os.path.join(path, 'bbox_images0')):
#         os.mkdir(os.path.join(path, 'bbox_images0'))
#     if not os.path.exists(os.path.join(path, 'bbox_images1')):
#         os.mkdir(os.path.join(path, 'bbox_images1'))
#     if not os.path.exists(os.path.join(path, 'bbox_images2')):
#         os.mkdir(os.path.join(path, 'bbox_images2'))
#     if not os.path.exists(os.path.join(path, 'attentions')):
#         os.mkdir(os.path.join(path, 'attentions'))
#     if not os.path.exists(os.path.join(path, 'sentence_figures')):
#         os.mkdir(os.path.join(path, 'sentence_figures'))
#     info = []
#     num_added = 0
#     for i in tqdm(example_indices, total=len(example_indices)):
#         instance = dataset[i]
#         patient_id = next(iter(instance.keys()))
#         study_id = next(iter(instance[patient_id].keys()))
#         dicom_id = next(iter(instance[patient_id][study_id]['images'].keys()))
#         if dicom_ids is not None and dicom_id not in dicom_ids:
#             continue
#         if modification_mode == 'overwrite' and df_from_file is not None:
#             df_from_file = df_from_file[df_from_file.dicom_id != dicom_id]
#         if df_from_file is not None and dicom_id in set(df_from_file.dicom_id):
#             continue
#         assert model is not None
#         outs = process_instance(instance, model, replace_sentences_with=replace_sentences_with)
#         for sent_id, sent, bbox_names, bboxes, \
#             sent_labels, sent_contexts, \
#             auroc, avg_precision, roc_curve, pr_curve, \
#             sent_images, attention in zip(
#                 outs['sent_ids'], outs['sents'], outs['bbox_names'], outs['new_bboxes'],
#                 outs['labels'], outs['contexts'],
#                 outs['aurocs'], outs['avg_precisions'], outs['roc_curves'], outs['pr_curves'],
#                 outs['images'], outs['attentions'],
#         ):
#             row = get_and_process_row(
#                 path, patient_id, study_id, dicom_id, sent_id, sent, bbox_names, bboxes, sent_labels,
#                 sent_contexts, auroc, avg_precision, roc_curve, pr_curve, sent_images, attention, plot=plot)
#             info.append(row)
#         num_added += 1
#         if num_added % save_every == 0:
#             save_rows(info, df=df_from_file)
#     return save_rows(info, df=df_from_file)


# def annotate(path, dataset=None, model=None, num_examples=None, labels=None, selector=None, show_identifier=True):
#     df = get_and_save_instance_results(path, dataset=dataset, model=model, num_examples=num_examples)
#     if labels is None:
#         labels = [0] * len(df)
#     sentences = df.sentence.tolist()
#     entities = []
#     for i, row in df.iterrows():
#         entities.append('')
#         ent_to_bbox = get_ent_to_bbox_from_row(row)
#         for k, v in ent_to_bbox.items():
#             entities[-1] += str(k) + ': ' + str(v) + '\n'
#     files = [name + '.jpg' for name in df.dicom_sent_id.tolist()]
#     classes = ['0 - Unselected', '1 - Positive', '2 - Ambiguous', '3 - Negative']
#     indexes = None if selector is None else df[df.apply(selector, axis=1)].index.tolist()
#     innotations = [TextInnotation(df.dicom_sent_id.tolist())]
#     innotations.extend([
#         TextInnotation(sentences),
#         TextInnotation(entities),
# #         ImageInnotation(files, path=os.path.join(path, 'bbox_images0'), width=10, height=10),
#         ImageInnotation(files, path=os.path.join(path, 'bbox_images1'), width=200, height=200),
#         ImageInnotation(files, path=os.path.join(path, 'bbox_images2'), width=200, height=200),
#     ])
#     return Innotater(
#         innotations,
#         MultiClassInnotation(labels, classes=classes),
#         indexes=indexes
#     ), labels


# def compute_metrics(path, dataset=None, model=None, num_examples=None, selector=None, overwrite=False, save=False):
#     df = get_and_save_instance_results(path, dataset=dataset, model=model, num_examples=num_examples)
#     if 'auroc' not in df.keys() or overwrite:
#         def compute_auroc(row):
#             sent_attention = torch.tensor(np.load(os.path.join(path, 'attentions', row.dicom_sent_id + '.npy')))
#             segmentation_label = sent_bboxes_to_segmentation_label(sent_attention.shape, eval(row.bboxes))
#             if segmentation_label.sum() > 0:
#                 return AUROC()(sent_attention.reshape(-1), segmentation_label.reshape(-1).long()).item()
#         df['auroc'] = df.apply(compute_auroc, axis=1)
#     if 'avg_precision' not in df.keys() or overwrite:
#         def compute_avg_precision(row):
#             sent_attention = torch.tensor(np.load(os.path.join(path, 'attentions', row.dicom_sent_id + '.npy')))
#             segmentation_label = sent_bboxes_to_segmentation_label(sent_attention.shape, eval(row.bboxes))
#             if segmentation_label.sum() > 0:
#                 return AveragePrecision()(sent_attention.reshape(-1), segmentation_label.reshape(-1).long()).item()
#         df['avg_precision'] = df.apply(compute_avg_precision, axis=1)
#     if save:
#         df.to_csv(os.path.join(path, 'sentences.csv'), index=False)
#     if selector is not None:
#         df = df[df.apply(selector, axis=1)]
#     print('auroc', df.auroc[~df.auroc.isnull()].mean())
#     print('avg_precision', df.avg_precision[~df.avg_precision.isnull()].mean())


class RowContainsOrDoesNotContainSelector:
    def __init__(self, contains=None, does_not_contain=None, only_contains=False):
        assert contains is not None or does_not_contain is not None
        if only_contains:
            assert does_not_contain is None
        self.contains = set(contains) if contains is not None else None
        self.does_not_contain = set(does_not_contain) if does_not_contain is not None else None
        self.only_contains = only_contains

    def get_row_set(self, row):
        raise NotImplementedError

    def __call__(self, row):
        row_set = self.get_row_set(row)
        if self.only_contains:
            return self.contains == row_set
        else:
            return_bool = True
            if self.contains is not None:
                return_bool = return_bool and len(self.contains - row_set) == 0
            if self.does_not_contain is not None:
                return_bool = return_bool and len(row_set - self.does_not_contain) == len(row_set)
            return return_bool


class RowLabelAndContextSelector(RowContainsOrDoesNotContainSelector):
    def get_row_set(self, row):
        return set(get_ent_to_bbox_from_row(row).keys())


class RowBBoxSelector(RowContainsOrDoesNotContainSelector):
    def get_row_set(self, row):
        return set(eval(row['bbox_names']))


class OrSelector:
    def __init__(self, *selectors):
        self.selectors = selectors
    
    def __call__(self, row):
        return_bool = False
        for selector in self.selectors:
            return_bool = return_bool or selector(row)
        return return_bool


from gloria.datasets.mimic_for_gloria import normalize
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2


def bbox_to_mask(bbox, image_shape):
    box_mask = torch.zeros(image_shape, dtype=torch.bool)
    box_mask[bbox[1]:bbox[3] + 1, bbox[0]:bbox[2] + 1] = 1
    return box_mask


def mask_to_bbox(box_mask):
    if box_mask.sum() == 0:
        return [-1, -1, -1, -1]
    indices0 = torch.arange(box_mask.shape[0])
    indices1 = torch.arange(box_mask.shape[1])
    indices0 = indices0.unsqueeze(1).expand(*box_mask.shape)[box_mask]
    indices1 = indices1.unsqueeze(0).expand(*box_mask.shape)[box_mask]
    return [indices1.min().item(), indices0.min().item(), indices1.max().item(), indices0.max().item()]


def to_rgb(image):
    return np.array((normalize(image) * 255).int().unsqueeze(-1).expand(*image.shape, 3).cpu(), dtype=np.uint8)


def get_ent_to_bbox(sent_labels, sent_contexts, sent_bbox_names):
    ent_to_bbox = {}
    for label, context, bbox in zip(sent_labels, sent_contexts, sent_bbox_names):
        if (label, context) not in ent_to_bbox.keys():
            ent_to_bbox[(label, context)] = set()
        ent_to_bbox[(label, context)].add(bbox)
    return ent_to_bbox


def sent_bboxes_to_segmentation_label(shape, sent_bboxes):
    segmentation_label = torch.zeros(shape, dtype=torch.bool)
    for bbox in sent_bboxes:
        segmentation_label = segmentation_label | bbox_to_mask(bbox, shape)
    return segmentation_label


def draw_bounding_boxes(image, bboxes, color=(255, 0, 0)):
    thickness = image.shape[0] // 100
    for bbox in bboxes:
        image = cv2.rectangle(image, bbox[:2], bbox[2:], color, thickness)
    return image


from PIL import Image
# Image.fromarray(A).save("your_file.jpeg")

def plot_info(attn_overlay_func, info, path=None, add_no_attn_bar=False):
    if path is not None:
        if not os.path.exists(path):
            os.mkdir(path)
        if not os.path.exists(os.path.join(path, 'sentence_figures')):
            os.mkdir(os.path.join(path, 'sentence_figures'))
        if not os.path.exists(os.path.join(path, 'image_with_bboxes')):
            os.mkdir(os.path.join(path, 'image_with_bboxes'))
        if not os.path.exists(os.path.join(path, 'attention_with_bboxes')):
            os.mkdir(os.path.join(path, 'attention_with_bboxes'))
    figs = []
    for dicom_sent_id, sent, sent_labels, sent_contexts, bbox_names, \
        image, bboxes, attn, auroc, avg_precision, roc_curve, pr_curve in tqdm(zip(
        *(info[k] for k in [
            'dicom_sent_id', 'sentence', 'sent_labels', 'sent_contexts', 'bbox_names',
            'image', 'bboxes', 'attn', 'auroc', 'avg_precision', 'roc_curve', 'pr_curve'])
    ), total=len(info['dicom_sent_id'])):
        image = torch.tensor(image)
        fig = plt.figure(figsize=(15, 5), tight_layout=True)
        a1 = plt.subplot2grid((2, 5), (1, 0), rowspan = 1, colspan = 1)
        a2 = plt.subplot2grid((2, 5), (1, 1), rowspan = 1, colspan = 1)
        a3 = plt.subplot2grid((2, 5), (1, 2), rowspan = 1, colspan = 1)
        a4 = plt.subplot2grid((2, 5), (1, 3), rowspan = 1, colspan = 1)
        a5 = plt.subplot2grid((2, 5), (1, 4), rowspan = 1, colspan = 1)
        text_a1 = plt.subplot2grid((2, 5), (0, 0), rowspan = 1, colspan = 3)
        text_a2 = plt.subplot2grid((2, 5), (0, 3), rowspan = 1, colspan = 1)
        text_a3 = plt.subplot2grid((2, 5), (0, 4), rowspan = 1, colspan = 1)
#         a1.imshow(draw_bounding_boxes(to_rgb(original_image), original_bboxes))
        image_with_bboxes = draw_bounding_boxes(to_rgb(image), bboxes)
        Image.fromarray(image_with_bboxes).save(os.path.join(path, 'image_with_bboxes', dicom_sent_id + '.jpg'))
        a2.imshow(image_with_bboxes)
        new_attn = attn_overlay_func(attn, image.shape[:2])
        if add_no_attn_bar:
            zeros_bar = torch.zeros((max(int(new_attn.shape[0] * .01), 1), new_attn.shape[1]))
            no_attn_bar = torch.ones((max(int(new_attn.shape[0] * .05), 1), new_attn.shape[1])) * (1 - attn.sum())
            new_attn = torch.cat([new_attn, zeros_bar, no_attn_bar], 0)
        attention_with_bboxes = draw_bounding_boxes(to_rgb(new_attn), bboxes)
        Image.fromarray(attention_with_bboxes).save(os.path.join(path, 'attention_with_bboxes', dicom_sent_id + '.jpg'))
        a3.imshow(attention_with_bboxes)
        if roc_curve is not None:
            a4.plot(roc_curve[0], roc_curve[1])
            a4.set_xlabel('1-Specificity')
            a4.set_ylabel('Sensitivity/Recall')
        if pr_curve is not None:
            a5.plot(pr_curve[1], pr_curve[0])
            a5.set(xlim=(0, 1), ylim=(0, 1))
            a5.set_xlabel('Sensitivity/Recall')
            a5.set_ylabel('Precision')
        text = 'sentence: %s' % sent
        ent_to_bbox = get_ent_to_bbox(sent_labels, sent_contexts, bbox_names)
        for k, v in ent_to_bbox.items():
            text += '\n' + str(k) + ': ' + str(v)
        text_a1.text(.0, .5, text, horizontalalignment='left', verticalalignment='bottom')
        text_a1.set_axis_off()
        if auroc is not None:
            text = 'auroc: %f' % auroc
            text_a2.text(.5, .5, text, horizontalalignment='center', verticalalignment='bottom')
        text_a2.set_axis_off()
        if avg_precision is not None:
            text = 'avg_precision: %f' % avg_precision
            text_a3.text(.5, .5, text, horizontalalignment='center', verticalalignment='bottom')
        text_a3.set_axis_off()
        if path is not None:
            plt.savefig(os.path.join(path, 'sentence_figures', dicom_sent_id + '.jpg'))
        figs.append(fig)
    return figs


import os
from tqdm import tqdm

def path_and_rows_to_info(path, rows=None):
    if rows is None:
        rows = pd.read_csv(os.path.join(path, 'sentences.csv'))
    info = {k: rows[k].tolist() for k in rows.keys()}
    for i, dicom_sent_id in tqdm(enumerate(info['dicom_sent_id']), total=len(info['dicom_sent_id'])):
        for k in ['bbox_names', 'sent_labels', 'sent_contexts', 'bboxes']:
            info[k][i] = eval(info[k][i])
        for k in ['image', 'attn', 'roc_curve', 'pr_curve']:
            if k not in info.keys():
                info[k] = []
            array = np.load(os.path.join(path, k, dicom_sent_id + '.npy'), allow_pickle=True)
            info[k].append(array if array.ndim > 0 else None)
    return info


from torch import nn
import skimage

def pyramid_attn_overlay(attn, image_shape):
    new_attn = torch.tensor(attn)
    new_attn = new_attn.unsqueeze(-1).expand(*new_attn.shape, 3)
    new_attn = skimage.transform.pyramid_expand(
        new_attn, sigma=20, upscale=image_shape[0] // new_attn.shape[0], multichannel=True)
    new_attn = torch.tensor(new_attn[:, :, 0])
    new_attn = nn.Upsample(size=image_shape)(new_attn.reshape(1, 1, *new_attn.shape))[0, 0]
    return new_attn
