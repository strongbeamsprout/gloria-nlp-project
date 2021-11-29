from pytorch_lightning.callbacks.base import Callback
from torchmetrics import Metric
from torchmetrics.functional import roc, precision_recall_curve, auroc, average_precision
from torch.distributions.categorical import Categorical
import copy
from gloria.datasets.mimic_for_gloria import GloriaCollateFn, normalize, original_tensor_to_numpy_image
import torch
import os
from torch import nn
import numpy as np
import pandas as pd
import wandb


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


def discrete_entropy(dist):
    return Categorical(dist).entropy()


def get_no_attn_weight(dist):
    return 1 - dist.sum(-1)


class Metrics:
    def __init__(self):
        self.attn_bbox_metrics = {
            'roc_curve': roc,
            'pr_curve': precision_recall_curve,
            'auroc': auroc,
            'avg_precision': average_precision,
        }
        self.attn_entropy = discrete_entropy
        self.no_attn_weight = get_no_attn_weight

    def __call__(self, attn, attn_overlay, bboxes):
        metrics = {'attn_entropy': self.attn_entropy(attn.reshape(-1)), 'no_attn_weight': self.no_attn_weight(attn.reshape(-1))}
        segmentation_label = sent_bboxes_to_segmentation_label(attn_overlay.shape, bboxes)
        for k, v in self.attn_bbox_metrics.items():
            if segmentation_label.sum() > 0:
                metrics[k] = v(attn_overlay.reshape(-1), segmentation_label.reshape(-1).long())
            else:
                metrics[k] = None
        return metrics


def get_grad_mask(args):
    s, image = args
    return torch.autograd.grad(s, image, retain_graph=True)[0].sum(0) > 0


def yield_results(func, args):
    for a in args:
        yield func(a)


def get_image_masks_for_outputs(image_encoder, image_shape):
    device = next(iter(image_encoder.parameters())).device
    image_encoder = copy.deepcopy(image_encoder).eval()
    for p in image_encoder.parameters():
        p.data.fill_(1.)
        p.requires_grad = False
    image = torch.ones(image_shape, device=device, requires_grad=True)
    with torch.autograd.enable_grad():
        output = image_encoder(image.unsqueeze(0)).squeeze(0)
        spatial_shape_out = output.shape[1:]
        vecsum = list(output.sum(0).reshape(-1))
    # with mpd.Pool() as p:
    # with mp.Pool() as p:
    #     masks = p.imap(get_grad_mask, [(s, image) for s in tqdm(vecsum, total=len(vecsum))])
    #     return spatial_shape_out, masks
    return spatial_shape_out, yield_results(get_grad_mask, [(s, image) for s in tqdm(vecsum, total=len(vecsum))])


def masks_to_windows_sequential(spatial_shape_out, mask_generator):
    windows = torch.stack(
        [masks_to_windows(mask, num_spatial_positions=len(spatial_shape_out))
         for mask in tqdm(mask_generator, total=np.prod(spatial_shape_out))])
    return windows.reshape(*spatial_shape_out, -1)


def masks_to_windows(masks, num_spatial_positions=2):
    assert masks.dim() >= num_spatial_positions
    spatial_shape = masks.shape[-num_spatial_positions:]
    batch_shape = masks.shape[:-num_spatial_positions]
    windows = []
    for i in range(num_spatial_positions):
        aranged = torch.arange(spatial_shape[i], device=masks.device).reshape(
            *(1 if j != i else spatial_shape[j] for j in range(num_spatial_positions))
        ).expand(*batch_shape, *spatial_shape)
        minimum = (aranged - aranged.max() - 1) * masks
        for j in range(num_spatial_positions):
            minimum = minimum.min(-1)[0]
        maximum = (aranged + 1) * masks
        for j in range(num_spatial_positions):
            maximum = maximum.max(-1)[0]
        windows.extend([minimum + aranged.max() + 1, maximum])
    windows = windows[::2] + windows[1::2]
    return torch.stack(windows).permute(*(i + 1 for i in range(len(batch_shape))), 0)


def windows_to_masks(spatial_shape, windows):
    b = windows.shape[0]
    dim_wise_indices = [torch.arange(d, device=windows.device) for d in spatial_shape]
    dim_wise_masks = [(index >= windows[:, i].unsqueeze(1)) & (index < windows[:, len(spatial_shape) + i].unsqueeze(1))
                      for i, index in enumerate(dim_wise_indices)]
    masks = torch.ones((b, *spatial_shape), device=windows.device, dtype=torch.bool)
    for i, dim_wise_mask in enumerate(dim_wise_masks):
        d = dim_wise_mask.shape[1]
        reshaped_dim_wise_mask = dim_wise_mask.reshape(b, *(d if i == j else 1 for j in range(len(spatial_shape))))
        masks = masks & reshaped_dim_wise_mask
    return masks


def get_train_outputs(outputs):
    assert len(outputs) > 0
    if len(outputs) > 1:
        raise NotImplementedError
    outputs = outputs[0]
    assert len(outputs) > 0
    if len(outputs) > 1:
        raise NotImplementedError
    outputs = outputs[0]
    outputs = outputs.extra
    return outputs


class EvaluateLocalization(Callback):
    def __init__(self, gloria_collate_fn, save_dir, batch_size=None, attn_overlay_mode='upsample', log_train_every=100):
        super().__init__()
        self.gloria_collate_fn = gloria_collate_fn
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.attn_overlay_mode = attn_overlay_mode
        self.log_train_every = log_train_every
        self.metrics = Metrics()
        self.shape_to_windows_cache = {}

    def get_windows(self, image_shape, gloria=None, cache=True):
        image_shape = (1, *image_shape[1:])
        if image_shape not in self.shape_to_windows_cache.keys():
            print('Computing image windows based on image shape')
            assert gloria is not None
            class ImageEncoderWrapper(nn.Module):
                def __init__(img_encoder):
                    self.img_encoder = img_encoder
                def forward(self, imgs):
                    img_feat_g, img_emb_l = self.img_encoder(imgs, get_local=True)
                    return img_emb_l
            windows = masks_to_windows_sequential(
                *get_image_masks_for_outputs(
                    ImageEncoderWrapper(gloria.img_encoder), image_shape)).cpu()
            if cache:
                print('Caching')
                self.shape_to_windows_cache[image_shape] = windows
            print('done')
            bb = windows.reshape(-1, windows.shape[-1])[0]
            print('bounding box shape:', bb[bb.shape[0] // 2:] - bb[:bb.shape[0] // 2])
        return self.shape_to_windows_cache[image_shape]

    def process_bboxes(self, image_shapes, bboxes):
        new_bboxes = []
        box_masks = []
        for shape, bbox in zip(image_shapes, bboxes):
            box_mask = bbox_to_mask(bbox, shape)
            box_masks.append(original_tensor_to_numpy_image(box_mask))
        new_box_masks = self.gloria_collate_fn.process_img(box_masks, 'cpu')
        new_box_masks = new_box_masks > 0
        new_bboxes = [mask_to_bbox(new_box_mask[0]) for new_box_mask in new_box_masks]
        return new_bboxes

    def process_instances(self, instances):
        info = {
            'dicom_sent_id': [],
            'patient_id': [],
            'study_id': [],
            'dicom_id': [],
            'sent_id': [],
            'sentence': [],
            'bbox_names': [],
            'sent_labels': [],
            'sent_contexts': [],
            'original_image': [],
            'original_bboxes': [],
        }
        bbox_names = []
        original_image_shapes = []
        bboxes = []
        for instance in instances:
            patient_id = next(iter(instance.keys()))
            study_id = next(iter(instance[patient_id].keys()))
            instance = instance[patient_id][study_id]
            dicom_id = next(iter(instance['images'].keys()))
            image = original_tensor_to_numpy_image(instance['images'][dicom_id])
            if 'sent_id' in instance.keys():
                sent_ids = [instance['sent_id']]
            else:
                sent_ids = sorted(list(instance['objects'][dicom_id]['sent_to_bboxes'].keys()))
            for i, sent_id in enumerate(sent_ids):
                dicom_sent_id = 'dicom_%s_sent_%s' % (dicom_id, sent_id)
                sent_info = instance['objects'][dicom_id]['sent_to_bboxes'][sent_id]
                sent = sent_info['sentence']
                info['dicom_sent_id'].append(dicom_sent_id)
                info['patient_id'].append(patient_id)
                info['study_id'].append(study_id)
                info['dicom_id'].append(dicom_id)
                info['sent_id'].append(sent_id)
                info['sentence'].append(sent)
                info['bbox_names'].append(sent_info['bboxes'])
                info['sent_labels'].append(sent_info['labels'])
                info['sent_contexts'].append(sent_info['contexts'])
                info['original_image'].append(image)
                info['original_bboxes'].append(sent_info['coords_original'])
                for bbox_name, bbox in zip(sent_info['bboxes'], sent_info['coords_original']):
                    if (dicom_id, bbox) not in bbox_names:
                        bbox_names.append((dicom_id, bbox_name))
                        original_image_shapes.append(image.shape)
                        bboxes.append(bbox)
        return info, bbox_names, original_image_shapes, bboxes

    def evaluate_instances(self, info, attn_overlay_mode='upsample'):
        evaluation_info = {}
        for image, attn, bboxes in zip(info['image'], info['attn'], info['bboxes']):
            attn_overlay = self.get_attn_overlay(attn, image.shape, mode=attn_overlay_mode)
            metrics = self.metrics(torch.tensor(attn), attn_overlay, bboxes)
            for k, v in metrics.items():
                if k not in evaluation_info.keys():
                    evaluation_info[k] = []
                evaluation_info[k].append(v.item() if isinstance(v, torch.Tensor) else v)
        return evaluation_info

    def info_to_df(self, info):
        columns = [
            'dicom_sent_id',
            'patient_id',
            'study_id',
            'dicom_id',
            'sent_id',
            'sentence',
            'bbox_names',
            'sent_labels',
            'sent_contexts',
            'bboxes',
            'auroc',
            'avg_precision',
            'attn_entropy',
            'no_attn_weight',
        ]
        rows = [info[col] for col in columns if col in info.keys()]
        rows = list(zip(*rows))
        df = pd.DataFrame(rows, columns=columns)
        return df

    def save_folder_files(self, info, path):
        folder_names = [
            'image',
            'attn',
            'roc_curve',
            'pr_curve',
        ]
        for folder in folder_names:
            if not os.path.exists(os.path.join(path, folder)):
                os.mkdir(os.path.join(path, folder))
            for dicom_sent_id, x in zip(info['dicom_sent_id'], info[folder]):
                np.save(os.path.join(path, folder, dicom_sent_id), x)

    def get_attn_overlay(self, attn, image_shape, mode='upsample'):
        attn = torch.tensor(attn)
        assert mode in {'windows', 'upsample', 'upsample_pyramid'}
        if mode == 'windows':
            assert image_shape in self.shape_to_windows_cache.keys()
            windows = self.get_windows(image_shape)
            raise NotImplementedError
            return new_attn
        elif mode in {'upsample', 'upsample_pyramid'}:
            new_attn = nn.Upsample(size=image_shape)(attn.reshape(1, 1, *attn.shape))[0, 0]
            if mode == 'upsample_pyramid':
                raise NotImplementedError
            return new_attn

    def plot_info(self, info, path=None, attn_overlay_mode='upsample'):
        figs = []
        for sent, sent_labels, sent_contexts, bbox_names, original_image, original_bboxes, \
            image, bboxes, attn, auroc, avg_precision, roc_curve, pr_curve in zip(
            info[k] for k in [
                'sentence', 'sent_labels', 'sent_contexts', 'bbox_names', 'original_image', 'original_bboxes'
                'image', 'bboxes', 'attn', 'auroc', 'avg_precision', 'roc_curve', 'pr_curve']
        ):
            fig = plt.figure(figsize=(15, 5), tight_layout=True)
            a1 = plt.subplot2grid((2, 5), (1, 0), rowspan = 1, colspan = 1)
            a2 = plt.subplot2grid((2, 5), (1, 1), rowspan = 1, colspan = 1)
            a3 = plt.subplot2grid((2, 5), (1, 2), rowspan = 1, colspan = 1)
            a4 = plt.subplot2grid((2, 5), (1, 3), rowspan = 1, colspan = 1)
            a5 = plt.subplot2grid((2, 5), (1, 4), rowspan = 1, colspan = 1)
            text_a1 = plt.subplot2grid((2, 5), (0, 0), rowspan = 1, colspan = 3)
            text_a2 = plt.subplot2grid((2, 5), (0, 3), rowspan = 1, colspan = 1)
            text_a3 = plt.subplot2grid((2, 5), (0, 4), rowspan = 1, colspan = 1)
            a1.imshow(draw_bounding_boxes(to_rgb(original_image), original_bboxes))
            a2.imshow(draw_bounding_boxes(to_rgb(image), bboxes))
            new_attn = self.get_attn_overlay(attn, image.shape[:2], mode=attn_overlay_mode)
            if attn.sum() < 1:
                no_attn_bar = torch.ones((max(int(new_attn.shape[0] * .05), 1), image.shape[1])) * (1 - attn.sum())
                new_attn = torch.cat([new_attn, no_attn_bar], 0)
            a3.imshow(draw_bounding_boxes(to_rgb(new_attn), bboxes))
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

    def instance_in_dataframe(self, instance, df):
        patient_id = next(iter(instance.keys()))
        study_id = next(iter(instance[patient_id].keys()))
        instance = instance[patient_id][study_id]
        dicom_id = next(iter(instance['images'].keys()))
        if 'sent_id' in instance.keys():
            sent_id = instance['sent_id']
            dicom_sent_id = 'dicom_%s_sent_%s' % (dicom_id, sent_id)
            return dicom_sent_id in df.dicom_sent_id
        else:
            return dicom_id in df.dicom_id

    def evaluate_and_save(self, path=None, instances=None, batch=None, outputs=None, pl_module=None, save_full_data=False,
                          plot=False):
        return_dict = {}
        if path is not None and not os.path.exists(path):
            os.mkdir(path)
        if instances is not None:
            assert batch is None and outputs is None
        elif batch is not None:
            assert instances is None
            instances = batch['instances']
            instance = instances[0]
            patient_id = next(iter(instance.keys()))
            study_id = next(iter(instance[patient_id].keys()))
            instance = instance[patient_id][study_id]
            if 'sent_id' not in instance.keys():
                batch, outputs = None, None
        else:
            raise Exception
        # load csv if it does exist
        if path is not None and os.path.exists(os.path.join(path, 'sentences.csv')):
            df_from_file = pd.read_csv(os.path.join(path, 'sentences.csv'))
        else:
            df_from_file = None
        info, bbox_names, original_image_shapes, bboxes = self.process_instances(instances)
        if batch is None:
            batch_size = len(info['original_image']) if self.batch_size is None else self.batch_size
            batches = (
                self.gloria_collate_fn.get_batch(
                    info['original_image'][i:i+batch_size], info['sentence'][i:i+batch_size])
                for i in range(0, len(info['original_image']), batch_size)
            )
        else:
            batches = [batch]
        # add reshaped image to info
        info['image'] = []
        for batch in batches:
            info['image'].extend(list(batch['imgs'][:, 0].detach().cpu().numpy()))
        # add new bounding boxes for reshaped image
        new_bboxes = self.process_bboxes(original_image_shapes, bboxes)
        new_bboxes = {k: v for k, v in zip(bbox_names, new_bboxes)}
        info['bboxes'] = []
        for i, (dicom_id, sent_bbox_names) in enumerate(zip(info['dicom_id'], info['bbox_names'])):
            info['bboxes'].append([new_bboxes[(dicom_id, name)] for name in sent_bbox_names])
        # add attention using model output
        info['attn'] = []
        if outputs is None:
            assert pl_module is not None
            for b in batches:
                b = pl_module.transfer_batch_to_device(b, pl_module.device)
                img_emb_l, img_emb_g, text_emb_l, text_emb_g, sents = pl_module.gloria(b)
                ams = pl_module.gloria.get_attn_maps(img_emb_l, text_emb_l, sents)
                ams = [am[0].mean(0).detach().cpu().numpy() for am in ams]
                info['attn'].extend(list(ams))
        else:
            ams = outputs['attn_maps']
            ams = [am[0].mean(0).detach().cpu().numpy() for am in ams]
            info['attn'].extend(list(ams))
        # evaluate instances
        info.update(self.evaluate_instances(info, attn_overlay_mode=self.attn_overlay_mode))
        return_dict['info'] = info
        if path is not None and save_full_data:
            # add folder if it doesn't exist
            if not os.path.exists(path):
                os.mkdir(path)
            # save anything that doesn't go in the dataframe
            self.save_folder_files(info, path)
        # save generated plots
        if plot:
            return_dict['figs'] = self.plot_info(info, path=path, attn_overlay_mode=self.attn_overlay_mode)
        # create and save dataframe
        df = self.info_to_df(info)
        if df_from_file is not None:
            df = pd.concat([df_from_file, df])
        if path is not None:
            df.to_csv(os.path.join(path, 'sentences.csv'), index=False)
        return_dict['df'] = df
        return return_dict

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if pl_module.global_step % self.log_train_every != 0:
            return
        with torch.no_grad():
            # outputs = get_train_outputs(outputs)
            # because crop is random during train, we need to redo the forward pass with deterministic crop
            self.gloria_collate_fn.device = pl_module.device
            return_dict = self.evaluate_and_save(batch=batch, pl_module=pl_module)
            df = return_dict['df']
            logger = pl_module.logger.experiment
            if (~df.auroc.isnull()).sum() > 0:
                logger.log({"train/auroc_step": df.auroc[~df.auroc.isnull()].mean(),
                            "global_step": trainer.global_step})
            if (~df.avg_precision.isnull()).sum() > 0:
                logger.log({"train/avg_precision_step": df.avg_precision[~df.avg_precision.isnull()].mean(),
                            "global_step": trainer.global_step})
            if (~df.attn_entropy.isnull()).sum() > 0:
                logger.log({"train/attn_entropy_step": df.attn_entropy[~df.attn_entropy.isnull()].mean(),
                            "global_step": trainer.global_step})
            if (~df.no_attn_weight.isnull()).sum() > 0:
                logger.log({"train/no_attn_weight_step": df.no_attn_weight[~df.no_attn_weight.isnull()].mean(),
                            "global_step": trainer.global_step})

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        path = os.path.join(self.save_dir, 'val_outputs_%i' % pl_module.current_epoch) \
            if self.save_dir is not None else None
        self.gloria_collate_fn.device = pl_module.device
        self.evaluate_and_save(path=path, batch=batch, outputs=outputs, pl_module=pl_module)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        path = os.path.join(self.save_dir, 'test_outputs_%i' % pl_module.current_epoch) \
            if self.save_dir is not None else None
        self.gloria_collate_fn.device = pl_module.device
        self.evaluate_and_save(path=path, batch=batch, outputs=outputs, pl_module=pl_module, save_full_data=True)

    def shared_epoch_end(self, trainer, pl_module, epoch_type):
        path = os.path.join(self.save_dir, '%s_outputs_%i' % (epoch_type, pl_module.current_epoch)) \
            if self.save_dir is not None else None
        if path is not None:
            df = pd.read_csv(os.path.join(path, 'sentences.csv'))
            logger = pl_module.logger.experiment
            if (~df.auroc.isnull()).sum() > 0:
                logger.log({"%s/auroc_step" % epoch_type: df.auroc[~df.auroc.isnull()].mean(),
                            "global_step": trainer.global_step})
            if (~df.avg_precision.isnull()).sum() > 0:
                logger.log({"%s/avg_precision_step" % epoch_type: df.avg_precision[~df.avg_precision.isnull()].mean(),
                            "global_step": trainer.global_step})
            if (~df.attn_entropy.isnull()).sum() > 0:
                logger.log({"%s/attn_entropy_step" % epoch_type: df.attn_entropy[~df.attn_entropy.isnull()].mean(),
                            "global_step": trainer.global_step})
            if (~df.no_attn_weight.isnull()).sum() > 0:
                logger.log({"%s/no_attn_weight_step" % epoch_type: df.no_attn_weight[~df.no_attn_weight.isnull()].mean(),
                            "global_step": trainer.global_step})

    def on_validation_epoch_end(self, trainer, pl_module):
        self.shared_epoch_end(trainer, pl_module, 'val')

    def on_test_epoch_end(self, trainer, pl_module):
        self.shared_epoch_end(trainer, pl_module, 'test')


class WeightInstancesByLocalization(Callback):
    def __init__(self, dm, weight_mode='entropy', temp=1.):
        self.dm = dm
        self.weight_mode = weight_mode
        assert self.weight_mode in {'entropy'}
        self.temp = temp
        self.weight_mask = torch.zeros(len(self.dm.train), dtype=torch.bool)
        self.train_weights = torch.ones(len(self.dm.train))

    def get_weight_metric(self, outputs):
        ams = outputs['attn_maps']
        ams = torch.stack([am[0].mean(0).detach().cpu() for am in ams])
        if self.weight_mode == 'entropy':
            return discrete_entropy(ams.reshape(ams.shape[0], -1))
        else:
            raise Exception

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        with torch.no_grad():
            outputs = get_train_outputs(outputs)
            indices = []
            for instance in batch['instances']:
                patient_id = next(iter(instance.keys()))
                study_id = next(iter(instance[patient_id].keys()))
                instance = instance[patient_id][study_id]
                indices.append(instance['index'])
            indices = torch.tensor(indices)
            self.weight_mask[indices] = True
            weights = self.get_weight_metric(outputs)
            self.train_weights[indices] = weights

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        # make sure all unset weights are set to the average of the set weights
        mean = self.train_weights[self.weight_mask].mean()
        self.train_weights[~self.weight_mask] = mean
        train_weights_softmax = torch.softmax(self.train_weights * self.temp, 0)
        # log entropy or normalized entropy of distribution
        self.dm.weight_instances(train_weights_softmax)
        logger = pl_module.logger.experiment
        logger.log({"train/weights_mean": mean,
                    "train/weights_hist": wandb.Histogram(self.train_weights.numpy()),
                    "train/weights_softmax_hist": wandb.Histogram(train_weights_softmax.numpy()),
                    "global_step": trainer.global_step})
