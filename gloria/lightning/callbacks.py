from pytorch_lightning.callbacks.base import Callback
from torchmetrics import Metric
from torchmetrics.functional import roc, precision_recall_curve, auroc, average_precision, precision_recall, f1
from torch.distributions.categorical import Categorical
import copy
from gloria.datasets.mimic_for_gloria import GloriaCollateFn, normalize, original_tensor_to_numpy_image
from gloria.datasets.visualization_utils import *
import torch
import os
from torch import nn
import numpy as np
import pandas as pd
import wandb


def discrete_entropy(dist):
    # note if no-attn is not turned on, this makes no difference because 1 - dist.sum() = 0
    dist = torch.cat([get_no_attn_weight(dist).unsqueeze(0), dist], 0)
    return Categorical(dist).entropy()


def get_no_attn_weight(dist):
    return 1 - dist.sum(-1)


class Metrics:
    def __init__(self, percentile_thresholds=[.1, .2, .3]):
        self.attn_bbox_metrics = {
            'roc_curve': roc,
            'pr_curve': precision_recall_curve,
            'auroc': auroc,
            'avg_precision': average_precision,
        }
        self.percentile_thresholds = percentile_thresholds
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
        total = np.prod(segmentation_label.shape)
        targets = segmentation_label.reshape(-1).long()
        for p in self.percentile_thresholds:
            if segmentation_label.sum() > 0:
                top_k = int(total * p)
                preds = attn_overlay.reshape(-1)
                threshold = torch.topk(preds, total - top_k, largest=False).values.max()
                pr, re = precision_recall(preds, targets, threshold=threshold)
                f = f1(preds, targets, threshold=threshold)
                metrics['precision_at_%f' % p] = pr
                metrics['recall_at_%f' % p] = re
                metrics['f1_at_%f' % p] = f
            else:
                metrics['precision_at_%f' % p] = None
                metrics['recall_at_%f' % p] = None
                metrics['f1_at_%f' % p] = None
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
    def __init__(self, gloria_collate_fn, save_dir=None, batch_size=None, eval_attn_overlay_mode='upsample',
                 plot_attn_overlay_mode='upsample', log_train_every=100, val_save_full_data=False,
                 percentile_thresholds=[.1, .2, .3]):
        super().__init__()
        self.gloria_collate_fn = gloria_collate_fn
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.eval_attn_overlay_mode = eval_attn_overlay_mode
        self.plot_attn_overlay_mode = plot_attn_overlay_mode
        self.log_train_every = log_train_every
        self.metrics = Metrics(percentile_thresholds=percentile_thresholds)
        self.shape_to_windows_cache = {}
        self.val_save_full_data = val_save_full_data

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
            'local_sims',
            'global_sims',
        ]
        for p in self.metrics.percentile_thresholds:
            columns.append('precision_at_%f' % p)
            columns.append('recall_at_%f' % p)
            columns.append('f1_at_%f' % p)
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
        assert mode in {'windows', 'pyramid', 'upsample'}
        if mode == 'windows':
            assert image_shape in self.shape_to_windows_cache.keys()
            windows = self.get_windows(image_shape)
            raise NotImplementedError
        elif mode == 'pyramid':
            new_attn = pyramid_attn_overlay(attn, image_shape)
        elif mode == 'upsample':
            new_attn = nn.Upsample(size=image_shape)(attn.reshape(1, 1, *attn.shape))[0, 0]
        return new_attn

    def plot_info(self, info, path=None, attn_overlay_mode='upsample'):
        attn_overlay_func = lambda attn, shape: self.get_attn_overlay(attn, shape, mode=attn_overlay_mode)
        return plot_info(attn_overlay_func, info, path=path)

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
                          plot=False, eval_attn_overlay_mode='upsample', plot_attn_overlay_mode='upsample'):
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
            batches = [
                self.gloria_collate_fn.get_batch(
                    info['original_image'][i:i+batch_size], info['sentence'][i:i+batch_size])
                for i in range(0, len(info['original_image']), batch_size)
            ]
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
        info['global_sims'] = []
        info['local_sims'] = []
        if outputs is None:
            assert pl_module is not None
            for b in batches:
                b = pl_module.transfer_batch_to_device(b, pl_module.device)
                img_emb_l, img_emb_g, text_emb_l, text_emb_g, sents = pl_module.gloria(b)
                ams = pl_module.gloria.get_attn_maps(img_emb_l, text_emb_l, sents)
                ams = [am[0].mean(0).detach().cpu().numpy() for am in ams]
                info['attn'].extend(list(ams))
                local_sims = pl_module.gloria.get_local_similarities(img_emb_l, text_emb_l, b['cap_lens'])
                info['local_sims'].extend(torch.diagonal(local_sims).tolist())
                global_sims = pl_module.gloria.get_global_similarities(img_emb_g, text_emb_g)
                info['global_sims'].extend(torch.diagonal(global_sims).tolist())
        else:
            assert batch is not None
            ams = outputs['attn_maps'].obj
            ams = [am[0].mean(0).detach().cpu().numpy() for am in ams]
            info['attn'].extend(list(ams))
            dtype = outputs['img_emb_l'].obj.dtype
            img_emb_l, text_emb_l = outputs['img_emb_l'].obj, outputs['text_emb_l'].obj
            img_emb_g, text_emb_g = outputs['img_emb_g'].obj, outputs['text_emb_g'].obj
            local_sims = pl_module.gloria.get_local_similarities(img_emb_l, text_emb_l, batch['cap_lens'])
            info['local_sims'].extend(torch.diagonal(local_sims).tolist())
            global_sims = pl_module.gloria.get_global_similarities(img_emb_g, text_emb_g)
            info['global_sims'].extend(torch.diagonal(global_sims).tolist())
        # evaluate instances
        info.update(self.evaluate_instances(info, attn_overlay_mode=eval_attn_overlay_mode))
        return_dict['info'] = info
        if path is not None and save_full_data:
            # add folder if it doesn't exist
            if not os.path.exists(path):
                os.mkdir(path)
            # save anything that doesn't go in the dataframe
            self.save_folder_files(info, path)
        # save generated plots
        if plot:
            return_dict['figs'] = self.plot_info(info, path=path, attn_overlay_mode=plot_attn_overlay_mode)
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
        with torch.no_grad(), torch.cuda.amp.autocast():
            # outputs = get_train_outputs(outputs)
            # because crop is random during train, we need to redo the forward pass with deterministic crop
            self.gloria_collate_fn.device = pl_module.device
            return_dict = self.evaluate_and_save(
                batch=batch, pl_module=pl_module,
                eval_attn_overlay_mode=self.eval_attn_overlay_mode,
                plot_attn_overlay_mode=self.plot_attn_overlay_mode)
            df = return_dict['df']
            if pl_module.logger is not None:
                logger = pl_module.logger.experiment
                metrics = ['auroc', 'avg_precision', 'attn_entropy', 'no_attn_weight']
                metrics += ['precision_at_%f' % p for p in self.metrics.percentile_thresholds]
                metrics += ['recall_at_%f' % p for p in self.metrics.percentile_thresholds]
                metrics += ['f1_at_%f' % p for p in self.metrics.percentile_thresholds]
                for metric in metrics:
                    if (~df[metric].isnull()).sum() > 0:
                        logger.log({"train/%s_step" % metric: df[metric][~df[metric].isnull()].mean()},
                                   step=trainer.global_step)

    def shared_on_batch_start(self, batch, batch_type):
        #if self.save_dir is None:
        #    return
        #new_batch = {}
        #path = os.path.join(self.save_dir, '%s_outputs_%i' % (batch_type, pl_module.current_epoch))
        #if not os.path.exists(os.path.join(path, 'sentences.csv')):
        #    return
        #df = pd.read_csv(os.path.join(path, 'sentences.csv'))
        #dicom_sent_ids = set(df.dicom_sent_id)
        #indices = []
        #for i, instance in enumerate(batch['instances']):
        #    
        #        if (instance) not in :
        #            indices.append(i)
        #indices = torch.tensor(indices)
        # TODO: update batch
        pass

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        self.shared_on_batch_start(batch, "val")

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        self.shared_on_batch_start(batch, "test")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        path = os.path.join(self.save_dir, 'val_outputs_%i' % pl_module.current_epoch) \
            if self.save_dir is not None else None
        self.gloria_collate_fn.device = pl_module.device
        with torch.no_grad(), torch.cuda.amp.autocast():
            self.evaluate_and_save(
                path=path, batch=batch, outputs=outputs, pl_module=pl_module,
                save_full_data=self.val_save_full_data,
                eval_attn_overlay_mode=self.eval_attn_overlay_mode,
                plot_attn_overlay_mode=self.plot_attn_overlay_mode)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        path = os.path.join(self.save_dir, 'test_outputs_%i' % pl_module.current_epoch) \
            if self.save_dir is not None else None
        self.gloria_collate_fn.device = pl_module.device
        with torch.no_grad(), torch.cuda.amp.autocast():
            self.evaluate_and_save(
                path=path, batch=batch, outputs=outputs, pl_module=pl_module, save_full_data=True,
                eval_attn_overlay_mode=self.eval_attn_overlay_mode,
                plot_attn_overlay_mode=self.plot_attn_overlay_mode)

    def shared_epoch_end(self, trainer, pl_module, epoch_type):
        path = os.path.join(self.save_dir, '%s_outputs_%i' % (epoch_type, pl_module.current_epoch)) \
            if self.save_dir is not None else None
        if path is not None:
            df = pd.read_csv(os.path.join(path, 'sentences.csv'))
            if pl_module.logger is not None:
                logger = pl_module.logger.experiment
                metrics = ['auroc', 'avg_precision', 'attn_entropy', 'no_attn_weight']
                metrics += ['precision_at_%f' % p for p in self.metrics.percentile_thresholds]
                metrics += ['recall_at_%f' % p for p in self.metrics.percentile_thresholds]
                metrics += ['f1_at_%f' % p for p in self.metrics.percentile_thresholds]
                for metric in metrics:
                    if (~df[metric].isnull()).sum() > 0:
                        logger.log({"%s/%s_epoch" % (epoch_type, metric): df[metric][~df[metric].isnull()].mean()},
                                   step=trainer.global_step)

    def on_validation_epoch_end(self, trainer, pl_module):
        self.shared_epoch_end(trainer, pl_module, 'val')

    def on_test_epoch_end(self, trainer, pl_module):
        self.shared_epoch_end(trainer, pl_module, 'test')


class WeightInstancesByLocalization(Callback):
    def __init__(self, dm, weight_mode='entropy', temp=1.):
        self.dm = dm
        self.weight_mode = weight_mode
        assert self.weight_mode in {'entropy', 'no_attn_score'}
        self.temp = temp
        self.weight_mask = torch.zeros(len(self.dm.train), dtype=torch.bool)
        self.train_weights = torch.ones(len(self.dm.train))

    def get_weight_metric(self, outputs):
        ams = outputs['attn_maps'].obj
        ams = torch.stack([am[0].mean(0).detach().cpu() for am in ams])
        if self.weight_mode == 'entropy':
            return discrete_entropy(ams.reshape(ams.shape[0], -1))
        if self.weight_mode == 'no_attn_score':
            return -get_no_attn_weight(ams.reshape(ams.shape[0], -1))
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
        if pl_module.logger is not None:
            logger = pl_module.logger.experiment
            logger.log({"train/weights_mean": mean,
                        "train/weights_hist": wandb.Histogram(self.train_weights.numpy()),
                        "train/weights_softmax_hist": wandb.Histogram(train_weights_softmax.numpy()),
                        "global_step": trainer.global_step})
