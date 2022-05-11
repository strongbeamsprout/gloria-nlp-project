from transformers import AutoTokenizer
import cv2
from PIL import Image
from gloria import builder
import torch
import numpy as np
import re
from nltk.tokenize import RegexpTokenizer
import random
#from gloria.datasets.visualization_utils import sent_bboxes_to_segmentation_label, mask_to_bbox


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


def sent_bboxes_to_segmentation_label(shape, sent_bboxes):
    segmentation_label = torch.zeros(shape, dtype=torch.bool)
    for bbox in sent_bboxes:
        segmentation_label = segmentation_label | bbox_to_mask(bbox, shape)
    return segmentation_label


def normalize(image):
    image = image.float()
    return ((image - image.min()) / (image.max() - image.min()))


def original_tensor_to_numpy_image(image):
    return np.array(normalize(image) * 255, dtype=np.uint8)


def process_bboxes(image_shapes, bboxes, gloria_collate_fn):
    new_bboxes = []
    box_masks = []
    for shape, bbox in zip(image_shapes, bboxes):
        box_mask = bbox_to_mask(bbox, shape)
        box_masks.append(original_tensor_to_numpy_image(box_mask))
    new_box_masks = gloria_collate_fn.process_img(box_masks, 'cpu')
    new_box_masks = new_box_masks > 0
    new_bboxes = [mask_to_bbox(new_box_mask[0]) for new_box_mask in new_box_masks]
    return new_bboxes


class GloriaCollateFn:
    def __init__(self, cfg, split, device='cpu', include_instances=True):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)
        self.transform = builder.build_transformation(self.cfg, split)
        self.split = split
        self.device = device
        self.ixtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}
        self.include_instances = include_instances

    def __call__(self, instances):
        imgs, cap_len, ids, tokens, attention, path = [], [], [], [], [], []
        # flattern
        captions, images, indices, bboxes = [], [], [], []
        for instance in instances:
            patient_id = next(iter(instance.keys()))
            study_id = next(iter(instance[patient_id].keys()))
            instance = instance[patient_id][study_id]
            dicom_id = next(iter(instance['images'].keys()))
            x = original_tensor_to_numpy_image(instance['images'][dicom_id])
            images.append(x)
            if 'sentence' in instance.keys():
                captions.append(instance['sentence'])
                sent_info = instance['objects'][dicom_id]['sent_to_bboxes'][instance['sent_id']]
                bboxes.append(sent_info['coords_original'])
            else:
                captions.append(instance['report'])
        return self.get_batch(images, captions, instances=instances if self.include_instances else None, bboxes=bboxes if len(bboxes) > 0 else None)

    def get_batch(self, images, captions, instances=None, sort=True, bboxes=None):
        imgs = self.process_img(images, self.device)
        if bboxes is not None:
            original_shapes = [img.shape for img in images]
            new_shape = imgs[0, 0].shape
            seg_labels = self.get_segmentation_labels(bboxes, original_shapes, new_shape, self.device)
        cap_return_dict = self.process_text(captions, self.device)

        # sort and add to dictionary
        if sort:
            sorted_cap_lens, sorted_cap_indices = torch.sort(torch.tensor(cap_return_dict["cap_lens"]), 0, True)
        else:
            sorted_cap_lens = torch.tensor(cap_return_dict["cap_lens"])
            sorted_cap_indices = torch.arange(len(sorted_cap_lens))
        sorted_cap_lens, sorted_cap_indices = sorted_cap_lens.to(self.device), sorted_cap_indices.to(self.device)
        return_dict = {k: v[sorted_cap_indices] for k, v in cap_return_dict.items() if k != "cap_lens"}
        return_dict["cap_lens"] = sorted_cap_lens
        return_dict["imgs"] = imgs[sorted_cap_indices]
        if bboxes is not None:
            return_dict["segmentation_labels"] = seg_labels[sorted_cap_indices]
        if instances is not None:
            return_dict['instances'] = [instances[i] for i in sorted_cap_indices]
        return return_dict

    def get_segmentation_labels(self, bboxes, original_shapes, new_shape, device):
        seg_labels = []
        for bbs, original_shape in zip(bboxes, original_shapes):
            new_bbs = process_bboxes([original_shape for bb in bbs], bbs, self)
            segmentation_label = sent_bboxes_to_segmentation_label(new_shape, new_bbs)
            seg_labels.append(segmentation_label)
        seg_labels = torch.stack(seg_labels, 0)
        return seg_labels.to(device)

    # almost completely copied from gloria method
    def process_img(self, images, device):

        all_imgs = []
        for x in images:
            # tranform images
            x = self._resize_img(x, self.cfg.data.image.imsize)
            img = Image.fromarray(x).convert("RGB")
            img = self.transform(img)
            all_imgs.append(img)

        all_imgs = torch.stack(all_imgs).to(device)

        return all_imgs

    # almost completely copied from gloria method
    def _resize_img(self, img, scale):
        """
        Args:
            img - image as numpy array (cv2)
            scale - desired output image-size as scale x scale
        Return:
            image resized to scale x scale with shortest dimension 0-padded
        """
        size = img.shape
        max_dim = max(size)
        max_ind = size.index(max_dim)

        # Resizing
        if max_ind == 0:
            # image is heigher
            wpercent = scale / float(size[0])
            hsize = int((float(size[1]) * float(wpercent)))
            desireable_size = (scale, hsize)
        else:
            # image is wider
            hpercent = scale / float(size[1])
            wsize = int((float(size[0]) * float(hpercent)))
            desireable_size = (wsize, scale)
        resized_img = cv2.resize(
            img, desireable_size[::-1], interpolation=cv2.INTER_AREA
        )  # this flips the desireable_size vector

        # Padding
        if max_ind == 0:
            # height fixed at scale, pad the width
            pad_size = scale - resized_img.shape[1]
            left = int(np.floor(pad_size / 2))
            right = int(np.ceil(pad_size / 2))
            top = int(0)
            bottom = int(0)
        else:
            # width fixed at scale, pad the height
            pad_size = scale - resized_img.shape[0]
            top = int(np.floor(pad_size / 2))
            bottom = int(np.ceil(pad_size / 2))
            left = int(0)
            right = int(0)
        resized_img = np.pad(
            resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
        )

        return resized_img

    # almost completely copied from gloria method
    def process_text(self, text, device, objects=None):
        if objects is not None:
            raise NotImplementedError

        processed_text_tensors = []
        for t in text:
            # use space instead of newline
            t = t.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(t)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            all_sents = []

            for t in captions:
                t = t.replace("\ufffd\ufffd", " ")
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(t.lower())

                if len(tokens) <= 1:
                    continue

                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                all_sents.append(" ".join(included_tokens))

            if self.cfg.data.text.full_report is True:
                t = " ".join(all_sents)
            else:
                sent_idx = random.randint(0, len(all_sents)-1)
                t = all_sents[sent_idx]

            text_tensors = self.tokenizer(
                t,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=self.cfg.data.text.word_num,
            )
            text_tensors["sent"] = [
                self.ixtoword[ix] for ix in text_tensors["input_ids"][0].tolist()
            ]
            processed_text_tensors.append(text_tensors)

        caption_ids = torch.stack([x["input_ids"] for x in processed_text_tensors])
        attention_mask = torch.stack(
            [x["attention_mask"] for x in processed_text_tensors]
        )
        token_type_ids = torch.stack(
            [x["token_type_ids"] for x in processed_text_tensors]
        )

        if len(text) == 1:
            caption_ids = caption_ids.squeeze(0).to(device)
            attention_mask = attention_mask.squeeze(0).to(device)
            token_type_ids = token_type_ids.squeeze(0).to(device)
        else:
            caption_ids = caption_ids.squeeze().to(device)
            attention_mask = attention_mask.squeeze().to(device)
            token_type_ids = token_type_ids.squeeze().to(device)

        #cap_lens = []
        #for txt in text:
        #    cap_lens.append(len([w for w in txt if not w.startswith("[")]))
        cap_lens = [
            len([w for w in x["sent"] if not w.startswith("[")]) + 1 for x in processed_text_tensors
        ]

        return {
            "caption_ids": caption_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "cap_lens": cap_lens,
        }
