import torch
import torch.nn as nn
import cv2
import re
import numpy as np
from sklearn import metrics

from PIL import Image
from .. import builder
from .. import loss
from .. import utils
from transformers import AutoTokenizer
from nltk.tokenize import RegexpTokenizer
from torch.distributions.categorical import Categorical


class PositionEmbeddings(nn.Module):
    def __init__(self, num_positions, hidden_size, num_spatial_dims=1):
        super().__init__()
        self.num_positions = num_positions
        self.hidden_size = hidden_size
        self.num_spatial_dims = num_spatial_dims
        self.image_position_embeddings = nn.Embedding(num_positions, hidden_size // num_spatial_dims)

    def forward(self, spatial_shape):
        if isinstance(spatial_shape, int):
            spatial_shape = (spatial_shape,) * self.num_spatial_dims
        for d in spatial_shape:
            assert d <= self.num_positions
        device = next(iter(self.image_position_embeddings.parameters())).device
        pos_embeddings = [self.image_position_embeddings(torch.arange(d, device=device)) for d in spatial_shape]
        pos_dim = pos_embeddings[0].shape[-1]
        pos_embeddings = [
            emb.reshape(
                *(1 if i != j else d for j, d in enumerate(spatial_shape)), pos_dim
            ).expand(*spatial_shape, pos_dim)
            for i, emb in enumerate(pos_embeddings)]
        padding = torch.zeros(
            *spatial_shape, self.hidden_size - len(spatial_shape) * pos_dim,
            device=device)
        positions = torch.cat(pos_embeddings + [padding], -1)
        return positions


class GLoRIA(nn.Module):
    def __init__(self, cfg):
        super(GLoRIA, self).__init__()

        self.cfg = cfg
        self.text_encoder = builder.build_text_model(cfg)
        self.img_encoder = builder.build_img_model(cfg)
        self.position_embeddings = PositionEmbeddings(
            self.cfg.model.image_position_embeddings.num, self.cfg.model.text.embedding_dim, num_spatial_dims=2) \
            if self.cfg.model.image_position_embeddings is not None else None
        self.image_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                self.cfg.model.text.embedding_dim, self.cfg.model.image_transformer.num_heads),
            self.cfg.model.image_transformer.num_layers) \
            if "image_transformer" in self.cfg.model.keys() else None
        self.no_attn_vec = nn.Parameter(torch.randn(self.cfg.model.text.embedding_dim)) \
            if self.cfg.model.gloria.no_attn_vec else None

        self.local_loss = loss.gloria_loss.local_loss
        self.global_loss = loss.gloria_loss.global_loss
        self.local_loss_weight = self.cfg.model.gloria.local_loss_weight
        self.global_loss_weight = self.cfg.model.gloria.global_loss_weight
        self.sparse_attn_loss_weight = self.cfg.model.gloria.sparse_attn_loss_weight
        self.no_attn_loss_weight = self.cfg.model.gloria.no_attn_loss_weight
        self.attention_divergence_loss_weight = self.cfg.model.gloria.attention_divergence_loss_weight
        self.attention_entropy_loss_weight = self.cfg.model.gloria.attention_entropy_loss_weight
        self.segmentation_loss_weight = self.cfg.model.gloria.segmentation_loss_weight

        self.temp1 = self.cfg.model.gloria.temp1
        self.temp2 = self.cfg.model.gloria.temp2
        self.temp3 = self.cfg.model.gloria.temp3
        self.batch_size = self.cfg.train.batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)
        self.ixtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}

    def text_encoder_forward(self, caption_ids, attention_mask, token_type_ids):
        text_emb_l, text_emb_g, sents = self.text_encoder(
            caption_ids, attention_mask, token_type_ids
        )
        return text_emb_l, text_emb_g, sents

    def image_encoder_forward(self, imgs):
        img_feat_g, img_emb_l = self.img_encoder(imgs, get_local=True)
        img_emb_g, img_emb_l = self.img_encoder.generate_embeddings(
            img_feat_g, img_emb_l
        )
        
        b, c, h, w = img_emb_l.shape
        if self.position_embeddings is not None:
            pos_embeddings = self.position_embeddings((h, w))
            pos_embeddings = pos_embeddings.permute(2, 0, 1).expand(b, c, h, w)
            img_emb_l = img_emb_l + pos_embeddings
        if self.image_transformer is not None:
            img_emb_l_flattened = img_emb_l.reshape(b, c, h * w).permute(2, 0, 1)
            img_emb_l_flattened = self.image_transformer(img_emb_l_flattened)
            img_emb_l = img_emb_l_flattened.permute(1, 2, 0).reshape(b, c, h, w)
        
        return img_emb_l, img_emb_g

    def _calc_local_loss(self, img_emb_l, text_emb_l, sents):

        cap_lens = [
            len([w for w in sent if not w.startswith("[")]) + 1 for sent in sents
        ]
        l_loss0, l_loss1, no_attn_loss, kl_loss, entropy_loss, attn_maps = self.local_loss(
            img_emb_l,
            text_emb_l,
            cap_lens,
            temp1=self.temp1,
            temp2=self.temp2,
            temp3=self.temp3,
            no_attn_vec=self.no_attn_vec,
            no_attn_loss_weight=self.no_attn_loss_weight,
            attention_divergence_loss_weight=self.attention_divergence_loss_weight,
            attention_entropy_loss_weight=self.attention_entropy_loss_weight
        )

        return l_loss0, l_loss1, no_attn_loss, kl_loss, entropy_loss, attn_maps

    def _calc_global_loss(self, img_emb_g, text_emb_g):
        g_loss0, g_loss1 = self.global_loss(img_emb_g, text_emb_g, temp3=self.temp3)
        return g_loss0, g_loss1

    def _calc_attn_loss(attn_maps, attn_labels):
        raise NotImplementedError

    def calc_loss(self, img_emb_l, img_emb_g, text_emb_l, text_emb_g, sents, segmentation_labels=None):
        # weighted loss
        loss = 0
        l_loss0, l_loss1, no_attn_loss, kl_loss, entropy_loss, attn_maps = self._calc_local_loss(
            img_emb_l, text_emb_l, sents,
        )
        if self.local_loss_weight != 0:
            loss += (l_loss0 + l_loss1) * self.local_loss_weight
        if self.global_loss_weight != 0:
            g_loss0, g_loss1 = self._calc_global_loss(img_emb_g, text_emb_g)
            loss += (g_loss0 + g_loss1) * self.global_loss_weight
        if segmentation_labels is not None and self.segmentation_loss_weight:
            mean_attn_maps = torch.cat([attn_map.mean(1) for attn_map in attn_maps], 0)
            mean_upsampled_attn_maps = nn.functional.interpolate(mean_attn_maps.unsqueeze(1), size=segmentation_labels.shape[1:]).squeeze(1)
            mean_upsampled_attn_maps = mean_upsampled_attn_maps / mean_upsampled_attn_maps.sum(-1, keepdims=True).sum(-2, keepdims=True)
            loss += -torch.log((segmentation_labels * mean_upsampled_attn_maps).sum(-1).sum(-1)).mean() * self.segmentation_loss_weight
        loss += no_attn_loss + kl_loss + entropy_loss

        return loss, attn_maps

    def forward(self, x):

        # img encoder branch
        img_emb_l, img_emb_g = self.image_encoder_forward(x["imgs"])

        # text encorder branch
        text_emb_l, text_emb_g, sents = self.text_encoder_forward(
            x["caption_ids"], x["attention_mask"], x["token_type_ids"]
        )

        return img_emb_l, img_emb_g, text_emb_l, text_emb_g, sents

    def get_global_similarities(self, img_emb_g, text_emb_g):
        img_emb_g = img_emb_g.detach().cpu().numpy()
        text_emb_g = text_emb_g.detach().cpu().numpy()
        global_similarities = metrics.pairwise.cosine_similarity(img_emb_g, text_emb_g)
        global_similarities = torch.Tensor(global_similarities)
        return global_similarities

    def get_local_similarities(self, img_emb_l, text_emb_l, cap_lens):

        batch_size = img_emb_l.shape[0]
        similarities = []

        for i in range(len(text_emb_l)):
            words_num = cap_lens[i]
            word = (
                text_emb_l[i, :, 1 : words_num + 1].unsqueeze(0).contiguous()
            )  # [1, 768, 25]

            word = word.repeat(batch_size, 1, 1)  # [48, 768, 25]
            context = img_emb_l  # [48, 768, 19, 19]

            weiContext, attn = loss.gloria_loss.attention_fn(
                word, context, 4.0, no_attn_vec=self.no_attn_vec
            )  # [48, 768, 25], [48, 25, 19, 19]

            word = word.transpose(1, 2).contiguous()  # [48, 25, 768]
            weiContext = weiContext.transpose(1, 2).contiguous()  # [48, 25, 768]

            word = word.view(batch_size * words_num, -1)  # [1200, 768]
            weiContext = weiContext.view(batch_size * words_num, -1)  # [1200, 768]
            #
            row_sim = loss.gloria_loss.cosine_similarity(word, weiContext)
            row_sim = row_sim.view(batch_size, words_num)  # [48, 25]

            row_sim.mul_(5.0).exp_()
            row_sim, max_row_idx = torch.max(row_sim, dim=1, keepdim=True)  # [48, 1]

            row_sim = torch.log(row_sim)

            similarities.append(row_sim)

        local_similarities = torch.cat(similarities, 1).detach().cpu()

        return local_similarities

    def get_attn_maps(self, img_emb_l, text_emb_l, sents):
        _, _, _, _, _, attn_maps = self._calc_local_loss(img_emb_l, text_emb_l, sents)
        return attn_maps

    def plot_attn_maps(self, attn_maps, imgs, sents, epoch_idx=0, batch_idx=0):

        img_set, _ = utils.build_attention_images(
            imgs,
            attn_maps,
            max_word_num=self.cfg.data.text.word_num,
            nvis=self.cfg.train.nvis,
            rand_vis=self.cfg.train.rand_vis,
            sentences=sents,
        )

        if img_set is not None:
            im = Image.fromarray(img_set)
            fullpath = (
                f"{self.cfg.output_dir}/"
                f"attention_maps_epoch{epoch_idx}_"
                f"{batch_idx}.png"
            )
            im.save(fullpath)

    def process_text(self, text, device):

        if type(text) == str:
            text = [text]

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

            t = " ".join(all_sents)

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

        cap_lens = []
        for txt in text:
            cap_lens.append(len([w for w in txt if not w.startswith("[")]))

        return {
            "caption_ids": caption_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "cap_lens": cap_lens,
        }

    def process_class_prompts(self, class_prompts, device):

        cls_2_processed_txt = {}
        for k, v in class_prompts.items():
            cls_2_processed_txt[k] = self.process_text(v, device)

        return cls_2_processed_txt

    def process_img(self, paths, device):

        transform = builder.build_transformation(self.cfg, split="test")

        if type(paths) == str:
            paths = [paths]

        all_imgs = []
        for p in paths:

            x = cv2.imread(str(p), 0)

            # tranform images
            x = self._resize_img(x, self.cfg.data.image.imsize)
            img = Image.fromarray(x).convert("RGB")
            img = transform(img)
            all_imgs.append(torch.tensor(img))

        all_imgs = torch.stack(all_imgs).to(device)

        return all_imgs

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
