import pytorch_lightning as pl
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from . import image_dataset
from . import pretraining_dataset
from .. import builder


class PretrainingDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.dataset = pretraining_dataset.MultimodalPretrainingDataset
        self.collate_fn = pretraining_dataset.multimodal_collate_fn

    def train_dataloader(self):
        transform = builder.build_transformation(self.cfg, "train")
        dataset = self.dataset(self.cfg, split="train", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="test", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )


from .mimic_data import ImaGenomeDataModule as IGDM, MimicCxrFiler, ImaGenomeFiler
from .mimic_for_gloria import GloriaCollateFn
import os
from getpass import getpass
class ImaGenomeDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        # TODO: use more of cfg
        username = None if not cfg.data.get_physio_creds else input('physio username: ')
        password = None if not cfg.data.get_physio_creds else getpass('physio password: ')
        mimic_cxr_filer = MimicCxrFiler(
            download_directory=cfg.data.mimic_cxr_download_directory, physio_username=username, physio_password=password)
        imagenome_filer = ImaGenomeFiler(
            download_directory=cfg.data.imagenome_download_directory, physio_username=mimic_cxr_filer.username,
            physio_password=mimic_cxr_filer.password)

        collate_fn = (
            GloriaCollateFn(cfg, 'train'),
            GloriaCollateFn(cfg, 'test'),
            GloriaCollateFn(cfg, 'test')
        )

        self.dm = IGDM(
            mimic_cxr_filer, imagenome_filer, batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers, collate_fn=collate_fn,
            get_images=True, get_reports=True, force=False, parallel=self.cfg.data.parallel,
            num_preprocessing_workers=os.cpu_count(), chunksize=1,
            split_slices=self.cfg.data.split_slices, gold_test=self.cfg.data.gold_test, randomize_reports=cfg.data.randomize_reports,
            randomize_objects_mode=self.cfg.data.randomize_objects_mode,
            group_by=self.cfg.data.group_by, drop_last=True, pin_memory=True)
        self.prepare_data()

    def prepare_data(self):
        self.dm.prepare_data()

    def train_dataloader(self):
        return self.dm.train_dataloader()

    def val_dataloader(self):
        return self.dm.val_dataloader()

    def test_dataloader(self):
        return self.dm.test_dataloader()


class CheXpertDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.dataset = image_dataset.CheXpertImageDataset

    def train_dataloader(self):
        transform = builder.build_transformation(self.cfg, "train")
        dataset = self.dataset(self.cfg, split="train", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def val_dataloader(self):
        transform = builder.build_transformation(self.cfg, "valid")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="test", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )


class PneumothoraxDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.dataset = image_dataset.PneumothoraxImageDataset

    def train_dataloader(self):
        transform = builder.build_transformation(self.cfg, "train")
        dataset = self.dataset(self.cfg, split="train", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def val_dataloader(self):
        transform = builder.build_transformation(self.cfg, "valid")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="test", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )


class PneumoniaDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.dataset = image_dataset.PneumoniaImageDataset

        if cfg.phase == "detection":
            self.collate_fn = image_dataset.detection_collate_fn
        else:
            self.collate_fn = None

    def train_dataloader(self):
        transform = builder.build_transformation(self.cfg, "train")
        dataset = self.dataset(self.cfg, split="train", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def val_dataloader(self):
        transform = builder.build_transformation(self.cfg, "valid")
        dataset = self.dataset(self.cfg, split="valid", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="test", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )
