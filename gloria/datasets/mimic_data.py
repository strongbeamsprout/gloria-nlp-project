import torch
from torch import default_generator
from torch.utils.data import Dataset, random_split, DataLoader, WeightedRandomSampler
from getpass import getpass
import subprocess
import pandas as pd
import pydicom
from tqdm import tqdm
import os
import numpy as np
import torchvision
from nibabel import nifti1
import pickle as pkl
tqdm.pandas()
import random
import multiprocessing as mp
import pytorch_lightning as pl
import zipfile
import json


def default_collate_fn(instances):
    return {k: [i[k] for i in instances] for k in instances[0].keys()}


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=1, num_workers=0, collate_fn=default_collate_fn, **kwargs):
        super().__init__()
        for i, split in enumerate(['train', 'val', 'test']):
            dataloader_kwargs = dict(
                batch_size=batch_size,
                num_workers=num_workers,
                collate_fn=collate_fn[i] if isinstance(collate_fn, tuple) else collate_fn
            )
            dataloader_kwargs.update(kwargs)
            setattr(self, '%s_dataloader_kwargs' % split, dataloader_kwargs)
        self.train_dataloader_kwargs['shuffle'] = True
        self._train = None
        self._val = None
        self._test = None

    def weight_instances(self, weights, split='train'):
        dataloader_kwargs = getattr(self, '%s_dataloader_kwargs' % split)
        dataloader_kwargs['sampler'] = WeightedRandomSampler(weights, len(weights))

    @property
    def train(self):
        if self._train is None:
            self.setup(stage='fit')
        return self._train

    @property
    def val(self):
        if self._val is None:
            self.setup(stage='fit')
        return self._val

    @property
    def test(self):
        if self._test is None:
            self.setup(stage='test')
        return self._test

    def train_dataloader(self):
        return DataLoader(self.train, **self.train_dataloader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.val, **self.val_dataloader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.test, **self.test_dataloader_kwargs)

    def get_train_examples(self, n):
        return [self.train[i] for i in range(n)]

    def get_val_examples(self, n):
        return [self.val[i] for i in range(n)]

    def get_test_examples(self, n):
        return [self.test[i] for i in range(n)]


class DownloadError(Exception):
    pass


def get_paths_that_start_with(path):
    split_path = path.split('/')
    files = []
    for file in os.listdir('/'.join(split_path[:-1])):
        if file.startswith(split_path[-1]):
            files.append('/'.join([*split_path[:-1], file]))
    return files


class MimicCxrFiler:
    """
    Handles downloading of reports and text and processing dicoms while downloading to avoid unnecessary memory
    consumption
    """
    def __init__(self, image_shape=None, download_directory=None, physio_username=None, physio_password=None):
        self.download_directory = download_directory \
            if download_directory is not None else os.path.join(os.getcwd(), 'mimic-cxr')
        if not os.path.exists(self.download_directory):
            os.mkdir(self.download_directory)
        self.full_download_directory = os.path.join(self.download_directory, 'physionet.org/files/mimic-cxr/2.0.0')
        self.base_call = ['wget', '-r', '-N', '-c', '-np']
        self.base_url = 'https://physionet.org/files/mimic-cxr/2.0.0'
        self.username = physio_username
        self.password = physio_password
        self.transform = torchvision.transforms.Resize(image_shape) if image_shape is not None else None
        self.image_shape = image_shape

    def download_file(self, relative_path='', force=False):
        url = os.path.join(self.base_url, relative_path)
        path = os.path.join(self.full_download_directory, relative_path)
        if force or not os.path.exists(path):
            if self.username is None:
                self.username = input('physionet username: ')
            if self.password is None:
                self.password = getpass('physionet password: ')
            completed_process = subprocess.run(
                self.base_call + ['--user', self.username, '--password', self.password, url, '--no-check-certificate'],
                cwd=self.download_directory, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if completed_process.returncode != 0:
                print(completed_process.stdout.decode())
                print(completed_process.stderr.decode())
                print('path:', path)
                # print('deleting')
                # subprocess.run(['rm', path])
                raise DownloadError()

    def get_full_path(self, path):
        return os.path.join(self.full_download_directory, path)

    def get_report_path(self, row):
        return self.get_full_path('/'.join(row.path.split('/')[:-1]) + '.txt')

    def save_report(self, row):
        self.download_file('/'.join(row.path.split('/')[:-1]) + '.txt')

    def get_report(self, row):
        with open(self.get_report_path(row), 'r') as f:
            return f.read()

    def get_report_labels_path(self, row):
        return self.get_full_path('/'.join(row.path.split('/')[:-1]) + '_labels.pkl')

    def save_report_labels(self, row, labels):
        with open(self.get_report_labels_path(row), 'wb') as f:
            pkl.dump(labels, f)

    def get_report_labels(self, row):
        with open(self.get_report_labels_path(row), 'rb') as f:
            return pkl.load(f)

    def get_dicom_path(self, row):
        return self.get_full_path(row.path)

    def save_dicom(self, row, force=False):
        self.download_file(row.path, force=force)

    def get_nifti_path(self, row):
        return '.'.join(self.get_dicom_path(row).split('.')[:-1]) + '.nii'

    def save_nifti_from_dicom(self, row, remove=True, force=False, process_function=None):
        if force or not os.path.exists(self.get_nifti_path(row)):
            self.save_dicom(row)
            try:
                image = pydicom.dcmread(self.get_dicom_path(row)).pixel_array
            except Exception as e:
                print('error with dicom path:', self.get_dicom_path(row))
                e.load_error_file_path = self.get_dicom_path(row)
                # print('deleting')
                # subprocess.run(['rm', self.get_dicom_path(row)])
                raise e
            if process_function is not None:
                image = process_function(image)
            niftiimage = nifti1.Nifti1Image(image, None)
            if remove and os.path.exists(self.get_dicom_path(row)):
                subprocess.call(['rm', self.get_dicom_path(row)])
            nifti1.save(niftiimage, self.get_nifti_path(row))

    def get_ptimage_path(self, row):
        return '.'.join(self.get_dicom_path(row).split('.')[:-1]) + '%s.pt' % \
               ('' if self.image_shape is None else ('_%ix%i' % self.image_shape))

    def remove_all_pts(self, row):
        for file in get_paths_that_start_with('.'.join(self.get_dicom_path(row).split('.')[:-1]) + '_'):
            subprocess.call(['rm', file])

    def save_ptimage_from_dicom(self, row, remove=True, force=False, process_function=None):
        if force or not os.path.exists(self.get_ptimage_path(row)):
            tries = 0
            while True:
                try:
                    tries += 1
                    self.save_dicom(row)
                    image = np.array(pydicom.dcmread(self.get_dicom_path(row)).pixel_array, dtype=np.int16)
                    break
                except Exception as e:
                    print('error with dicom path:', self.get_dicom_path(row))
                    e.load_error_file_path = self.get_dicom_path(row)
                    if isinstance(e, ValueError) and tries == 1:
                        print('deleting')
                        subprocess.run(['rm', self.get_dicom_path(row)])
                    else:
                        raise e
            if remove and os.path.exists(self.get_dicom_path(row)):
                subprocess.call(['rm', self.get_dicom_path(row)])
            self.save_ptimage(row, image, process_function=process_function)

    def save_ptimage_from_nifti(self, row, remove=True, remove_dicom=True, force=False, process_function=None):
        if force or not os.path.exists(self.get_ptimage_path(row)):
            self.save_nifti_from_dicom(row, remove=remove_dicom)
            try:
                image = np.array(nifti1.load(self.get_nifti_path(row)).dataobj)
            except Exception as e:
                print('error with nifti path:', self.get_nifti_path(row))
                e.load_error_file_path = self.get_nifti_path(row)
                # print('deleting')
                # subprocess.run(['rm', self.get_nifti_path(row)])
                raise e
            if remove and os.path.exists(self.get_nifti_path(row)):
                subprocess.call(['rm', self.get_nifti_path(row)])
            self.save_ptimage(row, image, process_function=process_function)

    def save_ptimage(self, row, image, process_function=None):
        if process_function is not None:
            image = process_function(image)
        image = torch.tensor(image.tolist())
        if self.transform is not None:
            image = self.transform(image.unsqueeze(0)).squeeze(0)
        torch.save(image, self.get_ptimage_path(row))

    def get_ptimage(self, row):
        try:
            return torch.load(self.get_ptimage_path(row))
        except Exception as e:
            print('error with pytorch image path:', self.get_ptimage_path(row))
            e.load_error_file_path = self.get_ptimage_path(row)
            # print('deleting')
            # subprocess.run(['rm', self.get_ptimage_path(row)])
            raise e

    def get_meta_path(self, row):
        return '.'.join(self.get_dicom_path(row).split('.')[:-1]) + '_meta.pkl'

    def get_meta(self, row):
        if not os.path.exists(self.get_meta_path(row)):
            self.save_dicom(row)
            dcm = pydicom.dcmread(self.get_dicom_path(row))
            viewpoint = dcm['00185101'][:]
            meta = {'viewpoint': viewpoint}
            with open(self.get_meta_path(row), 'wb') as f:
                pkl.dump(meta, f)
        else:
            with open(self.get_meta_path(row), 'rb') as f:
                meta = pkl.load(f)
        return meta


class ViewpointFilter:
    def __init__(self, filer, viewpoints):
        self.filer = filer
        self.viewpoints = viewpoints

    def __call__(self, row):
        if 'ViewPosition' in row.keys():
            return row['ViewPosition'] in self.viewpoints
        else:
            return self.filer.get_meta(row)['viewpoint'] in self.viewpoint


class HasGreaterThanNStudies:
    def __init__(self, filer, records, n=1):
        self.filer = filer
        self.counts = records[['subject_id', 'study_id']].drop_duplicates().groupby(['subject_id']).count()
        self.n = n

    def __call__(self, row):
        return self.counts.loc[row.subject_id].study_id > self.n


class ProcessDicoms:
    def __init__(self, filer, registration=None, records=None, to_nifti=True, to_pt=True, force_nifti=False,
                 force_pt=False, remove_dicom=True, remove_nifti=True):
        """
        if registration is not None, patients records are assumed to be processed in the order that they are in records
        """
        assert to_nifti or to_pt
        if registration is not None:
            assert to_nifti and records is not None
        self.filer = filer
        self.registration = registration
        self.records = records
        self.to_nifti = to_nifti
        self.to_pt = to_pt
        self.force_nifti = force_nifti
        self.force_pt = force_pt
        self.remove_dicom = remove_dicom
        self.remove_nifti = remove_nifti

    def __call__(self, row):
        if self.to_pt and not self.to_nifti:
            self.filer.save_ptimage_from_dicom(row, remove=self.remove_dicom)
        else:
            save_pt = self.to_pt and (self.force_pt or not os.path.exists(self.filer.get_ptimage_path(row)))
            save_nifti = self.force_nifti or (not os.path.exists(self.filer.get_nifti_path(row)) and
                                              (save_pt or not self.to_pt or (self.to_pt and not self.remove_nifti)))
            if save_nifti:
                patient_records = self.records[self.records.subject_id == row.subject_id]
                is_not_first = patient_records.iloc[0].name != row.name
                if self.registration is not None and is_not_first:
                    current_index = None
                    for i, (_, patient_row) in enumerate(patient_records.iterrows()):
                        if row.name == patient_row.name:
                            current_index = i
                            break
                    assert current_index is not None
                    baseline_record = patient_records.iloc[current_index - 1]
                    # baseline_record = patient_records.iloc[0]
                    self.filer.save_nifti_from_dicom(row, force=self.force_nifti, remove=self.remove_dicom)
                    try:
                        self.registration(
                            self.filer.get_nifti_path(baseline_record), self.filer.get_nifti_path(row),
                            output_volume=self.filer.get_nifti_path(row))
                    except Exception as e:
                        e.registration_error_row = row
                        raise e
                    if self.to_pt and self.remove_nifti and os.path.exists(self.filer.get_nifti_path(baseline_record)):
                        subprocess.run(['rm', self.filer.get_nifti_path(baseline_record)])
                else:
                    self.filer.save_nifti_from_dicom(row, force=self.force_nifti, remove=self.remove_dicom)
            if save_pt:
                remove = self.registration is None or \
                         self.records[self.records.subject_id == row.subject_id].iloc[-1].name == row.name
                self.filer.save_ptimage_from_nifti(row, remove=self.remove_nifti and remove, force=self.force_pt)


class SaveReport:
    def __init__(self, filer):
        self.filer = filer

    def __call__(self, row):
        self.filer.save_report(row)


def records_viewpoint_filter(records, filer, viewpoints, verbose=True):
    viewpoint_filter = ViewpointFilter(filer, viewpoints)
    if verbose:
        print('\nFilter dicoms so view position is \'%s\':' % str(viewpoints))
        return records.progress_apply(viewpoint_filter, axis=1)
    else:
        return records.apply(viewpoint_filter, axis=1)


def greater_than_n_studies_filter(records, filer, greater_than_n_studies, verbose=True):
    has_greater_than_n_studies = HasGreaterThanNStudies(filer, records, n=greater_than_n_studies)
    if verbose:
        print('\nFilter patients so the number of studies is greater than %s:' % greater_than_n_studies)
        return records.progress_apply(has_greater_than_n_studies, axis=1)
    else:
        return records.apply(has_greater_than_n_studies, axis=1)


def process_dicoms(records, filer, registration=None, to_nifti=True, to_pt=True, force_nifti=False, force_pt=False,
                   verbose=True):
    process_dicom = ProcessDicoms(
        filer, registration=registration, records=records, to_nifti=to_nifti, to_pt=to_pt,
        force_nifti=force_nifti, force_pt=force_pt)
    if verbose:
        print('\nSave dicoms to pytorch files:')
        records.progress_apply(process_dicom, axis=1)
    else:
        records.apply(process_dicom, axis=1)


def save_reports(records, filer, verbose=True):
    save_report = SaveReport(filer)
    if verbose:
        print('\nSave reports:')
        records.progress_apply(save_report, axis=1)
    else:
        records.apply(save_report, axis=1)
    return records


def process_records(records, filer, get_images=True, get_reports=True, registration=None, to_nifti=True, to_pt=True,
                    force_nifti=False, force_pt=False, greater_than_n_studies=0, chexpert_labeler=None, verbose=False,
                    viewpoints=['PA', 'AP']):
    """
    expects patient records will be kept together and in the right order
    """
    if get_images:
        viewpoint_filter = records_viewpoint_filter(records, filer, viewpoints, verbose=verbose)
        records[~viewpoint_filter].apply(lambda row: subprocess.run(['rm', filer.get_dicom_path(row)])
            if os.path.exists(filer.get_dicom_path(row)) else None, axis=1)
        records = records[viewpoint_filter]
        if len(records) == 0:
            return records
        if greater_than_n_studies > 0:
            has_greater_than_n_studies = greater_than_n_studies_filter(
                records, filer, greater_than_n_studies,verbose=verbose)
            records[~has_greater_than_n_studies].apply(
                lambda row: subprocess.run(['rm', filer.get_dicom_path(row)])
                if os.path.exists(filer.get_dicom_path(row)) else None, axis=1)
            records = records[has_greater_than_n_studies]
            if len(records) == 0:
                return records
        process_dicoms(
            records, filer, registration=registration, to_nifti=to_nifti, to_pt=to_pt, force_nifti=force_nifti,
            force_pt=force_pt, verbose=verbose)
    if get_reports:
        records = save_reports(records, filer, verbose=verbose)
        if chexpert_labeler is not None:
            chexpert_labeler(records, verbose=verbose)
    if verbose:
        print('')
    return records


def process_records_mapstyle(kwargs):
    return process_records(**kwargs)


class MimicCxr(Dataset):
    # Not intended for actual use as a dataset, outputs strings
    # Designed to be subclassed by or passed in to the actual dataset
    def __init__(self, df, filer, group_by='patient', randomize_reports=False):
        self.df = df
        self.group_by = group_by
        if self.group_by == 'patient':
            self.ids = sorted(list(set(self.df.subject_id)))
        elif self.group_by == 'study':
            self.ids = sorted(list(set(self.df.study_id)))
        elif self.group_by == 'image':
            self.ids = sorted(list(set(self.df.dicom_id)))
        else:
            raise Exception
        self.filer = filer
        self.randomize_reports = randomize_reports

    def __len__(self):
        return len(self.ids)
    
    def get_item_from_rows(self, rows):
        return_dict = {}
        for i, row in rows.iterrows():
            if row.subject_id not in return_dict.keys():
                return_dict[row.subject_id] = {}
            if row.study_id not in return_dict[row.subject_id].keys():
                return_dict[row.subject_id][row.study_id] = {'images': {}}
            return_dict[row.subject_id][row.study_id]['images'][row.dicom_id] = self.filer.get_ptimage(row)
            report_row = row if not self.randomize_reports else self.get_negative_row(row, same_study=False)
            return_dict[row.subject_id][row.study_id]['report'] = self.filer.get_report(report_row)
        return return_dict

    def __getitem__(self, item):
        id = self.ids[item]
        if self.group_by == 'patient':
            rows = self.df[self.df.subject_id == id]
        elif self.group_by == 'study':
            rows = self.df[self.df.study_id == id]
        elif self.group_by == 'image':
            rows = self.df[self.df.dicom_id == id]
        else:
            raise Exception
        return_dict = self.get_item_from_rows(rows)
        for k1, v1 in return_dict.items():
            for k2, v2 in v1.items():
                v2['index'] = item
        return return_dict

    def get_negative_row(self, row, same_study=None, same_patient=None):
        neg_rows = self.df[self.df.dicom_id != row.dicom_id]
        if same_study is not None:
            if same_study:
                assert same_patient is None
                temp_neg_rows = neg_rows[neg_rows.study_id == row.study_id]
                neg_rows = temp_neg_rows if len(temp_neg_rows) > 0 else neg_rows
            else:
                neg_rows = neg_rows[neg_rows.study_id != row.study_id]
        if same_patient is not None:
            if same_patient:
                temp_neg_rows = neg_rows[neg_rows.subject_id == row.subject_id]
                neg_rows = temp_neg_rows if len(temp_neg_rows) > 0 else neg_rows
            else:
                neg_rows = neg_rows[neg_rows.subject_id != row.subject_id]
        return neg_rows.sample().iloc[0]


class MimicCxrDataModule(BaseDataModule):
    def __init__(self, filer, get_images=True, get_reports=True, splits=(.8, .1), batch_size=1, num_workers=0,
                 dataslice=None, collate_fn=default_collate_fn, greater_than_n_studies=0, force=False, registration=None,
                 parallel=False, num_preprocessing_workers=os.cpu_count(), chunksize=1, chexpert_labeler=None,
                 randomize_reports=False, **kwargs):
        super().__init__(batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn, **kwargs)
        self.filer = filer
        self.get_images = get_images
        self.get_reports = get_reports
        self.splits = splits
        self.dataslice = dataslice
        self.greater_than_n_studies = greater_than_n_studies
        self.force = force
        self.registration = registration
        self.parallel = parallel
        self.num_preprocessing_workers = num_preprocessing_workers
        self.chunksize = chunksize
        self.chexpert_labeler = chexpert_labeler
        self.data_prepared = False
        self.randomize_reports = randomize_reports

    def get_kwargs(self, records):
        return dict(records=records, filer=self.filer, get_images=self.get_images, get_reports=self.get_reports,
                    registration=self.registration, to_nifti=self.registration is not None, to_pt=True,
                    force_nifti=self.force, force_pt=self.force, greater_than_n_studies=self.greater_than_n_studies,
                    chexpert_labeler=self.chexpert_labeler, verbose=False)

    def yield_args(self, records, subject_ids):
        for i, subject_id in enumerate(tqdm(subject_ids, total=len(subject_ids))):
            subject_records = records[records.subject_id == subject_id]
            kwargs = self.get_kwargs(subject_records)
            if i == 0:
                print('Setting one record\'s processing to verbose to serve as an example.')
                kwargs['verbose'] = True
            yield kwargs

    def prepare_data(self):
        if self.data_prepared:
            print('data already prepared')
            return
        self.filer.download_file('cxr-record-list.csv.gz')
        records = pd.read_csv(self.filer.get_full_path('cxr-record-list.csv.gz'), compression='gzip')
        # records = records[records.path.str.startswith('files/p10')]
        if self.dataslice is not None:
            records = records[self.dataslice]
        train_subjects, val_subjects = self.split_subjects(
            records, sum(self.splits), generator=torch.Generator().manual_seed(0))
        subject_ids = sorted(list(set(records.subject_id)))
        if not self.parallel:
            print('not parallelizing')
            # no parallel:
            records = pd.concat(
                [process_records(**kwargs) for kwargs in self.yield_args(records, subject_ids)])
        else:
            print('parallelizing')
            failed = True
            if self.filer.password is None:
                self.filer.password = ''
            while failed:
                try:
                    # parallel:
                    P = mp.Pool
                    with P(self.num_preprocessing_workers) as p:
                        print('starting processes:')
                        results_iterable = p.imap(
                            process_records_mapstyle,
                            list(self.yield_args(records, subject_ids)),
                            chunksize=self.chunksize
                        )
                        print('finishing processes:')
                        records = pd.concat([rs for rs in tqdm(results_iterable, total=len(subject_ids))])
                    failed = False
                except DownloadError:
                    self.filer.password = getpass('Please enter physio password: ')
        print('Total number of records:', len(records))
        train, test = self.split_dataframe_by_subjects(records, [train_subjects, val_subjects])
        print('Train:', len(train))
        print('Test:', len(test))
        train.to_csv(self.filer.get_full_path('train_records.csv.gz'), compression='gzip')
        test.to_csv(self.filer.get_full_path('test_records.csv.gz'), compression='gzip')
        self.data_prepared = True

    def get_dataset(self, records_df):
        return MimicCxr(records_df, self.filer, randomize_reports=self.randomize_reports)

    @staticmethod
    def split_subjects(records, frac, generator=default_generator):
        subject_ids = sorted(list(set(records.subject_id)))
        train_length = int(len(subject_ids) * frac)
        train_subjects, val_subjects = random_split(
            subject_ids, [train_length, len(subject_ids) - train_length], generator=generator)
        return train_subjects, val_subjects

    @staticmethod
    def split_dataframe_by_subjects(records, subject_lists):
        return [records[records.apply(lambda x: x.subject_id in subject_ids, axis=1)] for subject_ids in subject_lists]

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            records = pd.read_csv(self.filer.get_full_path('train_records.csv.gz'), compression='gzip')
            train, val = self.split_dataframe_by_subjects(records, self.split_subjects(
                records, self.splits[0], generator=torch.Generator().manual_seed(0)))
            self._train = self.get_dataset(train)
            self._val = self.get_dataset(val)
        if stage == 'test' or stage is None:
            self._test = self.get_dataset(pd.read_csv(self.filer.get_full_path('test_records.csv.gz'),
                                                      compression='gzip'))


class ImaGenomeFiler:
    def __init__(self, download_directory=None, physio_username=None, physio_password=None):
        self.download_directory = download_directory \
            if download_directory is not None else os.path.join(os.getcwd(), 'chest-imagenome')
        if not os.path.exists(self.download_directory):
            os.mkdir(self.download_directory)
        self.full_download_directory = os.path.join(self.download_directory, 'physionet.org/files/chest-imagenome/1.0.0')
        self.base_call = ['wget', '-r', '-N', '-c', '-np']
        self.base_url = 'https://physionet.org/files/chest-imagenome/1.0.0'
        self.username = physio_username
        self.password = physio_password

    def download_file(self, relative_path='', force=False, verbose=True):
        url = os.path.join(self.base_url, relative_path)
        path = os.path.join(self.full_download_directory, relative_path)
        if force or not os.path.exists(path):
            if verbose:
                print('downloading %s...' % url)
            if self.username is None:
                self.username = input('physionet username: ')
            if self.password is None:
                self.password = getpass('physionet password: ')
            if verbose:
                extra_kwargs = dict()
            else:
                extra_kwargs = dict(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            completed_process = subprocess.run(
                self.base_call + ['--user', self.username, '--password', self.password, url, '--no-check-certificate'],
                cwd=self.download_directory, **extra_kwargs)
            if completed_process.returncode != 0:
                if not verbose:
                    print(completed_process.stdout.decode())
                    print(completed_process.stderr.decode())
                print('path:', path)
                # print('deleting')
                # subprocess.run(['rm', path])
                raise DownloadError()
        if verbose:
            print('downloaded')

    def get_full_path(self, path):
        return os.path.join(self.full_download_directory, path)
    
    def unzip_file(self, relative_path, verbose=True):
        path = self.get_full_path(relative_path)
        target_path = path[:-4]
        if not os.path.exists(os.path.join(target_path, 'done')):
            if verbose:
                print('extracting %s...' % path)
            with zipfile.ZipFile(path) as zf:
                members = zf.infolist()
                if verbose:
                    members = tqdm(members, desc='Extracting ')
                for member in members:
                    try:
                        zf.extract(member)
                    except zipfile.error as e:
                        pass
            open(os.path.join(target_path, 'done'), 'w').close()
        if verbose:
            print('extracted')
#         completed_process = subprocess.run(['unzip', path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         if completed_process.returncode != 0:
#             print(completed_process.stdout.decode())
#             print(completed_process.stderr.decode())
#             print('path:', path)
#             raise Exception('unzipping failed')

    def get_split(self, split):
        if split in ['train', 'valid', 'test']:
            return pd.read_csv(self.get_full_path('silver_dataset/splits/%s.csv' % split))
        elif split == 'gold':
            return pd.read_csv(self.get_full_path('silver_dataset/splits/images_to_avoid.csv'))
        else:
            raise Exception

    def get_gold_file(self, file):
        if file.endswith('.txt'):
            return pd.read_csv(self.get_full_path(os.path.join('gold_dataset', file)), sep='\t')
        else:
            raise Exception

    def get_silver_scene_graph_json_file(self, dicom_id):
        return self.get_full_path('silver_dataset/scene_graph/%s_SceneGraph.json' % dicom_id)

    def get_silver_scene_graph_json(self, dicom_id):
        with open(self.get_silver_scene_graph_json_file(dicom_id), 'r') as f:
            return json.load(f)

    def get_objects_dir(self):
        return self.get_full_path('objects')

    def get_objects_file(self, dicom_id):
        return os.path.join(self.get_objects_dir(), dicom_id + '.pkl')

    def save_objects(self, obj, dicom_id):
        if not os.path.exists(self.get_objects_dir()):
            os.mkdir(self.get_objects_dir())
        with open(self.get_objects_file(dicom_id), 'wb') as f:
            pkl.dump(obj, f)

    def get_objects(self, dicom_id):
        with open(self.get_objects_file(dicom_id), 'rb') as f:
            return pkl.load(f)


def update_objects(objects, bbox, coord_original, sentence_id, sentence, label, context):
    if bbox not in objects['bbox_to_sents'].keys():
        objects['bbox_to_sents'][bbox] = {
            'coord_original': coord_original,
            'sentence_ids': [],
            'sentences': [],
            'labels': [],
            'contexts': [],
        }
    sent_info = objects['bbox_to_sents'][bbox]
    sent_info['sentence_ids'].append(sentence_id)
    sent_info['sentences'].append(sentence)
    sent_info['labels'].append(label)
    sent_info['contexts'].append(context)
    if sentence_id not in objects['sent_to_bboxes'].keys():
        objects['sent_to_bboxes'][sentence_id] = {
            'sentence': sentence,
            'bboxes': [],
            'coords_original': [],
            'labels': [],
            'contexts': [],
        }
    bbox_info = objects['sent_to_bboxes'][sentence_id]
    bbox_info['bboxes'].append(bbox)
    bbox_info['coords_original'].append(coord_original)
    bbox_info['labels'].append(label)
    bbox_info['contexts'].append(context)


def get_objects(dicom_id, gold, gold_objects_df=None, imagenome_filer=None):
    if gold:
        assert gold_objects_df is not None
        object_rows = gold_objects_df[gold_objects_df.image_id.str.replace('.dcm', '') == dicom_id]
        objects = {'bbox_to_sents': {}, 'sent_to_bboxes': {}}
        for i, row in object_rows.iterrows():
            update_objects(
                objects,
                bbox=row.bbox,
                coord_original=eval(row.coord_original),
                sentence_id=row.row_id,
                sentence=row.sentence,
                label=row.label_name,
                context=row.context
            )
    else:
        assert imagenome_filer is not None
        objects = {'bbox_to_sents': {}, 'sent_to_bboxes': {}}
        if not os.path.exists(imagenome_filer.get_silver_scene_graph_json_file(dicom_id)):
            return objects
        scene_graph = imagenome_filer.get_silver_scene_graph_json(dicom_id)
        temp_objects = {obj['object_id']: obj for obj in scene_graph['objects']}
        for bbox_attributes in scene_graph['attributes']:
            if bbox_attributes['object_id'] not in temp_objects.keys():
                continue
            obj = temp_objects[bbox_attributes['object_id']]
            for sentence_id, sentence, sentence_attributes in zip(
                    bbox_attributes['phrase_IDs'], bbox_attributes['phrases'], bbox_attributes['attributes']):
                coord_original = [
                    obj['original_x1'], obj['original_y1'], obj['original_x2'], obj['original_x2']]
                for attribute in sentence_attributes:
                    _, context, label = attribute.split('|')
                    update_objects(
                        objects,
                        bbox=obj['bbox_name'],
                        coord_original=coord_original,
                        sentence_id=sentence_id,
                        sentence=sentence,
                        label=label,
                        context=context
                    )
    return objects


def save_and_get_all_location_condition_pairs(dataset, filename):
    if not os.path.exists(filename):
        location_condition_pairs = {
            'location_to_condition': {},
            'condition_to_location': {}
        }
        for i in tqdm(range(len(dataset)), total=len(dataset)):
            instance = dataset[i]
            patient_id = next(iter(instance.keys()))
            study_id = next(iter(instance[patient_id].keys()))
            dicom_id = next(iter(instance[patient_id][study_id]['images'].keys()))
        #     print(dicom_id)d
            for k, v in instance[patient_id][study_id]['objects'][dicom_id]['sent_to_bboxes'].items():
                sent_condition_to_locations = {}
                for label, context, bbox in zip(v['labels'], v['contexts'], v['bboxes']):
                    if (label, context) not in sent_condition_to_locations.keys():
                        sent_condition_to_locations[(label, context)] = set()
                    sent_condition_to_locations[(label, context)].add(bbox)
                for (label, context), bboxes in sent_condition_to_locations.items():
                    bboxes = tuple(sorted(list(bboxes)))
                    if (label, context) not in location_condition_pairs['condition_to_location'].keys():
                        location_condition_pairs['condition_to_location'][(label, context)] = set()
                    location_condition_pairs['condition_to_location'][(label, context)].add(bboxes)
                    if bboxes not in location_condition_pairs['location_to_condition'].keys():
                        location_condition_pairs['location_to_condition'][bboxes] = set()
                    location_condition_pairs['location_to_condition'][bboxes].add((label, context))
        with open(filename, 'wb') as f:
            pkl.dump(location_condition_pairs, f)
    else:
        with open(filename, 'rb') as f:
            location_condition_pairs = pkl.load(f)
    return location_condition_pairs


class GenerateContextLocationConditionSentences:
    def __call__(self, conditions, contexts, locations):
        condition_to_locations = {}
        for context, loc, condition in zip(contexts, locations, conditions):
            if condition not in condition_to_locations.keys():
                condition_to_locations[condition] = []
            if context == 'yes':
                condition_to_locations[condition].append(loc)
        sentence = ''
        for cond, locs in condition_to_locations.items():
            new_locs = set()
            for loc in locs:
                if ('left' in loc or 'right' in loc) and \
                (loc.replace('left', 'right') in locs or loc.replace('right', 'left') in locs):
                    new_locs.add(loc.replace('left ', '').replace('right ', '') + 's')
                else:
                    new_locs.add(loc)
            new_locs = list(new_locs)
            if len(locs) == 0:
                sentence += ' There is no ' + cond + '.'
            else:
                if len(new_locs) > 2:
                    loclist = ', '.join(new_locs[:-1]) + ', and ' + new_locs[-1]
                else:
                    loclist = ' and '.join(new_locs)
                if cond == 'normal' or cond == 'abnormal':
                    sentence += ' The ' + loclist + (' are ' if len(locs) > 1 else ' is ') + cond + '.'
                else:
                    sentence += ' There is ' + cond + ' in the ' + loclist + '.'
        return sentence.strip()


class ImaGenomeDataset(MimicCxr):
    def __init__(self, df, mimic_cxr_filer, imagenome_filer, group_by='sentence', gold=False, randomize_reports=False,
                 randomize_objects_mode=None, sentences_df=None, sentence_selector=None, swap_left_right=False,
                 swap_left_right_coords=False, generate_sent=False, swap_conditions=False, valid_locations_conditions=None):
        self.group_by_sentence = group_by == 'sentence'
        if self.group_by_sentence:
            group_by = 'image'
        super().__init__(df, mimic_cxr_filer, group_by=group_by, randomize_reports=randomize_reports)
        self.imagenome_filer = imagenome_filer
        self.gold = gold
        if self.gold:
            self.gold_objects_df = self.imagenome_filer.get_gold_file('gold_object_attribute_with_coordinates.txt')
        else:
            self.gold_objects_df = None
        self.randomize_objects_mode = randomize_objects_mode
        self.sentences_df = sentences_df
        if self.group_by_sentence:
            assert self.sentences_df is not None
        self.sentence_selector = sentence_selector
        if self.sentence_selector is not None:
            assert self.group_by_sentence
            self.sentences_df = self.sentences_df[self.sentences_df.apply(self.sentence_selector, axis=1)]
        self.swap_left_right = swap_left_right
        if self.swap_left_right:
            assert self.group_by_sentence
        self.swap_left_right_coords = swap_left_right_coords
        if self.swap_left_right_coords:
            assert self.group_by_sentence and randomize_objects_mode is None
        self.generate_sent = generate_sent
        if self.generate_sent:
            assert self.group_by_sentence and not self.swap_left_right
            self.sentence_generator = GenerateContextLocationConditionSentences()
        else:
            self.sentence_generator = None
        self.swap_conditions = swap_conditions
        self.valid_locations_conditions = valid_locations_conditions
        if self.swap_conditions:
            assert self.generate_sent and self.valid_locations_conditions is not None

    def get_negative_parts_for_objects(self, objects, get_external_negatives=False, part_type='bbox', dicom_id=None):
        assert part_type in {'sentence', 'bbox'}
        neg_parts = []
        if get_external_negatives:
            assert dicom_id is not None
        else:
            assert part_type == 'bbox'
        while len(neg_parts) < len(objects['sent_to_bboxes']):
            if get_external_negatives:
                neg_row = self.get_negative_row(self.df[self.df.dicom_id == dicom_id].iloc[0])
                neg_objects = get_objects(
                    neg_row.dicom_id, self.gold, gold_objects_df=self.gold_objects_df,
                    imagenome_filer=self.imagenome_filer)
            else:
                neg_objects = objects
            for sentence_id, obj in neg_objects['sent_to_bboxes'].items():
                part = {k: v for k, v in obj.items() if k != 'sentence'} \
                    if part_type == 'bbox' else {'sentence': obj['sentence']}
                part['original_sentence_id'] = sentence_id
                part['part_randomized'] = part_type
                neg_parts.append(part)
        neg_parts = neg_parts[:len(objects['sent_to_bboxes'])]
        random.shuffle(neg_parts)
        return neg_parts

    def randomize_objects(self, objects, dicom_id=None, mode='random_sentences'):
        assert mode in {'random_bboxes', 'random_sentences', 'shuffle_bboxes_sentences'}
        part_type = 'sentence' if mode == 'random_sentences' else 'bbox'
        get_external_negatives = mode != 'shuffle_bboxes_sentences'
        neg_parts = self.get_negative_parts_for_objects(
            objects, get_external_negatives=get_external_negatives, part_type=part_type, dicom_id=dicom_id)
        new_objects = {k: v if k not in ['sent_to_bboxes', 'bbox_to_sents'] else {} for k, v in objects.items()}
        new_objects['mode'] = mode
        for (sentence_id, original_value), neg_part in zip(objects['sent_to_bboxes'].items(), neg_parts):
            new_value = dict(original_value)
            new_value.update(neg_part)
            sentence = new_value['sentence']
            for bbox, coord_original, label, context in zip(
                    new_value['bboxes'], new_value['coords_original'], new_value['labels'], new_value['contexts']):
                update_objects(new_objects, bbox, coord_original, sentence_id, sentence, label, context)
            # update with any additional information contained in new_value specifically for the sent_to_bboxes dict
            new_objects['sent_to_bboxes'][sentence_id].update(new_value)
        return new_objects

    def swap_left_right_coords(self, objects):
        raise NotImplementedError

    def __len__(self):
        if self.group_by_sentence:
            return len(self.sentences_df)
        else:
            return super().__len__()

    def __getitem__(self, item):
        if self.group_by_sentence:
            row = self.sentences_df.iloc[item]
            sent_id, dicom_id = row.sent_id, row.dicom_id
            rows = self.df[self.df.dicom_id == dicom_id]
            return_dict = self.get_item_from_rows(rows)
        else:
            sent_id = None
            return_dict = super().__getitem__(item)
        return_dict = self.add_objects(return_dict, sent_id=sent_id)
        if sent_id is not None:
            for subject_id, v1 in return_dict.items():
                for study_id, v2 in v1.items():
                    v2['index'] = item
        return return_dict

    def get_swapped_conditions(self, labels, contexts, bboxes):
        condition_to_locations = {}
        for label, context, bbox in zip(labels, contexts, bboxes):
            if (label, context) not in condition_to_locations.keys():
                condition_to_locations[(label, context)] = set()
            condition_to_locations[(label, context)].add(bbox)
        new_labels, new_contexts, new_bboxes = [], [], []
        for (label, context), bboxes in condition_to_locations.items():
            bboxes = tuple(sorted(list(bboxes)))
            potential_conditions = self.valid_locations_conditions['location_to_condition'][bboxes]
            potential_conditions = potential_conditions.difference(condition_to_locations.keys())
            if len(potential_conditions) > 0:
                potential_conditions = list(potential_conditions)
                random.shuffle(potential_conditions)
                label, context = potential_conditions[0]
            for bbox in bboxes:
                new_labels.append(label)
                new_contexts.append(context)
                new_bboxes.append(bbox)
        return new_labels, new_contexts, new_bboxes

    def add_objects(self, return_dict, sent_id=None):
        for subject_id, v1 in return_dict.items():
            for study_id, v2 in v1.items():
                objects = {}
                for dicom_id, v3 in v2['images'].items():
                    image_objects = self.imagenome_filer.get_objects(dicom_id)
                    if self.randomize_objects_mode is not None:
                        image_objects = self.randomize_objects(
                            image_objects, dicom_id=dicom_id, mode=self.randomize_objects_mode)
                    if self.swap_left_right_coords:
                        image_objects = self.swap_left_right_coords(image_objects)
                    if sent_id is not None:
                        # if sent_id is not None, group_by_sentence is True and there is only one image
                        sent_label = image_objects['sent_to_bboxes'][sent_id]
                        if self.generate_sent:
                            labels = sent_label['labels']
                            contexts = sent_label['contexts']
                            bboxes = sent_label['bboxes']
                            if self.swap_conditions:
                                labels, contexts, bboxes = self.get_swapped_conditions(labels, contexts, bboxes)
                            sent = self.sentence_generator(labels, contexts, bboxes)
                        else:
                            sent = sent_label['sentence']
                            if self.swap_left_right:
                                sent = sent.replace(
                                    'right', 'right*****').replace(
                                    'left', 'right').replace(
                                    'right*****', 'left')
                        v2['sentence'] = sent
                        # register sentence id
                        v2['sent_id'] = sent_id
                    objects[dicom_id] = image_objects
                v2['objects'] = objects
        return return_dict


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


def get_ent_to_bbox(sent_labels, sent_contexts, sent_bbox_names):
    ent_to_bbox = {}
    for label, context, bbox in zip(sent_labels, sent_contexts, sent_bbox_names):
        if (label, context) not in ent_to_bbox.keys():
            ent_to_bbox[(label, context)] = set()
        ent_to_bbox[(label, context)].add(bbox)
    return ent_to_bbox


def get_ent_to_bbox_from_row(row):
    return get_ent_to_bbox(eval(row['sent_labels']), eval(row['sent_contexts']), eval(row['bbox_names']))


class RowLabelAndContextSelector(RowContainsOrDoesNotContainSelector):
    def get_row_set(self, row):
        return set(get_ent_to_bbox_from_row(row).keys())


class RowBBoxSelector(RowContainsOrDoesNotContainSelector):
    def get_row_set(self, row):
        return set(eval(row['bbox_names']))


class ImaGenomeDataModule(BaseDataModule):
    def __init__(self, mimic_cxr_filer, imagenome_filer, batch_size=1, num_workers=0, collate_fn=default_collate_fn,
                 get_images=True, get_reports=True, force=False, parallel=False,
                 num_preprocessing_workers=os.cpu_count(), chunksize=1, split_slices='train,valid,test,gold', gold_test=False,
                 limit_to=None, randomize_reports=False, randomize_objects_mode=None, swap_left_right=False,
                 generate_sent=False, swap_conditions=False, group_by='sentence', **kwargs):
        super().__init__(batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn, **kwargs)
        self.mimic_cxr_filer = mimic_cxr_filer
        self.imagenome_filer = imagenome_filer
        self.get_images = get_images
        self.get_reports = get_reports
        self.force = force
        self.parallel = parallel
        self.num_preprocessing_workers = num_preprocessing_workers
        self.chunksize = chunksize
        self.data_prepared = False
        self.split_slices = {}
        for split_slice in split_slices.split(','):
            if ':' not in split_slice:
                key = split_slice
                value = None
            else:
                key_values = split_slice.split(':')
                key = key_values[0]
                value = slice(*[int(i) for i in key_values[1:]])
            if key == '': continue
            assert key in {'train', 'valid', 'test', 'gold'}
            self.split_slices[key] = value
        self.gold_test = gold_test
        self.limit_to = limit_to
        self.randomize_reports = randomize_reports
        self.randomize_objects_mode = randomize_objects_mode
        self.swap_left_right = swap_left_right
        self.generate_sent = generate_sent
        self.swap_conditions = swap_conditions
        self.group_by = group_by

    def get_kwargs(self, records):
        return dict(
            records=records, filer=self.mimic_cxr_filer, get_images=self.get_images, get_reports=self.get_reports,
            to_nifti=False, to_pt=True, force_nifti=self.force, force_pt=self.force)

    def yield_args(self, records, subject_ids):
        for i, subject_id in enumerate(tqdm(subject_ids, total=len(subject_ids))):
            subject_records = records[records.subject_id == subject_id]
            kwargs = self.get_kwargs(subject_records)
            if i == 0:
                print('Setting one record\'s processing to verbose to serve as an example.')
                kwargs['verbose'] = True
            yield kwargs

    def process_records(self, records, subject_ids):
        if not self.parallel:
                print('not parallelizing')
                # no parallel:
                records = pd.concat(
                    [process_records(**kwargs) for kwargs in self.yield_args(records, subject_ids)])
        else:
            print('parallelizing')
            failed = True
            if self.mimic_cxr_filer.password is None:
                self.mimic_cxr_filer.password = ''
            while failed:
                try:
                    # parallel:
                    P = mp.Pool
                    with P(self.num_preprocessing_workers) as p:
                        print('starting processes:')
                        results_iterable = p.imap(
                            process_records_mapstyle,
                            list(self.yield_args(records, subject_ids)),
                            chunksize=self.chunksize
                        )
                        print('finishing processes:')
                        records = pd.concat([rs for rs in tqdm(results_iterable, total=len(subject_ids))])
                    failed = False
                except DownloadError:
                    self.mimic_cxr_filer.password = getpass('Please enter physio password: ')
        return records

    def prepare_data(self):
        if self.data_prepared:
            print('data already prepared')
            return
        self.imagenome_filer.download_file()
        dicom_ids = set()
        gold_objects_df = self.imagenome_filer.get_gold_file('gold_object_attribute_with_coordinates.txt')
        for k, v in self.split_slices.items():
            if k in {'train', 'valid', 'test'}:
                self.imagenome_filer.unzip_file('silver_dataset/scene_graph.zip')
            split_df = self.imagenome_filer.get_split(k)
            if k == 'gold':
                gold_object_attribute_with_coordinates_df = \
                    self.imagenome_filer.get_gold_file('gold_object_attribute_with_coordinates.txt')
                dicom_ids = set(gold_object_attribute_with_coordinates_df.image_id.str.replace('.dcm', ''))
                split_df = split_df[split_df.progress_apply(lambda r: r.dicom_id in dicom_ids, axis=1)]
                assert len(split_df) == len(dicom_ids)
            if v is not None:
                split_df = split_df.iloc[v]
            print('Downloading %s %s (%i):' % (k, str(v), len(split_df)))
            subject_ids = set(split_df.subject_id)
            new_records = self.process_records(split_df, subject_ids)
            sent_ids = []
            sent_ids_df = {
                'dicom_sent_id': [],
                'patient_id': [],
                'sent_id': [],
                'dicom_id': [],
                'sent_labels': [],
                'sent_contexts': [],
                'bbox_names': []
            }
            print('Preprocessing objects')
            for i, row in tqdm(new_records.iterrows(), total=len(new_records)):
                if os.path.exists(self.imagenome_filer.get_objects_file(row.dicom_id)):
                    objects = self.imagenome_filer.get_objects(row.dicom_id)
                else:
                    objects = get_objects(
                        row.dicom_id, k == 'gold', gold_objects_df=gold_objects_df, imagenome_filer=self.imagenome_filer)
                    self.imagenome_filer.save_objects(objects, row.dicom_id)
                sent_ids.append(sorted(list(objects['sent_to_bboxes'].keys()), key=lambda x: float(x.split('|')[1])))
                sent_ids_df['sent_id'].extend(sent_ids[-1])
                sent_ids_df['dicom_id'].extend([row.dicom_id] * len(sent_ids[-1]))
                sent_ids_df['patient_id'].extend([row.subject_id] * len(sent_ids[-1]))
                for sent_id in sent_ids[-1]:
                    sent_ids_df['dicom_sent_id'].append('dicom_%s_sent_%s' % (row.dicom_id, sent_id))
                    sent_label = objects['sent_to_bboxes'][sent_id]
                    sent_ids_df['sent_labels'].append(sent_label['labels'])
                    sent_ids_df['sent_contexts'].append(sent_label['contexts'])
                    sent_ids_df['bbox_names'].append(sent_label['bboxes'])
            new_records['sent_ids'] = sent_ids
            new_records.to_csv(self.imagenome_filer.get_full_path('%s_subset.csv' % k))
            pd.DataFrame(sent_ids_df).to_csv(self.imagenome_filer.get_full_path('%s_sentences.csv' % k))
        save_and_get_all_location_condition_pairs(
            self.get_dataset('gold', generate_sent=False, swap_conditions=False),
            self.imagenome_filer.get_full_path('valid_locations_and_conditions.pkl'))
        self.data_prepared = True

    def get_dataset(self, split, **kwargs):
        for k in ['limit_to', 'randomize_reports', 'randomize_objects_mode', 'swap_left_right', 'generate_sent',
                  'swap_conditions', 'group_by']:
            if k not in kwargs.keys():
                kwargs[k] = getattr(self, k)
        if kwargs['limit_to'] is None:
            kwargs['sentence_selector'] = None
        elif kwargs['limit_to'] == 'abnormal':
            kwargs['sentence_selector'] = RowLabelAndContextSelector(contains={('abnormal', 'yes')})
        else:
            raise Exception
        del kwargs['limit_to']
        kwargs['gold'] = split == 'gold'
        split_df = pd.read_csv(self.imagenome_filer.get_full_path('%s_subset.csv' % split))
        kwargs['sentences_df'] = pd.read_csv(self.imagenome_filer.get_full_path('%s_sentences.csv' % split))
        if kwargs['generate_sent'] and kwargs['swap_conditions']:
            with open(self.imagenome_filer.get_full_path('valid_locations_and_conditions.pkl'), 'rb') as f:
                kwargs['valid_locations_conditions'] = pkl.load(f)
        return ImaGenomeDataset(split_df, self.mimic_cxr_filer, self.imagenome_filer, **kwargs)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self._train = self.get_dataset('train')
            self._val = self.get_dataset('valid')
        if stage == 'test' or stage is None:
            if self.gold_test:
                self._test = self.get_dataset('gold')
            else:
                self._test = self.get_dataset('test')
