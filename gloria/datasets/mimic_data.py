import torch
from torch import default_generator
from torch.utils.data import Dataset, random_split, DataLoader
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
    def __init__(self, batch_size=1, num_workers=0, collate_fn=default_collate_fn):
        super().__init__()
        for i, split in enumerate(['train', 'val', 'test']):
            setattr(self, '%s_dataloader_kwargs' % split, dict(
                batch_size=batch_size,
                num_workers=num_workers,
                collate_fn=collate_fn[i] if isinstance(collate_fn, tuple) else collate_fn
            ))
        self._train = None
        self._val = None
        self._test = None

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
        return DataLoader(self.train, shuffle=False, **self.train_dataloader_kwargs)

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
            self.save_dicom(row)
            try:
                image = np.array(pydicom.dcmread(self.get_dicom_path(row)).pixel_array, dtype=np.int16)
            except Exception as e:
                print('error with dicom path:', self.get_dicom_path(row))
                e.load_error_file_path = self.get_dicom_path(row)
                # print('deleting')
                # subprocess.run(['rm', self.get_dicom_path(row)])
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
    def __init__(self, df, filer, group_by='patient'):
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

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        id = self.ids[item]
        if self.group_by == 'patient':
            rows = self.df[self.df.subject_id == id].iterrows()
        elif self.group_by == 'study':
            rows = self.df[self.df.study_id == id].iterrows()
        elif self.group_by == 'image':
            rows = self.df[self.df.dicom_id == id].iterrows()
        else:
            raise Exception
        return_dict = {}
        for i, row in rows:
            if row.subject_id not in return_dict.keys():
                return_dict[row.subject_id] = {}
            if row.study_id not in return_dict[row.subject_id].keys():
                return_dict[row.subject_id][row.study_id] = {'images': {}}
            return_dict[row.subject_id][row.study_id]['images'][row.dicom_id] = self.filer.get_ptimage(row)
            return_dict[row.subject_id][row.study_id]['report'] = self.filer.get_report(row)
        return return_dict


class MimicCxrDataModule(BaseDataModule):
    def __init__(self, filer, get_images=True, get_reports=True, splits=(.8, .1), batch_size=1, num_workers=0,
                 dataslice=None, collate_fn=default_collate_fn, greater_than_n_studies=0, force=False, registration=None,
                 parallel=False, num_preprocessing_workers=os.cpu_count(), chunksize=1, chexpert_labeler=None):
        super().__init__(batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)
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
        return MimicCxr(records_df, self.filer)

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

    def get_silver_scene_graph_json(self, dicom_id):
        with open(self.get_full_path('silver_dataset/scene_graph/%s_SceneGraph.json' % dicom_id), 'r') as f:
            return json.load(f)


class ImaGenomeDataset(MimicCxr):
    def __init__(self, df, mimic_cxr_filer, imagenome_filer, group_by='patient', gold=False):
        super().__init__(df, mimic_cxr_filer, group_by=group_by)
        self.imagenome_filer = imagenome_filer
        self.gold = gold
        if self.gold:
            self.gold_objects_df = self.imagenome_filer.get_gold_file('gold_object_attribute_with_coordinates.txt')

    def get_objects(self, dicom_id):
        if self.gold:
            object_rows = self.gold_objects_df[self.gold_objects_df.image_id.str.replace('.dcm', '') == dicom_id]
            objects = {'original_object_rows': object_rows, 'bbox_to_sents': {}, 'sent_to_bboxes': {}}
            for i, row in object_rows.iterrows():
                coord_original = eval(row.coord_original)
                if row.bbox not in objects['bbox_to_sents'].keys():
                    objects['bbox_to_sents'][row.bbox] = {
                        'coord_original': coord_original,
                        'row_ids': [],
                        'sentences': [],
                        'labels': [],
                        'contexts': [],
                    }
                sent_info = objects['bbox_to_sents'][row.bbox]
                sent_info['row_ids'].append(row.row_id)
                sent_info['sentences'].append(row.sentence)
                sent_info['labels'].append(row.label_name)
                sent_info['contexts'].append(row.context)
                if row.row_id not in objects['sent_to_bboxes'].keys():
                    objects['sent_to_bboxes'][row.row_id] = {
                        'sentence': row.sentence,
                        'bboxes': [],
                        'coords_original': [],
                        'labels': [],
                        'contexts': [],
                    }
                bbox_info = objects['sent_to_bboxes'][row.row_id]
                bbox_info['bboxes'].append(row.bbox)
                bbox_info['coords_original'].append(coord_original)
                bbox_info['labels'].append(row.label_name)
                bbox_info['contexts'].append(row.context)
        else:
            scene_graph = self.imagenome_filer.get_silver_scene_graph_json(dicom_id)
#             raise NotImplementedError
            objects = None
        return objects

    def __getitem__(self, item):
        """super:
        id = self.ids[item]
        if self.group_by == 'patient':
            rows = self.df[self.df.subject_id == id].iterrows()
        elif self.group_by == 'study':
            rows = self.df[self.df.study_id == id].iterrows()
        elif self.group_by == 'image':
            rows = self.df[self.df.dicom_id == id].iterrows()
        else:
            raise Exception
        return_dict = {}
        for i, row in rows:
            if row.subject_id not in return_dict.keys():
                return_dict[row.subject_id] = {}
            if row.study_id not in return_dict[row.study_id].keys():
                return_dict[row.subject_id][row.study_id] = {'images': {}}
            return_dict[row.subject_id][row.study_id]['images'][row.dicom_id] = self.filer.get_ptimage(row)
            return_dict[row.subject_id][row.study_id]['report'] = self.filer.get_report(row)
        return return_dict
        """
        return_dict = super().__getitem__(item)
        for subject_id, v1 in return_dict.items():
            for study_id, v2 in v1.items():
                v2['objects'] = {dicom_id: self.get_objects(dicom_id) for dicom_id, v3 in v2['images'].items()}
        return return_dict


class ImaGenomeDataModule(BaseDataModule):
    def __init__(self, mimic_cxr_filer, imagenome_filer, batch_size=1, num_workers=0, collate_fn=default_collate_fn,
                 get_images=True, get_reports=True, force=False, parallel=False,
                 num_preprocessing_workers=os.cpu_count(), chunksize=1, split_slices='train,valid,test,gold', gold_test=False):
        super().__init__(batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)
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
            assert key in {'train', 'valid', 'test', 'gold'}
            self.split_slices[key] = value
        self.gold_test = gold_test

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

    def prepare_data(self):
        if self.data_prepared:
            print('data already prepared')
            return
        self.imagenome_filer.download_file()
        dicom_ids = set()
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
            new_records = []
            for kwargs in self.yield_args(split_df, subject_ids):
                try:
                    new_records.append(process_records(**kwargs))
                except Exception as e:
                    print(e)
            new_records = pd.concat(new_records)
            #new_records = pd.concat(
            #    [process_records(**kwargs) for kwargs in self.yield_args(split_df, subject_ids)])
            #new_records = process_records(**self.get_kwargs(split_df), verbose=True)
            new_records.to_csv(self.imagenome_filer.get_full_path('%s_subset.csv' % k))
        self.data_prepared = True

    def get_dataset(self, split):
#         split_df = self.imagenome_filer.get_split(split)
#         v = self.split_slices[split]
#         if v is not None:
#             split_df = split_df[v]
        split_df = pd.read_csv(self.imagenome_filer.get_full_path('%s_subset.csv' % split))
        return ImaGenomeDataset(split_df, self.mimic_cxr_filer, self.imagenome_filer, gold=split=='gold', group_by='image')

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self._train = self.get_dataset('train')
            self._val = self.get_dataset('valid')
        if stage == 'test' or stage is None:
            if self.gold_test:
                self._test = self.get_dataset('gold')
            else:
                self._test = self.get_dataset('test')


from transformers import AutoTokenizer
import cv2
from PIL import Image
from gloria import builder


def original_tensor_to_numpy_image(image):
    return np.array((image.float() / image.max()) * 255, dtype=np.uint8)


class GloriaCollateFn:
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)
        self.transform = builder.build_transformation(self.cfg, split)

    def __call__(self, instances):
        imgs, cap_len, ids, tokens, attention, path = [], [], [], [], [], []

        # flattern
        for instance in instances:
            patient_id = next(iter(instance.keys()))
            study_id = next(iter(instance[patient_id].keys()))
            instance = instance[patient_id][study_id]
            dicom_id = next(iter(instance['images'].keys()))

            # get image
            # x = cv2.imread(str(img_path), 0)
            x = original_tensor_to_numpy_image(instance['images'][dicom_id])
            # tranform images
            x = self._resize_img(x, self.cfg.data.image.imsize)
            img = Image.fromarray(x).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)
            # get caption
            cap = self.tokenizer(
                instance['report'],
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=self.cfg.data.text.word_num,
            )
            x_len = len([t for t in cap["input_ids"][0] if t != 0])
            ids.append(cap["input_ids"])
            cap_len.append(x_len)
            tokens.append(cap["token_type_ids"])
            attention.append(cap["attention_mask"])

        # stack
        imgs = torch.stack(imgs)
        ids = torch.stack(ids).squeeze()
        tokens = torch.stack(tokens).squeeze()
        attention = torch.stack(attention).squeeze()

        # sort and add to dictionary
        sorted_cap_lens, sorted_cap_indices = torch.sort(torch.tensor(cap_len), 0, True)
        return_dict = {
            "caption_ids": ids[sorted_cap_indices],
            "token_type_ids": tokens[sorted_cap_indices],
            "attention_mask": attention[sorted_cap_indices],
            "imgs": imgs[sorted_cap_indices],
            "cap_lens": sorted_cap_lens,
        }

        return return_dict

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


def normalize(image):
    image = image.float()
    return ((image - image.min()) / (image.max() - image.min()))
