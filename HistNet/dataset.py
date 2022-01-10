# %% Imports
import random
from numpy.lib.utils import source

import torch
from torch.utils import data
from torchio.data.inference import aggregator
from torchvision.transforms.transforms import ColorJitter
from utils import getFiles, resizeAndPad, Align_subject, resample
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import pandas as pd
from ast import literal_eval
import numpy.linalg as LA

import torchio as tio

class AnhirPatches():
    '''
    Main Dataset

    Parameters
    ----------
    summary_path: str
        path to main file contating relative paths to files and affine matrices.
    root: str
        path to dataset root.
    base_transforms
        transforms applied to both training and testing dataset.
    train_transforms
        transforms applied only to training dataset.
    '''
    def __init__(self, summary_path:str, 
                       root:str=r'data/raw/', 
                       base_transforms=None,
                       train_transforms=None,
                       ) -> None:

        self.base_transforms = base_transforms
        self.train_transforms = train_transforms
        self.summary_path = summary_path
        self.root = root
        
    def get_training_subjects(self) -> tio.SubjectsDataset:
        '''
        Returns torchio Subject dataset. Main difference to pytorch dataset is that
        it returns dictionaries for images with optional metadata and can be used in
        patch-based pipeline.
        Merges base and training transforms.
        '''
        # get python lists of files
        src_img_paths, trg_img_paths, shapes, trans_mats = self._parse_summary(True)
        # create torchio Subjects (dictionaries with optional metadata)
        subjects = self.get_subjects(src_img_paths, trg_img_paths, shapes, trans_mats, self.custom_reader)

        # merge transforms if needed
        transforms = self.base_transforms
        if self.train_transforms is not None: transforms += self.train_transforms
        transforms = tio.Compose(transforms)
        return tio.SubjectsDataset(subjects, transform=transforms)

    def get_eval_subjects(self, mode:str = 'train_eval') -> tio.SubjectsDataset:
        '''
        Params
        ------
        mode: str
            'train_eval' for evaluation on training subjects
            'test_eval' for evaluation on testing subjects (no csvs avilable)
            In both cases only base transforms are applied
        '''
        if mode == 'train_eval':
            src_img_paths, trg_img_paths, shapes, trans_mats = self._parse_summary(True)
        elif mode == 'test_eval':
            src_img_paths, trg_img_paths, shapes, trans_mats = self._parse_summary(False)
        # create torchio Subjects (dictionaries with optional metadata)
        subjects = self.get_subjects(src_img_paths, trg_img_paths, shapes, trans_mats, self.custom_reader)

        transforms = tio.Compose(self.base_transforms)
        return tio.SubjectsDataset(subjects, transform=transforms)
    
    def get_queue(self, patch_size=(256, 256, 1), **kwargs):
        '''
        Creates instance of torchio queue for patch-based pipeline.
        In every iteration it returns a respective patch from source and
        target image as well as theirs positions in [x1, y1, z1, x2, y2, z2] format.
        For 2D data we ignore the z dimension as it is always 1 (singleton dimension).

        Kwargs for tio.Queue
        --------------------
            max_length : int
                controls maximum number of images loaded in memory
            samples_per_volume : int
            
            num_workers : int 
                for multithreading no lambdas

            shuffle_subjects : bool

            shuffle_patches : bool

            start_background : bool

            verbose : bool
        '''

        training_set = self.get_training_subjects()
        sampler = tio.data.UniformSampler(patch_size) # random uniform sampling
        
        # queue doesn't guarantee to go through the whole dataset
        # my current understanding is that it will take `samples_per_volume` number of patches
        # from each image in the dataset
        patches_training_set = tio.Queue(
        subjects_dataset=training_set,
        max_length=kwargs['max_length'],
        samples_per_volume=kwargs['samples_per_volume'],
        sampler=sampler,
        num_workers=kwargs['num_workers'], # for multithreading no lambdas
        shuffle_subjects=True,
        shuffle_patches=True,
        start_background=kwargs['start_background'],
        verbose=False
        )

        return patches_training_set
    
    def get_eval_loader(self, **kwargs):
        '''
        Generator that yields a single image as a patch_loader and
        a patch aggregator at a time.

        Kwargs for GridSampler
        ----------------------
        patch_size : tuple
            size(patch_size) should == 3
        patch_overlap : tuple
            size(patch_overlap) should == 3

        (last dimension == 1 (singleton dimension) because torchio works on 3D images by default)
        '''
        eval_set = self.get_eval_subjects(mode='train_eval') #NOTE currently hard-coded
        for subject in eval_set:
            # grid sampler is different to random sampling only that it samples uniformly and makes sure the whole image is patchified
            grid_sampler = tio.inference.GridSampler(
                subject,
                patch_size=kwargs['patch_size'],
                patch_overlap=kwargs['patch_overlap'],
            )

            patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=1, shuffle=False) #NOTE: hard coded batch_size=1 is needed, possible fix is to use a lsit of aggregators
            aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode='average') # combines and averages displacement fields
            yield patch_loader, aggregator

    def get_eval_sampler(self, low, high):
        '''In progres, curently not used anywhere.'''
        eval_set = self.get_eval_subjects()
        for subject in eval_set:
            ...

    def _parse_summary(self, training):
        '''
        Creates lists of files for dataset creation
        
        Parameters
        ----------
        training: str
            controls what pairs are used with respect to status field
        '''
        data = pd.read_csv(self.summary_path, converters={'Source transformation matrix': lambda s: np.array(literal_eval(s)),
                                              'Image diagonal [pixels]': literal_eval,
                                              'Image size [pixels]': literal_eval,
                                              'Source image size': literal_eval,
                                              })
        # choosing which data entries are used
        if training:
            data = data[data['status']=='training']
        else:
            data = data[data['status']=='evaluation']
        data = data.reset_index(drop=True)

        src_img_paths = self.root + data['Source image']
        trg_img_paths = self.root + data['Target image']
        src_shapes = data['Source image size']
        trg_shapes = data['Image size [pixels]']
        trans_mats = data['Source transformation matrix'] # affine matricies for later alignment

        # shapes are not currently used
        # they were planned as a fail safe for working with original dataset size
        src_shapes = np.array(src_shapes.to_list())
        trg_shapes = np.array(trg_shapes.to_list())
        x = np.max(np.concatenate((src_shapes[:, :1], trg_shapes[:, :1]), axis=1), axis=1)
        y = np.max(np.concatenate((src_shapes[:, 1:2], trg_shapes[:, 1:2]), axis=1), axis=1)
        assert x.shape == y.shape

        channels = np.ones_like(y)*3
        shapes = np.array([x, y, channels])

        return src_img_paths, trg_img_paths, shapes, trans_mats
    
    @staticmethod
    def get_subjects(src_img_paths, trg_img_paths, shapes, trans_mats, reader):
        '''
        Creates a list of torchio subjects (dictionaries with metadata) for dataset creation.
        '''
        subjects = []
        for i, (src_path, trg_path) in enumerate(zip(src_img_paths, trg_img_paths)):
            subject = tio.Subject(
                src=tio.ScalarImage(src_path, reader=reader),
                trg=tio.ScalarImage(trg_path, reader=reader),
                shape=shapes[:, i],
                trans_mat=trans_mats[i]
            )
            subjects.append(subject)

        return subjects

    @staticmethod
    def custom_reader(path):
        '''
        Reader for lazy loading in torchio Subjects.
        Default one messes up the orientation.
        '''
        path = str(path)
        image = cv2.imread(path)
        affine = np.eye(4)
        return image, affine

# %% Test
def test_patches_train():
    transforms = [
        Align_subject(15),
        tio.RescaleIntensity(out_min_max=(-1, 1)),
        ]
    Anhir_Dataset = AnhirPatches('data/processed/edited_data_v3.csv', 'data/raw/', transforms)

    dataset = Anhir_Dataset.get_training_subjects()
    print(f'{len(dataset)=}')
    sample_subjects = next(iter(dataset))
    print(f'{sample_subjects=}')
    print(f'{sample_subjects["src"]=}')
    print(f'{sample_subjects["trg"]=}')
    print()

    queue = Anhir_Dataset.get_queue(max_length=2,
                                    samples_per_volume=2,
                                    num_workers=0, # for multithreading no lambdas
                                    shuffle_subjects=False,
                                    shuffle_patches=False,
                                    start_background=False,
                                    verbose=True)
    sample_queue = next(iter(queue))
    print(f'{sample_queue=}')
    print(f'{sample_queue["src"]=}')
    print(f'{sample_queue[tio.LOCATION].shape=}')
    print(f'{sample_queue["src"][tio.DATA].shape=}')
    print()

    training_loader_patches = torch.utils.data.DataLoader(
    queue, batch_size=1)
    sample_queue_loader = next(iter(training_loader_patches))
    print(f'{sample_queue_loader[tio.LOCATION].shape=}')
    print(f'{sample_queue_loader["src"][tio.DATA].shape=}')

def test_patches_eval():

    transforms = [
        Align_subject(15),
        tio.RescaleIntensity(out_min_max=(-1, 1)),
        ]
    Anhir_Dataset = AnhirPatches('data/processed/edited_data_v3.csv', 'data/raw/', transforms)
    model = nn.Identity().eval()

    outputs_list = []

    params = {'patch_size':(256, 256, 1), 'patch_overlap':(50, 50, 0)}
    with torch.no_grad():
        for i, (patch_loader, aggregator) in enumerate(Anhir_Dataset.get_eval_loader(**params)):
            for patches_batch in patch_loader:
                input_tensor = patches_batch['src'][tio.DATA]
                locations = patches_batch[tio.LOCATION]
                outputs = model(input_tensor)
                aggregator.add_batch(outputs, locations)

            print(f'{input_tensor.shape=}')
            print(f'{outputs.shape=}')
            print(f'{aggregator.get_output_tensor().shape=}')
            outputs_list += [aggregator.get_output_tensor()]
            print(f'--------- {i} -----------')
            if i > 2: break
    
    print()
    print(f'{len(outputs_list)=}')
    print(f'{outputs_list[0].shape=}')

def subject_test():
    transforms = [
        Align_subject(15),
        tio.RescaleIntensity(out_min_max=(-1, 1)),
        ]
    Anhir_Dataset = AnhirPatches('data/processed/edited_data_v3.csv', 'data/raw/', transforms)
    eval_set = Anhir_Dataset.get_eval_subjects()
    print([eval_set[i] for i in range(10, 20)])

if __name__ == '__main__':
    import torch
    import torch.nn as nn
    from torchvision import transforms
    from torch.utils.data.dataloader import DataLoader
    import utils as utils
    # import albumentations as A
    # from albumentations.pytorch import ToTensorV2
    from utils import resizeAndPad
    import cv2

    print('\nTesting training... ⏳')
    test_patches_train()
    print('\nTesting evaluation... ⏳')
    test_patches_eval()
    print('\nTesting subject... ⏳')
    subject_test()
    print('\nDone... ✔️')


# %%
