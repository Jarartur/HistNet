# %% Imports
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from utils import Align_subject, resizeAndPad, resize_dataset
from torchvision import transforms
import torchio as tio
import os

run_identifier = 'multiresolution-level_1_2'
# run_identifier = 'hparams-testing'
resize = 6
# %% General config
config = {'epochs': 1500,
          'sample_every': 1,
          'eval_every': 50,
          # 'flow_every': 1,
          # 'test_every': 100,
        #   'lr_scheduler_every': 250,
          # 'lmbd_grad_reg': 1000,
          'checkpoint_every': 1,
          }

# %% Data config
# normal transform used for evaluation
data_config = {'train_root': f'resized_{resize}/',
               'test_root': f'resized_{resize}/',
               'main_file_path': 'raw/edited_data_v3.csv',
               'summary_path': f'runs/multires/{run_identifier}',
               'resample_rate': resize,
               'model_checkpoint': 'checkpoints/multires/multiresolution-patches_512-levels_2-freeze_[2]-cost_ncc_local-reg_diffusion-lr_0.0001-decay_0.99-bsize_4-lmd_reg_6000-lmd_trans_None.tar'}

base_transforms = [
        # Align_subject(data_config['resample_rate']),
        Align_subject(1),
        tio.RescaleIntensity(out_min_max=(0, 1)),
        ]
# %% Model config
model_config = {#'learning_rate': 1e-4,
                # 'decay_rate': 0.995,
                'betas': [0.9, 0.999],
                'resume': True,
                'reset_optim': False,
                'reset_epoch': False,
                'checkpoint_path': f'checkpoints/multires/{run_identifier}',
                }

def init_data(root='raw/', resize=4, summary_path='raw/edited_data_v3.csv', save_root=''):
  '''
  Function for resizing dataset for reduced footprint.
  Checks if resized dataset for a given factor exists. If not it creates it.'''
  # check if dir exists
  # if not make dir
    # and populate with resized images
  savepath = save_root + f'resized_{resize}/'

  if os.path.exists(savepath):
    print("Using existing data")
  else:
    print("Creating new resized copy of data...")
    resize_dataset(root, resize, summary_path, savepath)
    print("Done")