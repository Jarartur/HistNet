# %% Imports
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from utils import Align_subject, resizeAndPad, resize_dataset
from torchvision import transforms
import torchio as tio
import os

run_identifier = 'longer_epochs-w_avg'
# run_identifier = 'hparams-testing'
resize = 6
# %% General config
config = {'epochs': 150,
          'sample_every': 1,
          # 'flow_every': 1,
          # 'test_every': 100,
        #   'lr_scheduler_every': 250,
          # 'lmbd_grad_reg': 1000,
          'checkpoint_every': 25,
          }

# %% Data config
# normal transform used for evaluation
data_config = {'train_root': f'resized_{resize}/',
               'test_root': f'resized_{resize}/',
               'main_file_path': 'raw/edited_data_v3.csv',
               'summary_path': f'runs/registration_v2/{run_identifier}',
        #        'sample_dir': 'sample',
              #  'batch_size': 64,
               'resample_rate': resize,
               'model_checkpoint': 'checkpoints/registration_v2/RegistrationNetwork-mv_test_avg-cost_ncc_local-vecint_None-reg_diffusion-lr_0.001-decay_0.995-bsize_8-lmd_reg_4000-lmd_trans_None.tar'}

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
                'reset_optim': True,
                'reset_epoch': True,
                'checkpoint_path': f'checkpoints/registration_v2/{run_identifier}',
                }

lr_config = {'decay_kickin': 0,
             'swa_kickin': 130}

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