import numpy as np
import pandas as pd
import scipy.ndimage as nd
from ast import literal_eval
import random
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision
import torchio as tio
from utils import Align_subject
from dataset import AnhirPatches
# import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.cuda.amp as amp

from utils import resizeAndPad_landmarks, tc_df_to_np_df, df_field_v2, scale_patch, rescale_to_tc_df, make_grid, resample_landmarks, df_field

def evaluate(DataSet, registration_model, transfer_model, main_file_path, device, data_path, resample_rate, **params):
    '''
    Wrapper for model evaluation
    
    Parameters
    ----------
    DataSet: dataset.AnhirPatches
        custom dataset
    registration_model:
        non rigid registration model
    transfer_model:
        style transfer model (optional, omitted if == None)
    main_file_path: 
        summary_file
    device:
        pytorch device
    data_path: str
        data root
    resample_rate: int
        factor by which original dataset have been resized
    
    Returns
    -------
    grid: torch.Tensor
        grid of full images (coresponding to resampled dataset size)
    '''
    grid, deformation_fields = inference_to_grid_df(DataSet, registration_model, transfer_model, device, **params)
    rTRE = inference_to_rtre(main_file_path, deformation_fields, data_path, resample_rate)

    return grid, np.mean(rTRE)

def inference_to_grid_df(DataSet, registration_model, transfer_model, device, **params):
    '''
    Main test-time inference.
    Goes over the whole dataset image-by-image
    patch-by-patch and aggregates output displacement fields.

    Parameters
    ----------
    DataSet: dataset.AnhirPatches
        custom dataset
    registration_model:
        non rigid registration model
    transfer_model:
        style transfer model (optional, omitted if == None)
    main_file_path: 
        summary_file
    device:
        pytorch device
    '''
    # inference
    registration_model.eval()
    if transfer_model is not None: transfer_model.eval()

    testloader = DataSet.get_eval_loader(**params)

    deformation_fields = []
    with torch.no_grad(): # replace with torch.inference_mode() for pytorch >= 1.9
        # get patch sampler and respective aggregator for a single image
        for patch_loader, aggregator in tqdm(testloader, desc='Evaluation'):
            # iterate over all patches
            for patches_batch in patch_loader:
                locations = patches_batch[tio.LOCATION]
                content_image = patches_batch['src'][tio.DATA].squeeze(-1).to(device)
                style_image = patches_batch['trg'][tio.DATA].squeeze(-1).to(device)

                x = TF.rgb_to_grayscale(content_image)
                # transfer / grayscale
                if transfer_model is not None:
                    intense_tensor = transfer_model(content_image, style_image)
                    content_image = ((content_image * intense_tensor[:, 0:3, :, :]) + intense_tensor[:, 3:, :, :])

                content_image = TF.rgb_to_grayscale(content_image)
                style_image = TF.rgb_to_grayscale(style_image)

                # non rigid transform
                #NOTE this can be streamlined by doing df_field after aggregating output
                deformation_field = registration_model(style_image, content_image)                                     # permute(0, 3, 1, 2) to cheat torchio aggregator which has hard-coded data dimension
                n_grid = df_field_v2(content_image, deformation_field, None, device).permute(0, 3, 1, 2).unsqueeze(-1) # -1 for z-dimension to cheat torchio aggregator
                n_grid = scale_patch(n_grid, locations[0]) # scaling to numpy displacement format for  aggregating to make sense
                aggregator.add_batch(n_grid, locations)

            deformation_fields += [aggregator.get_output_tensor().squeeze(-1).permute(1, 2, 0).unsqueeze(0)] # getting aggregator output for a single image

        deformation_fields = [rescale_to_tc_df(df, df.shape[1:3]).float() for df in deformation_fields]
        #NOTE there's gotta be a better way but i don't have time
        nr_samples = 3
        low = random.randint(1, len(deformation_fields)-nr_samples)
        high = low+nr_samples
        subjects = DataSet.get_eval_subjects()
        contents = [subjects[i] for i in range(low, high)]

        srcs = [TF.rgb_to_grayscale(subject['src'][tio.DATA].squeeze(-1).unsqueeze(0)).to(device) for subject in contents]
        trgs = [TF.rgb_to_grayscale(subject['trg'][tio.DATA].squeeze(-1).unsqueeze(0)).to(device) for subject in contents]
        grids = [deformation_fields[i].to(device) for i in range(low, high)]

        outputs = [F.grid_sample(x, n_grid, mode='bilinear', padding_mode='zeros') for x, n_grid in zip(srcs, grids)]

        grid = make_grid(outputs, nr_samples, srcs, trgs)

    registration_model.train()
    if transfer_model is not None: transfer_model.train()

    return grid, deformation_fields

def inference_to_rtre(main_file_path, deformation_fields, data_path, resample_rate):
    '''
    Transforming landmarks and calculating rTRE.
    '''

    rTRE = []
    data = pd.read_csv(main_file_path, converters={'Source transformation matrix': lambda s: np.array(literal_eval(s)),
                                              'Image diagonal [pixels]': literal_eval,
                                              'Image size [pixels]': literal_eval,
                                              'Source image size': literal_eval,
                                              })
    data = data[data['status']=='training'] # only training landmarks have both src and trg files
    data.reset_index(drop=True, inplace=True)

    for i in range(len(data)):
        src_landmarks = data['Source landmarks'][i]
        trg_landmarks = data['Target landmarks'][i]
        src_landmarks = pd.read_csv(data_path+src_landmarks).to_numpy()[:, 1:]
        trg_landmarks = pd.read_csv(data_path+trg_landmarks).to_numpy()[:, 1:]
        
        if (len(src_landmarks) != len(trg_landmarks)):
            continue
            
        src_size = data['Source image size'][i]
        trg_size = data['Image size [pixels]'][i]
        diag = data['Image diagonal [pixels]'][i]

        src_landmarks_X, src_landmarks_Y, = src_landmarks[0], src_landmarks[1]
        trg_landmarks_X, trg_landmarks_Y, = trg_landmarks[0], trg_landmarks[1]
        
        # resizing landmarks to currently used dataset size
        src_landmarks_X, src_landmarks_Y, diag_src = resample_landmarks(src_landmarks, resample_rate, src_size, trg_size)
        trg_landmarks_X, trg_landmarks_Y, diag_trg = resample_landmarks(trg_landmarks, resample_rate, trg_size, src_size)
        assert np.isclose(diag_src, diag_trg)

        displacement_field = tc_df_to_np_df(deformation_fields[i])
        u_X = displacement_field[0, :, :]
        u_Y = displacement_field[1, :, :]

        ux = nd.map_coordinates(u_X, [src_landmarks_Y, src_landmarks_X], mode='nearest')
        uy = nd.map_coordinates(u_Y, [src_landmarks_Y, src_landmarks_X], mode='nearest')
        output_landmarks = np.stack((src_landmarks_X + ux, src_landmarks_Y + uy), axis=1)
        trg_landmarks = np.stack((trg_landmarks_X, trg_landmarks_Y), axis=1)

        rTRE += [rtre(output_landmarks, trg_landmarks, diag)]

    return rTRE

def rtre(output, target, diag):
    '''
    rTRE calculations
    '''
    return np.mean(np.linalg.norm(output-target, axis=1) / diag)

if __name__ == '__main__':
    from NonRigidNet.nonrigidnet import Nonrigid_Registration_Network

    def test_eval():
        summary_file = 'data/processed/edited_data_v3.csv'
        data_root = 'data/processed/'
        device = 'cpu'
        channels = 2 #6 if transfer is not None else 2
        transfer = None
        base_transforms = [
            Align_subject(15),
            tio.RescaleIntensity(out_min_max=(-1, 1)),
        ]
        params = {'patch_size':(256, 256, 1), 'patch_overlap':(50, 50, 0)}
        AnhirSet = AnhirPatches('data/processed/edited_data_v3.csv', 'data/raw/', base_transforms=base_transforms)
        nonrigid = Nonrigid_Registration_Network(channels).to(device)
        grid, rTRE = evaluate(AnhirSet, nonrigid, transfer, summary_file, device, data_root, 15, **params)

        print(f'{grid=}')
        print(f'{rTRE=}')
        
    print('Testing...')
    test_eval()
    print('Done...✔️')