




# ------------------------- #
# Not updated, not relevant #
# ------------------------- #







from ast import literal_eval

from tqdm import tqdm
from sample import sample_eval
from UNet.unet_model import UNet
from AffineSampler.affine_separate_rotation import AffineSampler
from config import data_config, transform

import pandas as pd

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# unet = UNet(n_channels=6, output_channels=6, bilinear=False).to(device)
unet = None
if unet is not None: unet.eval()

main_file = 'processed/edited_data.csv'
data = pd.read_csv(main_file, converters={'Source transformation matrix': lambda s: np.array(ast.literal_eval(s))})
data['Image size [pixels]'] = [literal_eval(x) for x in data['Image size [pixels]']]
data['Source image size'] = [literal_eval(x) for x in data['Source image size']]

data_root = 'processed'
for i, warped_source, t in tqdm(sample_eval(unet, affine_sampler, data, transform, data_root, device)):
    ws = pd.DataFrame()
    ws['X'] = warped_source[:, 0]
    ws['Y'] = warped_source[:, 1]

    path = f'testing/submission/ws_{i}.csv'
    ws.to_csv(path)
    data.loc[i, 'Warped source landmarks'] = f'ws_{i}.csv'
    data.loc[i, 'Execution time [minutes]'] = t/60
    # if i > 20: break # for testing

data.to_csv('testing/submission/registration-results.csv')