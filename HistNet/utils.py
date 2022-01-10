# %% Imports
from typing import Union
import numpy as np
from numpy.lib.arraysetops import isin
from numpy.matrixlib.defmatrix import matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import os
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from ast import literal_eval
import torchio as tio
import shutil
import tqdm
#%%
def resize_dataset(root, factor, summary_path, savepath):
    '''
    Creating folders and resampling images to speed up training.
    When done in train-time takes much longer to make patches of images.
    '''
    # summary_path = r'C:\Users\arjur\GitHub\Histology-Style-Transfer-Research\data\processed\edited_data_v3.csv'
    summary = pd.read_csv(summary_path, converters={'Source transformation matrix': lambda s: np.array(literal_eval(s)),
                                                  'Image diagonal [pixels]': literal_eval,
                                                  'Image size [pixels]': literal_eval,
                                                  'Source image size': literal_eval,
                                                  })
    files_src = set(summary['Source image'].to_list())
    files_trg = set(summary['Target image'].to_list())

    csv_src = set(summary['Source landmarks'].to_list())

    summary = summary[summary['status']=='training']
    summary = summary.reset_index(drop=True)
    csv_trg = set(summary['Target landmarks'].to_list())

    # merging doubles
    csv = csv_src | csv_trg
    files = files_src | files_trg
    f = []
    # getting directories paths
    for file in files:
        folder, _ = os.path.split(file)
        f += [savepath+folder]
    f = set(f)
    # creating directories
    for path in f:
        os.makedirs(path)
    interp = cv2.INTER_AREA # Resampling method
    # reading, resampling and saving images
    for file in (files):
        image = cv2.imread(root+file)
        h, w = image.shape[:2]
        sh, sw = h//factor, w//factor
        resized = cv2.resize(image, (sw, sh), interpolation=interp)
        cv2.imwrite(savepath+file, resized)
    # copying landmarks files
    for landmarks in csv:
        shutil.copyfile(root+landmarks, savepath+landmarks)

def get_images(samples: dict, device, smoothing=False, **params):
    '''
    Returns pytorch tensors with added random noise to prevent NaNs in ncc in white patches.
    Not needed in negatives?
    Optional gaussian smoothing.
    '''
    content_image = samples['src'][tio.DATA].squeeze(-1).to(device)
    style_image = samples['trg'][tio.DATA].squeeze(-1).to(device)

    if smoothing:
        kernel = get_gaussian_kernel(device, **params)
        content_image = kernel(content_image)
        style_image = kernel(style_image)

    content_image =  content_image + 0.00001*torch.randn_like(content_image)
    style_image =  style_image + 0.00001*torch.randn_like(style_image)

    return content_image, style_image

def get_gaussian_kernel(device='cpu', kernel_size=3, sigma=2, channels=3):
    '''
    Taken from DeepHistReg
    '''
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    mean = (kernel_size - 1)/2.
    variance = sigma**2.
    gaussian_kernel = (1./(2.*math.pi*variance)) * torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1) /(2*variance))
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False, padding=int((kernel_size / 2))).to(device)
    gaussian_filter.weight.data = gaussian_kernel.to(device)
    gaussian_filter.weight.requires_grad = False
    return gaussian_filter

def load_checkpoint(checkpoint_path, model_registration, model_transfer, optimizer, scheduler, model_config):
    '''
    General checkpoint loading
    '''
    checkpoint = torch.load(checkpoint_path)

    if model_registration is not None:
        model_registration.load_state_dict(checkpoint['nonrigid_model_state_dict'])
        model_registration.train()
        print('loaded registration model weights')
    if model_transfer is not None:
        model_transfer.load_state_dict(checkpoint['tranfer_model_state_dict'])
        model_transfer.train()
        print('loaded transfer model weights')
    if not model_config['reset_optim']:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print('restored optims')
    else: print('reset/default optims')
    if model_config['reset_epoch']:
        epoch = 0
        print('starting epoch=0')
    else:
        epoch = checkpoint['epoch'] + 1
        print('starting on resumed epoch, treating loading as initialization...')
    return epoch

def moving_average(model, model_test, beta=0.999):
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)
# %%
# def flow_field(n_grid):

#     vutils.make_grid

# def heatmap(x: torch.Tensor) -> torch.Tensor:
#     assert x.dim() == 2

#     # We're expanding to create one more dimension, for mult. to work.
#     xt = x.expand((3, x.shape[0], x.shape[1])).permute(1, 2, 0)
    
#     # this part is the mask: (xt >= 0) * (xt < 0.5) ...
#     # ... the rest is the original function translated
#     color_tensor = (
#         (xt >= 0) * (xt < 0.5) * ((1 - xt * 2) * torch.tensor([0.9686, 0.9686, 0.9686]) + xt * 2 * torch.tensor([0.9569, 0.6471, 0.5098]))
#         +
#         (xt >= 0) * (xt >= 0.5) * ((1 - (xt - 0.5) * 2) * torch.tensor([0.9569, 0.6471, 0.5098]) + (xt - 0.5) * 2 * torch.tensor([0.7922, 0.0000, 0.1255]))
#         +
#         (xt < 0) * (xt > -0.5) * ((1 - (-xt * 2)) * torch.tensor([0.9686, 0.9686, 0.9686]) + (-xt * 2) * torch.tensor([0.5725, 0.7725, 0.8706]))
#         +
#         (xt < 0) * (xt <= -0.5) * ((1 - (-xt - 0.5) * 2) * torch.tensor([0.5725, 0.7725, 0.8706]) + (-xt - 0.5) * 2 * torch.tensor([0.0196, 0.4431, 0.6902]))
#     ).permute(2, 0, 1)
    
#     return color_tensor

def warp_checks(n_grid, shape, device, **params):
    '''
    Create and warp grids by displacement fields from model
    '''
    checks = make_checks(shape, device, **params)
    warped_checks = F.grid_sample(checks, n_grid, mode='bilinear', padding_mode='zeros', align_corners=False)

    # grid = vutils.make_grid(warped_checks, warped_checks.shape[0])
    return warped_checks

def make_checks(shape, device, num_stripes=16, width=2):
    '''
    Creates grid for warping with displacement fields
    '''
    width = width // 2
    t = torch.zeros(shape, device=device)
    dimy = shape[-1] // num_stripes
    dimx = shape[-2] // num_stripes
    for i in range(num_stripes-1):
        ax = dimx*(i+1)
        ay = dimy*(i+1)
        t[..., ax-width:ax+width, :] = 1
        t[..., :, ay-width:ay+width] = 1
    return t

def make_grid(outputs: list, rows, srcs: list, trgs: list, src_cp=None, **params):
    '''
    Input
    -----
    outputs, srcs, trgs: list | torch.Tensor
        list of [1, 1, H, W] tensors
        if torch.Tensor then they have to have the same dimensions

    Output
    ------
        1st row: warped grids
        2nd row: outputs
        3rd row: sources
        4th row: targets
    '''
    # case for train-time inference when all patches have the same size
    if isinstance(outputs, torch.Tensor) and isinstance(srcs, torch.Tensor) and isinstance(trgs, torch.Tensor):
        
        checks = warp_checks(n_grid=params['n_grid'], shape=outputs.shape, device=params['device'])
        if src_cp is not None:
            outputs = F.grid_sample(src_cp, params['n_grid'], mode='bilinear', padding_mode='zeros', align_corners=False)
            grid = torch.cat([checks, outputs, srcs, trgs], dim=0)
        else:
            grid = torch.cat([checks, outputs, srcs, trgs], dim=0)
        grid = vutils.make_grid(grid, rows)

    # case for test-time inference where images have different sizes
    elif isinstance(outputs, list) and isinstance(srcs, list) and isinstance(trgs, list):
        outs = []
        s = []
        t = []
        
        x_sizes = []
        y_sizes = []
        for output in outputs:
            x_sizes += [output.shape[-2]]
            y_sizes += [output.shape[-1]]
        x_size = max(x_sizes)
        y_size = max(y_sizes)

        for output, src, trg in zip(outputs, srcs, trgs):
            pad_x = np.round(x_size-output.shape[-2]).astype(int)
            pad_x = pad_x / 2
            pad_x_1, pad_x_2 = np.floor(pad_x).astype(int), np.ceil(pad_x).astype(int)

            pad_y = np.round(y_size-output.shape[-1]).astype(int)
            pad_y = pad_y / 2
            pad_y_1, pad_y_2 = np.floor(pad_y).astype(int), np.ceil(pad_y).astype(int)

            pad = (pad_y_1, pad_y_2, pad_x_1, pad_x_2)
            outs += [F.pad(output, pad)]
            s += [F.pad(src, pad)]
            t += [F.pad(trg, pad)]
        
        outs = torch.cat(outs)
        srcs = torch.cat(s)
        trgs = torch.cat(t)
        grid = torch.cat([srcs, outs, trgs], dim=0)
        grid = vutils.make_grid(grid, rows)

    return grid

class Align_subject():
    '''
    Initial transformation for preprocessing.
    Aligns src and trg images by affine matrix from summary_file.
    It is apllied before the images are patchified.
    '''
    def __init__(self, resample_rate=15, color_mode='bgr', whole_mode_size=None) -> None:
        self.resample_rate = resample_rate
        self.colorspace = color_mode
        self.whole_mode = whole_mode_size
    
    def __call__(self, data):
        return self.align_subject(data, self.resample_rate, self.colorspace, whole_mode_size=self.whole_mode)

    @staticmethod
    def align_subject(subject, resample_rate=15, color_mode='bgr', whole_mode_size=None):
        '''1st Transform for tio.Compose'''
        if color_mode == 'bgr' or color_mode == 'rgb':
            padColor=255
        elif color_mode == 'cielab':
            padColor=[255,128,128] # hard-coded padding color
        # getting data from torchio subject (which we get from iterating over SubjectDataset)
        src_img = subject['src'][tio.DATA].squeeze(3).permute(1, 2, 0).numpy()
        trg_img = subject['trg'][tio.DATA].squeeze(3).permute(1, 2, 0).numpy()
#         shape = subject['shape'] #NOTE when using full size dataset it will be a fail safe

        if len(src_img.shape) == 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
            padColor = [padColor]*3
        
        warp_mat = subject['trans_mat']

        # padding images to the same dimensions
        src_img, trg_img, _, _ = adjust_images(src_img, trg_img, padColor)
        shape = src_img.shape
        assert tuple(shape) == src_img.shape == trg_img.shape

        # additional resizing
        if resample_rate != 1:
            src_img = resample(src_img, resample_rate)
            trg_img = resample(trg_img, resample_rate)
        assert src_img.shape == trg_img.shape
        shape = src_img.shape

        # affine warping
        warp_mat[0, 2] *= shape[1]
        warp_mat[1, 2] *= shape[0]
        warp_img = cv2.warpAffine(src_img, warp_mat, 
                                (src_img.shape[1], src_img.shape[0]), 
                                borderMode=cv2.BORDER_CONSTANT, 
                                borderValue=padColor)

        if whole_mode_size is not None:
            warp_img = resizeAndPad(warp_img, whole_mode_size, padColor) #TOCHANGE not ideal way
            trg_img = resizeAndPad(trg_img, whole_mode_size, padColor)

        # dummy singleton dimension (torchio works on 3D images by default)
        src_img = warp_img.transpose(2, 0, 1)[..., np.newaxis]
        trg_img = trg_img.transpose(2, 0, 1)[..., np.newaxis]
        
        subject['src'].set_data(src_img)
        subject['trg'].set_data(trg_img)

        return subject

def resample_landmarks(landmarks, resample_rate, size_1, size_2):
    '''
    Scales landmarks to currently used dataset size
    '''
    hight = np.max([size_1[0], size_2[0]])
    width = np.max([size_1[1], size_2[1]])
    img_1_pads = calculate_paddings(hight, width, size_1)

    padded_landmarks_X = (landmarks[:, 0] + img_1_pads['left']) / resample_rate
    padded_landmarks_Y = (landmarks[:, 1] + img_1_pads['top']) / resample_rate

    diag = np.sqrt((size_1[0] + img_1_pads['top'] + img_1_pads['bottom'])**2 + (size_1[1] + img_1_pads['left'] + img_1_pads['right'])**2)

    return padded_landmarks_X, padded_landmarks_Y, diag

def adjust_images(img1, img2, padColor:int=255):
    '''
    Pads images to the same size
    '''
    hight = np.max([img1.shape[0], img2.shape[0]])
    width = np.max([img1.shape[1], img2.shape[1]])
    if len(img1.shape) == 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    #img1
    img1_pads = calculate_paddings(hight, width, img1.shape)
    img2_pads = calculate_paddings(hight, width, img2.shape)

    img1_padded = cv2.copyMakeBorder(img1, borderType=cv2.BORDER_CONSTANT, value=padColor, **img1_pads)
    img2_padded = cv2.copyMakeBorder(img2, borderType=cv2.BORDER_CONSTANT, value=padColor, **img2_pads)

    return img1_padded, img2_padded, img1_pads, img2_pads

def calculate_paddings(hight, width, shape):
    pad_vert = np.abs(hight - shape[0]) / 2
    pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
    pad_horz = np.abs(width - shape[1]) / 2
    pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)

    return {'top': pad_top, 'bottom': pad_bot, 'left': pad_left, 'right': pad_right}

def resample(img, factor: Union[int, float]=2, padColor=255):
    '''
    Resamples image by a given factor so aspect ratio stays the same
    '''
    h, w = img.shape[:2]
    sh, sw = h//factor, w//factor #NOTE: will keep original aspect ratio

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC

    # set pad color
    if len(img.shape) == 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (sw, sh), interpolation=interp)
    return scaled_img

def resizeAndPad(img, size=(256, 256), padColor=255, **kwargs):
    '''
    - Currently not in use -

    Function to resize images keeping the aspect ratio and then adding padding where neccessary to keep specified dimenstions
    Adapted from StackOverflow (have link somewhere)

    Examples
    --------
    ```
    v_img = cv2.imread('v.jpg') # vertical image
    scaled_v_img = resizeAndPad(v_img, (200,200), 127)

    h_img = cv2.imread('h.jpg') # horizontal image
    scaled_h_img = resizeAndPad(h_img, (200,200), 127)

    sq_img = cv2.imread('sq.jpg') # square image
    scaled_sq_img = resizeAndPad(sq_img, (200,200), 127)
    ```
    '''
    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) == 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img

def resizeAndPad_landmarks(org_size, target_size, landmarks):
    '''
    - Currently not in use -
    '''

    h, w = org_size
    sh, sw = target_size

    aspect = w/h

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    scalex = new_h/h
    scaley = new_w/w

    padded_landmarks_X = (landmarks[:, 0] * scalex) + pad_left
    padded_landmarks_Y = (landmarks[:, 1] * scaley) + pad_top

    scales = [scalex, scaley]
    padding = [pad_left, pad_top]

    return padded_landmarks_X, padded_landmarks_Y, scales, padding

def descale_landmarks(scales, paddings, landmarks):

    padded_landmarks_X = (landmarks[:, 0]-paddings[0]) / scales[0]
    padded_landmarks_Y = (landmarks[:, 1]-paddings[1]) / scales[1]
    new_landmarks = np.stack((padded_landmarks_X, padded_landmarks_Y), axis=1)
    return new_landmarks

def denorm(img_lab):
    '''
    - Currently not in use -

    used to denormalize images from CIEALB colorspace to RGB
    '''
    img_lab_norm = img_lab.permute(1,2,0).cpu().numpy()
    img_lab_norm2 = (((img_lab_norm + 1) / 2)*255).astype(np.uint8)
    img_rec_norm = cv2.cvtColor(img_lab_norm2, cv2.COLOR_Lab2RGB)
    return img_rec_norm

def getFiles(dirName:str) -> list:
    '''
    - Currently not used -

    Function to get a list of files from a directory and all subdirectories
    Discards .csv extension
    ### Example:
    ```
    listOfFiles = getFiles(dirName)
    ```
    '''
    listOfFile = os.listdir(dirName)
    completeFileList = list()
    for file in listOfFile:
        completePath = os.path.join(dirName, file)
        if os.path.isdir(completePath):
            completeFileList = completeFileList + getFiles(completePath)
        else:
            completeFileList.append(completePath)

    for file in completeFileList.copy():
      _, extension = os.path.splitext(file)
      if extension == '.csv':
        completeFileList.remove(file)

    return completeFileList

# %% Plotting
def sample_plot(source, output, target, path, epoch: int):
    '''
    - Legacy function -
    Now all plotting is handled by tensorboard logger.

    '''
    size = 4
    loop = 1
    plt.figure()
    for i in range(size):
        c = source[i].detach().cpu().squeeze(0).numpy()
        x = output[i].detach().cpu().squeeze(0).numpy()
        s = target[i].detach().cpu().squeeze(0).numpy()
        plt.subplot(size, 3, loop)
        loop +=1
        plt.imshow(x, cmap='gray')
        plt.title(f'output for epoch: {epoch}')
        plt.axis('off')
        plt.subplot(size, 3, loop)
        loop +=1
        plt.imshow(s, cmap='gray')
        plt.title('target')
        plt.axis('off')
        plt.subplot(size, 3, loop)
        loop +=1
        plt.imshow(c, cmap='gray')
        plt.title('source')
        plt.axis('off')

    plt.savefig(os.path.join(path, f'sample_{epoch}.png'))
    plt.close()

def plot_losses(losses, path, epoch, name):
    '''
    Legacy, handled by tensorboard
    '''
    plt.figure()
    plt.scatter([0], [0], c='white')
    plt.plot(losses)
    plt.title(f'{name} | epoch: {epoch}')
    plt.savefig(os.path.join(path, f'loss_{name}.png'))
    plt.close()

# %% Init
def he_init(module):
    '''
    Kiming initialization adapted from StarGAN
    '''
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

def to_csv(lists, names, path):
    '''
    Not used
    '''
    d = {}
    for name, list in zip(names, lists):
        d[name] = list
    data = pd.DataFrame(d)
    data.to_csv(path)

def reset_grads(optims):
    '''
    Not used, legacy
    '''
    for optim in optims:
        optim.zero_grad()
# %% Helpers
def generate_grid_tc(tensor_size: torch.Tensor, device="cpu"):
    '''
    © Marek Wodziński
    '''
    identity_transform = torch.eye(len(tensor_size)-1, device=device)[:-1, :].unsqueeze(0)
    identity_transform = torch.repeat_interleave(identity_transform, tensor_size[0], dim=0)
    grid = F.affine_grid(identity_transform, tensor_size, align_corners=False)
    return grid

def tc_transform_to_tc_df(transformation: torch.Tensor, size: torch.Size, device: str="cpu"):
    '''
    © Marek Wodziński
    '''
    deformation_field = F.affine_grid(transformation, size=size, align_corners=False).to(device)
    size = (deformation_field.size(0), 1) + deformation_field.size()[1:-1]
    grid = generate_grid_tc(size, device=device)
    displacement_field = deformation_field - grid
    return displacement_field

def tc_df_to_np_df(displacement_field_tc: torch.Tensor):
    '''
    © Marek Wodziński
    '''
    ndim = len(displacement_field_tc.size()) - 2
    if ndim == 2:
        displacement_field_np = displacement_field_tc.detach().cpu()[0].permute(2, 0, 1).numpy()
        shape = displacement_field_np.shape
        temp_df_copy = displacement_field_np.copy()
        displacement_field_np[0, :, :] = temp_df_copy[0, :, :] / 2.0 * (shape[2])
        displacement_field_np[1, :, :] = temp_df_copy[1, :, :] / 2.0 * (shape[1])
    elif ndim == 3:
        displacement_field_np = displacement_field_tc.detach().cpu()[0].permute(3, 0, 1, 2).numpy()
        shape = displacement_field_np.shape
        temp_df_copy = displacement_field_np.copy()
        displacement_field_np[0, :, :, :] = temp_df_copy[1, :, :, :] / 2.0 * (shape[2])
        displacement_field_np[1, :, :, :] = temp_df_copy[2, :, :, :] / 2.0 * (shape[1])
        displacement_field_np[2, :, :, :] = temp_df_copy[0, :, :, :] / 2.0 * (shape[3])
    return displacement_field_np

def scale_patch(df, locations):
    '''Scale to numpy format'''
    i_ini, j_ini, _, i_fin, j_fin, _ = locations
    df_cp = torch.zeros_like(df)
    df_cp[0, 0, ...] = scale_linear(df[0, 0, ...], j_ini, j_fin)
    df_cp[0, 1, ...] = scale_linear(df[0, 1, ...], i_ini, i_fin)

    return df_cp

def rescale_to_tc_df(df, shape):
    '''Rescale back to pytorch format'''
    i_ini, j_ini, i_fin, j_fin = 0, 0, shape[0], shape[1]
    df[0, ..., 0] = rescale_linear(df[0, ..., 0], j_ini, j_fin)
    df[0, ..., 1] = rescale_linear(df[0, ..., 1], i_ini, i_fin)

    return df

def scale_linear(array, new_min, new_max):
    """Scale to numpy format."""
    minimum, maximum = -1, 1
    m = (new_max - new_min) / (maximum - minimum)
    b = new_min - m * minimum
    return m * array + b

def rescale_linear(array, minimum, maximum):
    """Rescale back to pytorch format."""
    new_min, new_max = -1, 1
    m = (new_max - new_min) / (maximum - minimum)
    b = new_min - m * minimum
    return m * array + b

def df_field(tensors, displacement_fields, compat=None, device="cpu"):
    '''
    © Marek Wodziński
    '''
    size = tensors.size()
    no_samples = size[0]
    x_size = size[3]
    y_size = size[2]
    gy, gx = torch.meshgrid(torch.arange(y_size), torch.arange(x_size))
    gy = gy.type(torch.FloatTensor).to(device)
    gx = gx.type(torch.FloatTensor).to(device)
    grid_x = (gx / (x_size - 1) - 0.5)*2
    grid_y = (gy / (y_size - 1) - 0.5)*2
    n_grid_x = grid_x.view(1, -1).repeat(no_samples, 1).view(-1, grid_x.size(0), grid_x.size(1))
    n_grid_y = grid_y.view(1, -1).repeat(no_samples, 1).view(-1, grid_y.size(0), grid_y.size(1))
    n_grid = torch.stack((n_grid_x, n_grid_y), dim=3)
    # displacement_fields = displacement_fields.permute(0, 2, 3, 1)
    u_x = displacement_fields[:, :, :, 0]
    u_y = displacement_fields[:, :, :, 1]
    u_x = u_x / (x_size - 1) * 2
    u_y = u_y / (y_size - 1) * 2
    n_grid[:, :, :, 0] = n_grid[:, :, :, 0] + u_x
    n_grid[:, :, :, 1] = n_grid[:, :, :, 1] + u_y
#     transformed_tensors = F.grid_sample(tensors, n_grid, mode='bilinear', padding_mode='zeros')
    return n_grid

def df_field_v2(tensor: torch.Tensor, displacement_field: torch.Tensor, grid: torch.Tensor=None, device: str="cpu", mode: str='bilinear'):
    """
    © Marek Wodziński
    Adapted to return displacement fileds

    Transforms a tensor with a given displacement field.
    Uses F.grid_sample for the structured data interpolation (only linear and nearest supported).
    Be careful - the autogradient calculation is possible only with mode set to "bilinear".

    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to be transformed (BxYxXxZxD)
    displacement_field : torch.Tensor
        The PyTorch displacement field (BxYxXxZxD)
    grid : torch.Tensor (optional)
        The input identity grid (optional - may be provided to speed-up the calculation for iterative algorithms)
    device : str
        The device used for warping (e.g. "cpu" or "cuda:0")
    mode : str
        The interpolation mode ("bilinear" or "nearest")

    Returns
    ----------
    transformed_tensor : torch.Tensor
        The transformed tensor (BxYxXxZxD)
    """
    if grid is None:
        grid = generate_grid(tensor.size(), device=device)
    sampling_grid = grid + displacement_field
    # transformed_tensor = F.grid_sample(tensor, sampling_grid, mode=mode, padding_mode='zeros', align_corners=False)
    return sampling_grid

def generate_grid(tensor_size: torch.Tensor, device="cpu"):
    """
    © Marek Wodziński
    Generates the identity grid for a given tensor size.

    Parameters
    ----------
    tensor_size : torch.Tensor or torch.Size
        The tensor size used to generate the regular grid
    device : str
        The device used for resampling (e.g. "cpu" or "cuda:0")
    
    Returns
    ----------
    grid : torch.Tensor
        The regular grid (relative for warp_tensor with align_corners=False)
    """
    identity_transform = torch.eye(len(tensor_size)-1, device=device)[:-1, :].unsqueeze(0)
    identity_transform = torch.repeat_interleave(identity_transform, tensor_size[0], dim=0)
    grid = F.affine_grid(identity_transform, tensor_size, align_corners=False)
    return grid