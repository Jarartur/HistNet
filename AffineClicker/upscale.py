#%%
from matplotlib import scale
from numpy.lib.arraypad import pad
from numpy.lib.utils import source
import pandas as pd
import cv2
from ast import literal_eval
import numpy as np
import matplotlib.pyplot as plt
import os
#%%
def plot(src, warp, trg):
    '''plot stuff'''
    plt.figure(figsize=(8, 12))
    plt.subplot(3, 1, 1)
    plt.imshow(src)
    plt.title('src')
    plt.subplot(3, 1, 2)
    plt.imshow(warp)
    plt.title('warped')
    plt.subplot(3, 1, 3)
    plt.imshow(trg)
    plt.title('trg')
    plt.show()
    plt.close()

def Pad(img, padColor:int=255):

    if img.shape[0] > img.shape[1]:
        sh, sw = (img.shape[0], img.shape[0])
    else:
        sh, sw =(img.shape[1], img.shape[1])
    h, w = img.shape[:2]
    # sh, sw = size

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
    # scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    paddings = [pad_top, pad_bot, pad_left, pad_right]

    return scaled_img, paddings

def resizeAndPad(img, size=(256, 256), padColor:int=255, **kwargs):
    '''
    Function to resize images keeping the aspect ratio and then adding padding where neccessary to keep specified dimenstions

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
#%%
main_file_path = '../data/processed/edited_data_v2.csv'
root = '../data/raw/'

data = pd.read_csv(main_file_path, converters={'Source image size': lambda s: literal_eval(s),
                                               'Image size [pixels]': lambda s: literal_eval(s),
                                               'Source transformation matrix': lambda s: np.array(literal_eval(s))})

sources = data['Source image']
targets = data['Target image']
affine_transformations = data['Source transformation matrix'].to_numpy()
#%%
index = 0
path_source_img = os.path.join(root, sources[index])
path_target_img = os.path.join(root, targets[index])
source_img = cv2.imread(path_source_img)
target_img = cv2.imread(path_target_img)
#%%
# def pads(img):
#     paddings = resizeAndPad(img)
#     print(f'{paddings=}')
#     paddings = [pad/256 for pad in paddings]
#     paddings[2:] = [int(pad*img.shape[1]) for pad in paddings[2:]]
#     paddings[:2] = [int(pad*img.shape[0]) for pad in paddings[:2]]
#     return paddings
# 
# pads_src = pads(source_img)
# pads_trg = pads(target_img)
#%%
warped_src, pads_src = Pad(source_img)
# warped_trg, pads_trg = Pad(target_img)
#%%
warp_mat = affine_transformations[index].copy()
warp_mat[:, 2] /= 256
warp_mat[0, 2] *= warped_src.shape[1]
warp_mat[1, 2] *= warped_src.shape[0]
#%%
warped_src = cv2.warpAffine(warped_src, warp_mat, 
                            (warped_src.shape[1], warped_src.shape[0]), 
                            borderMode=cv2.BORDER_REPLICATE)
#%%
def unpad(img, paddings):
    return img[:, paddings[2]:-paddings[3], :]
#%%
plot(source_img, unpad(warped_src, pads_src), target_img)

# %%
#%%
def adjust_images(img1, img2, padColor:int=255):
    hight = np.max([img1.shape[0], img2.shape[0]])
    width = np.max([img1.shape[1], img2.shape[1]])
    if len(img1.shape) == 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    #img1
    img1_pads = calculate_paddings(hight, width, img1)
    img2_pads = calculate_paddings(hight, width, img2)

    img1_padded = cv2.copyMakeBorder(img1, borderType=cv2.BORDER_CONSTANT, value=padColor, **img1_pads)
    img2_padded = cv2.copyMakeBorder(img2, borderType=cv2.BORDER_CONSTANT, value=padColor, **img2_pads)

    return img1_padded, img2_padded, img1_pads, img2_pads
    
def calculate_paddings(hight, width, img):
    pad_vert = np.abs(hight - img.shape[0]) / 2
    pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
    pad_horz = np.abs(width - img.shape[1]) / 2
    pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)

    return {'top': pad_top, 'bottom': pad_bot, 'left': pad_left, 'right': pad_right}

def resample(img, size=(256, 256), padColor=255):
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
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
    else: # square image
        new_h, new_w = sh, sw

    # set pad color
    if len(img.shape) == 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    return scaled_img
# %%
scaled_src, scaled_trg, _, _ = adjust_images(source_img, target_img)

warp_mat = affine_transformations[index].copy()
warp_mat[0, 2] = 0
warp_mat[1, 2] = 0
warped_src = cv2.warpAffine(scaled_src, warp_mat, 
                            (scaled_src.shape[1], scaled_src.shape[0]), 
                            borderMode=cv2.BORDER_REPLICATE)

# scaled_src = resample(scaled_src)
# scaled_trg = resample(scaled_trg)
# warped_src = resample(warped_src)

plot(resample(scaled_src), resample(warped_src), resample(scaled_trg))
# %%
