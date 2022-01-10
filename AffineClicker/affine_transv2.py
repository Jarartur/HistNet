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
from tqdm import trange
#%% defs
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

def adjust_images_nocopy(img1, img2):
    hight = np.max([img1.shape[0], img2.shape[0]])
    width = np.max([img1.shape[1], img2.shape[1]])
    #img1
    img1_pads = calculate_paddings(hight, width, img1)
    img2_pads = calculate_paddings(hight, width, img2)
    shapes =(hight, width)

    return shapes, img1_pads, img2_pads

def adjust_images_presize(size1, size2):
    hight = np.max([size1[0], size2[0]])
    width = np.max([size1[1], size2[1]])
    shapes =(hight, width)
    #img1
    img1_pads = calculate_paddings_v2(hight, width, size1)
    img2_pads = calculate_paddings_v2(hight, width, size2)

    return shapes, img1_pads, img2_pads

def calculate_paddings(hight, width, img):
    pad_vert = np.abs(hight - img.shape[0]) / 2
    pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
    pad_horz = np.abs(width - img.shape[1]) / 2
    pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)

    return {'top': pad_top, 'bottom': pad_bot, 'left': pad_left, 'right': pad_right}

def calculate_paddings_v2(hight, width, size):
    pad_vert = np.abs(hight - size[0]) / 2
    pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
    pad_horz = np.abs(width - size[1]) / 2
    pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)

    return {'top': pad_top, 'bottom': pad_bot, 'left': pad_left, 'right': pad_right}

def resample(img, size=(512, 512), padColor=255):
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

def resample_nocopy(size, newsize):
    h, w = size[:2]
    sh, sw = newsize

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

    return (new_h, new_w)

def least_squares_transform(primary, secondary, print=False):

    # Pad the data with ones, so that our transformation can do translations too
    n = primary.shape[0]
    pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
    unpad = lambda x: x[:,:-1]
    X = pad(primary)
    Y = pad(secondary)

    # Solve the least squares problem X * A = Y
    # to find our transformation matrix A
    A, res, rank, s = np.linalg.lstsq(X, Y)

    transform = lambda x: unpad(np.dot(pad(x), A))

    if print:
        print("Target:")
        print(secondary)
        print("Result:")
        print(transform(primary))
        print("Max error:", np.abs(secondary - transform(primary)).max())

    A[np.abs(A) < 1e-10] = 0 # set really small values to zero
    return A.T

def depad_landmarks(landmarks, pads):
    ...

def plot(src, warp, trg, i, srcTri, trgTri):
    fig = plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.imshow(src)
    plt.scatter(srcTri[i][:, 0], srcTri[i][:, 1])
    plt.title('src')
    plt.subplot(3, 1, 2)
    plt.imshow(warp)
    plt.title('warped')
    plt.subplot(3, 1, 3)
    plt.imshow(trg)
    plt.scatter(trgTri[i][:, 0], trgTri[i][:, 1])
    plt.title('trg')
    fig.suptitle(f'Image nr: {i}')
    plt.show()
    # plt.close()
#%%
points = 'data.csv'
main_file_path = '../data/processed/edited_data_v2.csv'
data_prepath = '../data/raw/'

points = pd.read_csv(points)
main_file = pd.read_csv(main_file_path)
main_file['Source image size'] = [literal_eval(x) for x in main_file['Source image size']]
main_file['Image size [pixels]'] = [literal_eval(x) for x in main_file['Image size [pixels]']]

col = points['source_tri_points']
src_tri = np.array([literal_eval(x) for x in col], dtype=np.float32)

col = points['target_tri_points']
dst_tri = np.array([literal_eval(x) for x in col], dtype=np.float32)
#%%
assert src_tri.shape[0] == dst_tri.shape[0]
warp_mat = np.zeros([src_tri.shape[0], 3, 3])

for i in range(src_tri.shape[0]):
    warp_mat[i] = least_squares_transform(src_tri[i], dst_tri[i])
#%%
inspect = 480
for i in range(i, i+1):
    src_img = cv2.imread(data_prepath+main_file['Source image'][i])
    trg_img = cv2.imread(data_prepath+main_file['Target image'][i])

    src_img, trg_img, src_pads, trg_pads = adjust_images(src_img, trg_img)
    assert src_img.shape == trg_img.shape
    shapes = src_img.shape
    resample_img = resample(src_img, (512, 512)) # 512 used in main.py
    resample_shape = resample_img.shape

    warp_mat_c = warp_mat[i, :2, :].copy()
    warp_mat_c[0, 2] /= resample_shape[1]
    warp_mat_c[0, 2] *= shapes[1]
    warp_mat_c[1, 2] /= resample_shape[0]
    warp_mat_c[1, 2] *= shapes[0]
    print(warp_mat_c)
    print(f'{shapes=}')
    print(f'{resample_shape=}')

    # t = mat.copy()
    # t[0, 2] *= shapes[1]
    # t[1, 2] *= shapes[0]
    # print(t)

    warp_img = cv2.warpAffine(src_img, warp_mat_c, 
                              (src_img.shape[1], src_img.shape[0]), 
                              borderMode=cv2.BORDER_CONSTANT, 
                              borderValue=(255, 255, 255))
    
    plot(resample(src_img), resample(warp_img), resample(trg_img), i, src_tri, dst_tri)
#%%
warp_mat_list = []
for i in trange(0, warp_mat.shape[0]):
    src_img = cv2.imread(data_prepath+main_file['Source image'][i])
    trg_img = cv2.imread(data_prepath+main_file['Target image'][i])

    hight = np.max([src_img.shape[0], trg_img.shape[0]])
    width = np.max([src_img.shape[1], trg_img.shape[1]])
    shapes = (hight, width)
    resample_shape = resample_nocopy(shapes, (512, 512)) # 512 used in main.py

    mat = warp_mat[i, :2, :].copy()
    mat[0, -1] /= resample_shape[1]
    mat[1, -1] /= resample_shape[0]

    # print(f'{mat=}')

    warp_mat_list += [mat.tolist()]
#%%
main_file['Source transformation matrix'] = warp_mat_list
main_file = main_file.drop('Unnamed: 0', 1)
main_file.to_csv('../data/processed/edited_data_v3.csv')
#%%
path = '../data/processed/edited_data_v3.csv'
main_file = pd.read_csv(path, converters={'Source transformation matrix': lambda s: np.array(literal_eval(s)),
                                          'Image diagonal [pixels]': lambda s: literal_eval(s),
                                          'Image size [pixels]': lambda s: literal_eval(s),
                                          'Source image size': lambda s: literal_eval(s),
                                          })
#%%
for line in trange(0, len(main_file)):
    path_to_landmarks = data_prepath + main_file['Source landmarks'][line]
    src_landmarks = pd.read_csv(path_to_landmarks).to_numpy()[:, 1:]
    source_size = main_file['Source image size'][line]
    target_size = main_file['Image size [pixels]'][line]

    shape, pads_src, pads_trg = adjust_images_presize(source_size, target_size)
    src_landmarks[:, 0] += pads_src['left']
    src_landmarks[:, 1] += pads_src['top']
    src_landmarks = np.hstack((src_landmarks, np.ones((src_landmarks.shape[0], 1))))

    warp_mat = main_file['Source transformation matrix'][line].copy()
    warp_mat[0, 2] *= shape[1]
    warp_mat[1, 2] *= shape[0]
    new_landmarks = src_landmarks @ warp_mat.T
    
    new_landmarks[:, 0] -= pads_trg['left']
    new_landmarks[:, 1] -= pads_trg['top']

    # src_img = cv2.imread(data_prepath+main_file['Source image'][line])
    # trg_img = cv2.imread(data_prepath+main_file['Target image'][line])
    # src_landmarks = pd.read_csv(path_to_landmarks).to_numpy()[:, 1:]
    # path_to_landmarks = data_prepath + main_file['Target landmarks'][line]
    # trg_landmarks = pd.read_csv(path_to_landmarks).to_numpy()[:, 1:]
    # # NOTE: wrong because images need first to bu adjusted()
    # # warp_img = cv2.warpAffine(src_img, warp_mat, 
    # #                           (src_img.shape[1], src_img.shape[0]), 
    # #                           borderMode=cv2.BORDER_CONSTANT, 
    # #                           borderValue=(255, 255, 255))
    
    # plt.figure(figsize=(8, 12))
    # plt.subplot(3, 1, 1)
    # plt.imshow(src_img)
    # plt.scatter(src_landmarks[:, 0], src_landmarks[:, 1])
    # plt.title('src')
    # plt.subplot(3, 1, 2)
    # plt.imshow(trg_img)
    # plt.scatter(new_landmarks[:, 0], new_landmarks[:, 1])
    # plt.title('warp')
    # plt.subplot(3, 1, 3)
    # plt.imshow(trg_img)
    # plt.scatter(trg_landmarks[:, 0], trg_landmarks[:, 1])
    # plt.title('trg')

    
    path = f'../data/submission_manual_affine_v2/ws_{line}.csv'
    ws = pd.DataFrame()
    ws['X'] = new_landmarks[:, 0]
    ws['Y'] = new_landmarks[:, 1]
    ws.to_csv(path)
#%% test
for line in trange(52, len(main_file)):
    src_img = cv2.imread(data_prepath+main_file['Source image'][line])
    trg_img = cv2.imread(data_prepath+main_file['Target image'][line])

    path_to_landmarks = data_prepath + main_file['Source landmarks'][line]
    src_landmarks = pd.read_csv(path_to_landmarks).to_numpy()[:, 1:]
    path_to_landmarks = data_prepath + main_file['Target landmarks'][line]
    trg_landmarks = pd.read_csv(path_to_landmarks).to_numpy()[:, 1:]

    new_landmarks = pd.read_csv(f'C:\\Users\\arjur\\GitHub\\Histology-Style-Transfer-Research\\data\\submission_manual_affine\\ws_{line}.csv').to_numpy()[:, 1:]

    warp_mat = main_file['Source transformation matrix'][line].copy()
    warp_mat[0, 2] *= shape[1]
    warp_mat[1, 2] *= shape[0]
    
    plt.figure(figsize=(8, 12))
    plt.subplot(3, 1, 1)
    plt.imshow(src_img)
    plt.scatter(src_landmarks[:, 0], src_landmarks[:, 1])
    plt.title('src')
    plt.subplot(3, 1, 2)
    plt.imshow(trg_img)
    plt.scatter(new_landmarks[:, 0], new_landmarks[:, 1])
    plt.title('warp')
    plt.subplot(3, 1, 3)
    plt.imshow(trg_img)
    plt.scatter(trg_landmarks[:, 0], trg_landmarks[:, 1])
    plt.title('trg')

    
    # path = f'../data/submission_manual_affine_v2/ws_{i}.csv'
    # ws = pd.DataFrame()
    # ws['X'] = new_landmarks[:, 0]
    # ws['Y'] = new_landmarks[:, 1]
    # ws.to_csv(path)
    break

# %%
