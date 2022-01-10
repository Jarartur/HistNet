#%%
import pandas as pd
import cv2
from ast import literal_eval
import numpy as np
import matplotlib.pyplot as plt
import ast
import re
from tqdm import trange
import numpy.linalg as LA
#%%
def resizeAndPad_landmarks(org_size, target_size, landmarks):

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

    # print(f'new_h: {new_h}')
    # print(f'new_w: {new_w}')

    scales = [scalex, scaley]
    padding = [pad_left, pad_top]

    if landmarks is not None:
        padded_landmarks_X = (landmarks[:, 0] * scalex) + pad_left
        padded_landmarks_Y = (landmarks[:, 1] * scaley) + pad_top
        return padded_landmarks_X, padded_landmarks_Y, scales, padding

    return scales, padding

def descale_landmarks(scales, paddings, landmarks):

    padded_landmarks_X = (landmarks[:, 0]-paddings[0]) / scales[0]
    padded_landmarks_Y = (landmarks[:, 1]-paddings[1]) / scales[1]
    new_landmarks = np.stack((padded_landmarks_X, padded_landmarks_Y), axis=1)
    return new_landmarks

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
#%%
save_path = 'data.csv'
main_file_path = '../data/processed/edited_data.csv'
data_prepath = '../data/processed/'

past_data = pd.read_csv(save_path)
main_file = pd.read_csv(main_file_path)
main_file['Source image size'] = [literal_eval(x) for x in main_file['Source image size']]
main_file['Image size [pixels]'] = [literal_eval(x) for x in main_file['Image size [pixels]']]

col = past_data['source_tri_points']
src_tri = np.array([literal_eval(x) for x in col], dtype=np.float32)

col = past_data['target_tri_points']
dst_tri = np.array([literal_eval(x) for x in col], dtype=np.float32)
# %%
assert src_tri.shape[0] == dst_tri.shape[0]
warp_mat = np.zeros([src_tri.shape[0], 3, 3])

for i in range(src_tri.shape[0]):
    warp_mat[i] = least_squares_transform(src_tri[i], dst_tri[i])

# %%

for line in trange(0, len(main_file)):
    i = line
    path_to_landmarks = data_prepath + main_file['Source landmarks'][line]
    src_landmarks = pd.read_csv(path_to_landmarks).to_numpy()[:, 1:]
    source_size = main_file['Source image size'][line]
    target_size = main_file['Image size [pixels]'][line]
    source_scaled_landmarks_X, source_scaled_landmarks_Y, scales_source, paddings_source = resizeAndPad_landmarks(source_size, [256, 256], src_landmarks)
    scales_target, paddings_target = resizeAndPad_landmarks(target_size, [256, 256], None)

    source_scaled_landmarks = np.stack((source_scaled_landmarks_X, source_scaled_landmarks_Y), axis=1)
    source_scaled_landmarks = np.hstack((source_scaled_landmarks, np.ones((source_scaled_landmarks.shape[0], 1))))
    
    new_landmarks = source_scaled_landmarks @ warp_mat[line].T
    descaled_new_landmarks = descale_landmarks(scales_target, paddings_target, new_landmarks)
    
    path = f'../data/submission_manual_affine/ws_{i}.csv'
    ws = pd.DataFrame()
    ws['X'] = descaled_new_landmarks[:, 0]
    ws['Y'] = descaled_new_landmarks[:, 1]
    ws.to_csv(path)

# %%
for i in range(4, 5):
    src_img = resizeAndPad(cv2.imread(data_prepath+main_file['Source image'][i]))
    trg_img = resizeAndPad(cv2.imread(data_prepath+main_file['Target image'][i]))

    warp_dst = cv2.warpAffine(src_img, warp_mat[i, :2, :], 
                                    (src_img.shape[1], src_img.shape[0]), 
                                    borderMode=cv2.BORDER_CONSTANT, 
                                    borderValue=(255, 255, 255))
    
    print()
    print(f'#-------------- img nr: {i} --------------#')
    print(warp_mat[i])
    print(f'#-----------------------------------------#')
    print()
    
    fig = plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.imshow(src_img)
    plt.scatter(src_tri[i][:, 0], src_tri[i][:, 1])
    plt.title('source')
    plt.subplot(3, 1, 2)
    plt.imshow(warp_dst)
    plt.title('warped')
    plt.subplot(3, 1, 3)
    plt.imshow(trg_img)
    plt.scatter(dst_tri[i][:, 0], dst_tri[i][:, 1])
    plt.title('target')
    fig.suptitle(f'Image nr: {i}')
    plt.show()
    plt.close()

    print('#-------------- now the inverse ---------------#')

    inv_mat = LA.inv(warp_mat[i, :, :])
    warp_dst = cv2.warpAffine(trg_img, inv_mat[:2, :], 
                                    (src_img.shape[1], src_img.shape[0]), 
                                    borderMode=cv2.BORDER_CONSTANT, 
                                    borderValue=(255, 255, 255))

    fig = plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.imshow(trg_img)
    plt.scatter(dst_tri[i][:, 0], dst_tri[i][:, 1])
    plt.title('trg')
    plt.subplot(3, 1, 2)
    plt.imshow(warp_dst)
    plt.title('warped')
    plt.subplot(3, 1, 3)
    plt.imshow(src_img)
    plt.scatter(src_tri[i][:, 0], src_tri[i][:, 1])
    plt.title('src')
    fig.suptitle(f'Image nr: {i}')
    plt.show()
    plt.close()
    break
# %%
warp_mat_list = []
for i in range(0, warp_mat.shape[0]):
    mat = warp_mat[i, :2, :]
    mat[:, -1] = mat[:, -1]
    # print(mat)
    warp_mat_list += [mat.tolist()]
#%%
main_file['Source transformation matrix'] = warp_mat_list
# %%
main_file = main_file.drop('Unnamed: 0', 1)
main_file.to_csv('../data/processed/edited_data_v2.csv')
# %%
t = pd.read_csv('../data/processed/edited_data_v2.csv', converters={'Source transformation matrix': lambda s: np.array(ast.literal_eval(s))})
