#%%
from ast import literal_eval
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Cursor
import pandas as pd
import numpy as np
import cv2

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

def resample(img, size=256, padColor=255):
    h, w = img.shape[:2]
    sh = size
    sw = size

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
#%%
main_file = '../data/processed/edited_data.csv'
main_data_path = '../data/raw/'
save_path = 'data.csv'
data = pd.read_csv(main_file)
source_image_list = data.iloc[:, 3]
target_image_list = data.iloc[:, 5]
current_image = 452
counter = 0

srcTri = np.zeros([4, 2])
dscTri = np.zeros([4, 2])

source_tri_points = []
target_tri_points = []

#%%
if current_image > 0:
    past_data = pd.read_csv(save_path)
    img_nr = 451

    col = past_data['source_tri_points'][img_nr:].str.replace('\n', '')
    col = col.str.replace('\[  ', '[')
    col = col.str.replace('\[ ', '[')
    col = col.str.replace('[ ]{1,}', ',')

    l1 = [literal_eval(x) for x in past_data['source_tri_points'][:img_nr]]
    l2 = [literal_eval(x) for x in col]
    source_tri_points = l1 + l2
    
    col = past_data['target_tri_points'][img_nr:].str.replace('\n', '')
    col = col.str.replace('\[  ', '[')
    col = col.str.replace('\[ ', '[')
    col = col.str.replace('[ ]{1,}', ',')

    l1 = [literal_eval(x) for x in past_data['target_tri_points'][:img_nr]]
    l2 = [literal_eval(x) for x in col]
    target_tri_points = l1 + l2

    print(f'Resuming from {current_image=}, loaded previous data points...')
# %%
fig = plt.figure(figsize=(16, 9))
plt.subplots_adjust(bottom=0.2)

src_img = (cv2.imread(main_data_path+source_image_list[current_image]))
trg_img = (cv2.imread(main_data_path+target_image_list[current_image]))
src_img, trg_img, _, _ = adjust_images(src_img, trg_img)
src_img = resample(src_img, 512)
trg_img = resample(trg_img, 512)

ax = []
ax.append( fig.add_subplot(1, 2, 1) )
im_source = ax[0].imshow(src_img)
p1, = ax[0].plot(srcTri[:,0], srcTri[:,1], ls='', marker='o')
ann1_src = ax[0].annotate('1', (srcTri[0, 0], srcTri[0, 1]))
ann2_src = ax[0].annotate('2', (srcTri[1, 0], srcTri[1, 1]))
ann3_src = ax[0].annotate('3', (srcTri[2, 0], srcTri[2, 1]))
ann4_src = ax[0].annotate('4', (srcTri[3, 0], srcTri[3, 1]))
ax.append( fig.add_subplot(1, 2, 2) )
im_target = ax[1].imshow(trg_img)
p2, = ax[1].plot(dscTri[:,0], dscTri[:,1], ls='', marker='o')
ann1_trg = ax[1].annotate('1', (dscTri[0, 0], dscTri[0, 1]))
ann2_trg = ax[1].annotate('2', (dscTri[1, 0], dscTri[1, 1]))
ann3_trg = ax[1].annotate('3', (dscTri[2, 0], dscTri[2, 1]))
ann4_trg = ax[1].annotate('4', (dscTri[3, 0], dscTri[3, 1]))
#%%
axButton_next = plt.axes([0.1, 0.1, 0.2, 0.1])
Button_next = Button(ax = axButton_next,
                        label='Next',
                        color='green',
                        hovercolor='teal')

axButton_reset = plt.axes([0.4, 0.1, 0.2, 0.1])
Button_reset = Button(ax = axButton_reset,
                        label='Reset',
                        color='red',
                        hovercolor='tomato')

axButton_left = plt.axes([0.7, 0.1, 0.1, 0.1])
Button_left = Button(ax = axButton_left,
                        label='Switch image left',
                        color='blue',
                        hovercolor='royalblue')
axButton_right = plt.axes([0.8, 0.1, 0.1, 0.1])
Button_right = Button(ax = axButton_right,
                        label='Switch image right',
                        color='blue',
                        hovercolor='royalblue')

Cursor_left = Cursor(ax[0],
                        horizOn=True,
                        vertOn=True,
                        color='green',
                        linewidth=1.0)
Cursor_right = Cursor(ax[1],
                        horizOn=True,
                        vertOn=True,
                        color='blue',
                        linewidth=1.0)
#%%

def save(save_path):
    #TODO: convert to list before saving
    global source_tri_points, target_tri_points
    print('saving...')
    data = pd.DataFrame({'source_tri_points':source_tri_points, 'target_tri_points':target_tri_points})
    data.to_csv(save_path)
    print('done ✔️')

def next(event):
    global current_image, source_tri_points, target_tri_points, srcTri, dscTri, save_path

    print('Setting next...')
    source_tri_points.append(srcTri.copy())
    target_tri_points.append(dscTri.copy())

    current_image += 1
    if current_image >= len(source_image_list):
        save(save_path)
        print(f'ending on file: {current_image=}, saving...')
        return None

    src_img = (cv2.imread(main_data_path+source_image_list[current_image]))
    trg_img = (cv2.imread(main_data_path+target_image_list[current_image]))
    src_img, trg_img, _, _ = adjust_images(src_img, trg_img)
    src_img = resample(src_img, 512)
    trg_img = resample(trg_img, 512)

    # ax[0].cla()
    # ax[1].cla()
    # ax[0].set_xlim(left=0, right=src_img.shape[1])
    # ax[0].set_ylim(bottom=src_img.shape[0], top=0)
    # ax[0].imshow(src_img)
    im_size = (-0.5, src_img.shape[1]-0.5, src_img.shape[0]-0.5, -0.5)
    im_source.set_extent(im_size)
    im_source.set_data(src_img)

    # im_source.autoscale()
    # ax[1].set_xlim(left=0, right=trg_img.shape[1])
    # ax[1].set_ylim(bottom=trg_img.shape[0], top=0)
    # ax[1].imshow(trg_img)
    trg_size = (-0.5, trg_img.shape[1]-0.5, trg_img.shape[0]-0.5, -0.5)
    im_target.set_extent(trg_size)
    im_target.set_data(trg_img)

    # im_target.autoscale()
    plt.draw()
    print('Done ✔️')
    
def reset(event):
    global srcTri, dscTri, counter
    srcTri = np.zeros([4, 2])
    dscTri = np.zeros([4, 2])
    counter = 0
    print('Reseted')
    print()

def onclick(event):
    global counter, srcTri, dscTri, p1, p2, ax, ann1_src, ann2_src, ann3_src, ann4_src, ann1_trg, ann2_trg, ann3_trg, ann4_trg
    if event.inaxes in ax:
        print(f'Got {counter} point')
        x1, y1 = event.xdata, event.ydata
        if counter < 4:
            srcTri[counter, :] = [x1, y1]
        elif counter < 8:
            dscTri[counter-4, :] = [x1, y1]
        
        counter += 1
        if counter == 8: counter = 0

        p1.set_data(srcTri[:,0], srcTri[:,1])
        ann1_src.set_position((srcTri[0, 0], srcTri[0, 1]))
        ann2_src.set_position((srcTri[1, 0], srcTri[1, 1]))
        ann3_src.set_position((srcTri[2, 0], srcTri[2, 1]))
        ann4_src.set_position((srcTri[3, 0], srcTri[3, 1]))
        p2.set_data(dscTri[:,0], dscTri[:,1])
        ann1_trg.set_position((dscTri[0, 0], dscTri[0, 1]))
        ann2_trg.set_position((dscTri[1, 0], dscTri[1, 1]))
        ann3_trg.set_position((dscTri[2, 0], dscTri[2, 1]))
        ann4_trg.set_position((dscTri[3, 0], dscTri[3, 1]))
        
        plt.draw()

def switch_left(event):
    global counter
    counter = 0
    print(f'Switched to left, {counter=}')

def switch_right(event):
    global counter
    counter = 4
    print(f'Switched to rigt, {counter=}')

def on_press(event):
    print()
    print('press', event.key)
    if event.key == 'c':
        print('Current points:')
        print(f'{srcTri=}')
        print(f'{dscTri=}')
    elif event.key == 'x':
        print('All points:')
        print(f'{source_tri_points=}')
        print(f'{target_tri_points=}')
    elif event.key == 'z': save(save_path)
    elif event.key == 'n': next(event)

fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('key_press_event', on_press)
Button_next.on_clicked(next)
Button_reset.on_clicked(reset)
Button_left.on_clicked(switch_left)
Button_right.on_clicked(switch_right)
#%%
plt.show()
print(current_image)
# %%
