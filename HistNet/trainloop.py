




# -------------------------------------------------- #
# Currently not updated, everything is in hparams.py #
# -------------------------------------------------- #





# %% Imports
# PyTorch
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision
# mixed precision
import torch.cuda.amp as amp
# tensorboard
from torch.utils.tensorboard import SummaryWriter

# Additional
from tqdm import trange, tqdm
from dataset import AnhirPatches
import torchio as tio
import numpy as np
import os

# Local modules
from config import config, base_transforms, data_config, model_config, init_data
from NonRigidNet.nonrigidnet import Nonrigid_Registration_Network
from UNet.unet_model import UNet
from losses import Grad, NCC, diffusion, ncc_local
from dataset import AnhirPatches
from utils import load_checkpoint, df_field_v2, he_init, get_images, make_grid, df_field, warp_checks
from losses import ncc_loss_global
from samplev2 import evaluate
# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
if os.path.isdir(data_config['summary_path']):
    print('Reminder to change experiment name, aborting...')
    quit()
init_data(root='raw/', resize=data_config['resample_rate'], summary_path=data_config['main_file_path'], save_root='')
# %% Dataset
AnhirSet = AnhirPatches(data_config['main_file_path'], data_config['train_root'], base_transforms=base_transforms)
dataset = AnhirSet.get_training_subjects()
queue = AnhirSet.get_queue(patch_size=(256, 256, 1),
                           max_length=data_config['batch_size']*8,
                           samples_per_volume=data_config['batch_size']//4,
                           num_workers=4,
                           shuffle_subjects=True,
                           shuffle_patches=True,
                           start_background=True,)
training_loader_patches = torch.utils.data.DataLoader(queue, batch_size=data_config['batch_size'])
params = {'patch_size':(256, 256, 1), 'patch_overlap':(50, 50, 0)}
# %% Model init
# transfer = UNet(6, 6).to(device).apply(he_init)
transfer = None
channels = 2 #6 if transfer is not None else 2
nonrigid = Nonrigid_Registration_Network(channels).to(device)#.apply(he_init)
nets = []
nets += [nonrigid]
if transfer is not None: nets += [transfer]

# l = NCC(device) # local ncc
# cost_function = l.loss
# cost_function = ncc_loss_global # global ncc
# grad_loss = Grad()

parameters = []
for net in nets:
    parameters += list(net.parameters())

optimizer = optim.Adam(parameters, model_config['learning_rate'], betas=model_config['betas'])
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: model_config['decay_rate']**epoch)
scaler = amp.GradScaler()

if model_config['resume']: epoch_resume = load_checkpoint(model_config['checkpoint_path'], nonrigid, transfer, optimizer, scheduler, scaler)
else: epoch_resume = 0
print(f'lr: {scheduler.get_last_lr()}')
print(f'{epoch_resume=}')

writer = SummaryWriter(data_config['summary_path'])
# %% Lōōp
step = 0
# loss_list = []
rTRE_list = []

for epoch in trange(epoch_resume, config['epochs']+1, desc='Epochs'):
    batch_loss_ncc = []
    batch_loss_grad = []
    for samples in tqdm(training_loader_patches, desc='Dataloader'):
        content_image, style_image = get_images(samples, device)
        
        # with amp.autocast(enabled=False):
        if transfer is not None:
            intense_tensor = transfer(content_image, style_image)
            content_image = ((content_image * intense_tensor[:, 0:3, :, :]) + intense_tensor[:, 3:, :, :])

        content_image = TF.rgb_to_grayscale(content_image)
        style_image = TF.rgb_to_grayscale(style_image)

        content_image = 1 - content_image
        style_image = 1 - style_image
        
        # non rigid transform
        deformation_field = nonrigid(style_image, content_image)
        output_image, n_grid = df_field_v2(content_image, deformation_field, None, device)
        # loss
        loss_ncc = ncc_local(output_image[:, :, 50:-50, 50:-50], style_image[:, :, 50:-50, 50:-50], device, win_size=9) #NOTE: 20 is to not calculate ncc over overlapping regions
        loss_grad = config['lmbd_grad_reg']*diffusion(deformation_field)
        loss = loss_ncc + loss_grad
        batch_loss_ncc += [loss_ncc.item()]
        batch_loss_grad += [loss_grad.item()]
        
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # lr adjustments
    scheduler.step()
    writer.add_scalar("NCC loss", np.mean(batch_loss_ncc), global_step=epoch)
    writer.add_scalar("Grad loss (scaled)", np.mean(batch_loss_grad), global_step=epoch)
    
    if epoch % config['sample_every'] == 0:
        content_image = 1 - content_image
        style_image = 1 - style_image
        output_image = 1 - output_image

        checks = warp_checks(n_grid, output_image.shape, device)
        img_grid = make_grid(output_image, output_image.shape[0], content_image, style_image)
        writer.add_image("Registration (& transfer) samples", img_grid, global_step=epoch)
        writer.add_image("Registration (& transfer) grids", checks, global_step=epoch)
    
    if epoch % config['checkpoint_every'] == 0 and epoch != 0:
        torch.save({
            'epoch': epoch,
            'nonrigid_model_state_dict': nonrigid.state_dict(),
            'tranfer_model_state_dict': transfer.state_dict() if transfer is not None else False,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            }, model_config['checkpoint_path'])
        print('Saved checkpoint\n')
        
    if epoch % config['test_every'] == 0 and epoch != 0:
        grid, rTRE = evaluate(AnhirSet, nonrigid, transfer, data_config['main_file_path'], device, data_config['train_root'], data_config['resample_rate'], **params) #NOTE: *4 for reduced dataset
        rTRE_list += [rTRE]
        print(f'{rTRE=}')
        writer.add_scalar("rTRE metric", rTRE, global_step=epoch)
        writer.add_image("Registration (& transfer) test", grid.float(), global_step=epoch)