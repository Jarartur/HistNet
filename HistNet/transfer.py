# %% Imports
# PyTorch
from numpy.lib.utils import source
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision
import torchvision.utils as vutils
from kornia.color import lab_to_rgb
# fancy stuff
import torch.cuda.amp as amp
from torch.optim.swa_utils import AveragedModel, SWALR
# tensorboard
from torch.utils.tensorboard import SummaryWriter

# Additional
from tqdm import trange, tqdm
from dataset import AnhirPatches
import torchio as tio
import numpy as np
import os
import copy

# Local modules
from config import config, base_transforms, data_config, model_config, init_data, lr_config
from NonRigidNet.nonrigidnet import Nonrigid_Registration_Network
from NonRigidNet.nonrigidnet_v2 import RegistrationNetwork
from UNet.unet_model import UNet
from losses import Grad, NCC, diffusion, jcob_det_3, ncc_local, curvature_regularization, mind_loss
from dataset import AnhirPatches
from utils import load_checkpoint, df_field_v2, he_init, get_images, make_grid, moving_average
from losses import ncc_loss_global
from samplev2 import evaluate
# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Check for existing folder with resized data, if not presen make one and populate with resized images from original dataset
init_data(root='raw/', resize=data_config['resample_rate'], summary_path=data_config['main_file_path'], save_root='')

rTRE_list = []

# %% Hyperparameters definitions
lambda_reg_list = [1]
learning_rate_list = [1e-3]
decay_list = [0.995]
batch_size_list = [8]
reg_functions = [diffusion]
cost_functions = [mind_loss]
vecint_list = [None]
lambda_trans_list = [8000]

for lambda_trans in lambda_trans_list:
    for cost in cost_functions:
        for int_num in vecint_list:
            for regularization in reg_functions:
                for lambda_reg in lambda_reg_list:
                    for batch_size in batch_size_list:
                        for learning_rate in learning_rate_list:
                            for decay in decay_list:
                                # Tensorboard logger annotation and model checkpoint name
                                exp_path = f'-cost_{cost.__name__}-vecint_{int_num}-reg_{regularization.__name__}-lr_{learning_rate}-decay_{decay}-bsize_{batch_size}-lmd_reg_{lambda_reg}-lmd_trans_{lambda_trans}'
                                if cost.__name__ == 'ncc_local': loss_type = 'NCC'
                                elif cost.__name__ == 'mind_loss': loss_type = 'MIND'

                                print("Using name:")
                                print(data_config['summary_path']+exp_path)
                                print("Using lr config:")
                                print(lr_config)
                                # Dataset initialization
                                AnhirSet = AnhirPatches(data_config['main_file_path'], data_config['train_root'], base_transforms=base_transforms, color_mode='cielab')
                                dataset = AnhirSet.get_training_subjects()
                                # torchio queue for patch-based pipeline
                                queue = AnhirSet.get_queue(patch_size=(256, 256, 1),
                                                           max_length=batch_size*8*2,
                                                           samples_per_volume=batch_size*8,
                                                           num_workers=4,
                                                           shuffle_subjects=True,
                                                           shuffle_patches=False,
                                                           start_background=True,)
                                training_loader_patches = torch.utils.data.DataLoader(queue, batch_size=batch_size)
                                params = {'patch_size':(256, 256, 1), 'patch_overlap':(20, 20, 0)} # parameters on test-time inference
                                
                                # %% Model init
                                transfer = RegistrationNetwork(input_channels=3, ouput_channels=4).apply(he_init).to(device) # for cielab we use all L*a*b channels but ouput only *a*b
                                nonrigid = None

                                nets = []
                                nets += [transfer]

                                # parameters list for optimizer
                                parameters = []
                                for net in nets:
                                    parameters += list(net.parameters())

                                # optimizer, scheduler and tensorboard logger initialization
                                optimizer = optim.Adam(parameters, learning_rate, betas=model_config['betas'])
                                scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: decay**step)
                                transfer_swa = AveragedModel(transfer)
                                

                                # Logging
                                writer = SummaryWriter(data_config['summary_path']+exp_path)

                                # pretrained model resuming if needed
                                if model_config['resume']:
                                    if data_config['model_checkpoint'] is not None:
                                        epoch_resume = load_checkpoint(data_config['model_checkpoint'], nonrigid, transfer, optimizer, scheduler, model_config=model_config)
                                        print("loaded arbitrary model")
                                    else:
                                        epoch_resume = load_checkpoint(model_config['checkpoint_path']+exp_path+'.tar', nonrigid, transfer, optimizer, scheduler, model_config=model_config)
                                        print("loaded previous model")
                                else: epoch_resume = 0

                                # main trainloop
                                for epoch in trange(epoch_resume, config['epochs']+1, desc='Epochs'):
                                    batch_loss_ncc = []
                                    batch_loss_trans = []
                                    for samples in tqdm(training_loader_patches, desc='Dataloader'):

                                        # loading images, transfering to gpu and adding noise
                                        source_image, style_image = get_images(samples, device, False)

                                        # conditional style transfer
                                        if transfer is not None:
                                            intense_tensor = transfer(source_image, style_image)
                                            ab_image = ((source_image[:, 1:3, :, :] * intense_tensor[:, 0:2, :, :]) + intense_tensor[:, 2:, :, :]) # leave L channel alone

                                        # cielab to rgb shenanigans
                                        lab_source = (source_image[:, 0, :, :] * 100).unsqueeze(1)
                                        ab_image = ((ab_image * 2) - 1) * 127
                                        output_image = torch.cat([lab_source, ab_image], dim=1)
                                        output_image = lab_to_rgb(output_image, clip=True)

                                        ab_source = ((source_image[:, 1:3, :, :] * 2) - 1) * 127
                                        source_image = torch.cat([lab_source, ab_source], dim=1)
                                        source_image = lab_to_rgb(source_image, clip=True)

                                        lab_style = (style_image[:, 0, :, :] * 100).unsqueeze(1)
                                        ab_style = ((style_image[:, 1:3, :, :] * 2) - 1) * 127
                                        style_image = torch.cat([lab_style, ab_style], dim=1)
                                        style_image = lab_to_rgb(style_image, clip=True)

                                        # transfer to grayscale
                                        output_image = TF.rgb_to_grayscale(output_image)
                                        style_image = TF.rgb_to_grayscale(style_image)
                                        source_image = TF.rgb_to_grayscale(source_image)

                                        # loss
                                        loss_ncc = cost(output_image, style_image, device)
                                        if transfer is not None: loss_transfer = lambda_trans*diffusion(intense_tensor, device)
                                        loss = loss_ncc
                                        if transfer is not None: loss = loss + loss_transfer
                                        batch_loss_ncc += [loss_ncc.item()]
                                        if transfer is not None: batch_loss_trans += [loss_transfer.item()]

                                        # gradient calculations
                                        loss.backward()
                                        optimizer.step()
                                        optimizer.zero_grad()


                                    # lr adjustments
                                    if epoch == lr_config['swa_kickin']:
                                        print(f'lr for swa: {scheduler.get_last_lr()}')
                                        swa_scheduler = SWALR(optimizer, swa_lr=scheduler.get_last_lr())
                                        print(f'swa scheduler init, {epoch=}')
                                    elif epoch > lr_config['swa_kickin']:
                                        transfer_swa.update_parameters(nonrigid)
                                        swa_scheduler.step()
                                    elif epoch >= lr_config['decay_kickin']:
                                        scheduler.step()

                                    # tensorboard loging
                                    writer.add_scalar(f"Learning rate", optimizer.param_groups[0]['lr'], global_step=epoch)
                                    writer.add_scalar(f"{loss_type} loss", np.mean(batch_loss_ncc), global_step=epoch)
                                    # writer.add_scalar("Grad loss (scaled)", np.mean(batch_loss_grad), global_step=epoch)
                                    if transfer is not None: writer.add_scalar(f"Trans regularization", np.mean(batch_loss_trans), global_step=epoch)

                                    # tensorboard logging
                                    if epoch % config['sample_every'] == 0:

                                        # image examples and warped grids examples
                                        grid = torch.cat([output_image, source_image, style_image], dim=0)
                                        grid = vutils.make_grid(grid, output_image.shape[0])
                                        writer.add_image("Registration (& transfer) samples", grid, global_step=epoch)
                                        writer.flush()

                                    # model checkpoint
                                    if epoch % config['checkpoint_every'] == 0 and epoch != 0:
                                        torch.save({
                                            'epoch': epoch,
                                            'nonrigid_model_state_dict': nonrigid.state_dict() if nonrigid is not None else False,
                                            'tranfer_model_state_dict': transfer.state_dict() if transfer is not None else False,
                                            'optimizer_state_dict': optimizer.state_dict(),
                                            'scheduler_state_dict': scheduler.state_dict(),
                                            }, model_config['checkpoint_path']+exp_path+'.tar')
                                        print('Saved checkpoint\n')

                                writer.flush()

                                torch.save({
                                            'epoch': epoch,
                                            'transfer_swa_model_state_dict': transfer_swa.state_dict(),
                                            'tranfer_model_state_dict': transfer.state_dict() if transfer is not None else False,
                                            'optimizer_state_dict': optimizer.state_dict(),
                                            'scheduler_state_dict': scheduler.state_dict(),
                                            'swa_scheduler_state_dict': swa_scheduler.state_dict(),
                                            }, model_config['checkpoint_path']+exp_path+'-swa'+'.tar')
                                print('Saved checkpoint\n')
# %%
