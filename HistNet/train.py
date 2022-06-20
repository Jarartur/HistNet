if __name__ == '__main__':
    # %% Imports
    # PyTorch
    import torch
    from torch.utils.data import DataLoader
    import torch.optim as optim
    import torch.nn.functional as F
    import torchvision.transforms.functional as TF
    import torchvision
    # fancy stuff
    import torch.cuda.amp as amp
    from torch.optim.swa_utils import AveragedModel, SWALR
    # tensorboard
    from torch.utils.tensorboard import SummaryWriter

    # Additional
    from tqdm import trange, tqdm
    import torchio as tio
    import numpy as np
    import os
    import copy

    # Local modules
    from dataset import AnhirPatches
    from config import config, base_transforms, data_config, model_config, init_data
    from NonRigidNet.nonrigidnet import Nonrigid_Registration_Network
    from NonRigidNet.nonrigidnet_v2 import RegistrationNetwork
    from UNet.unet_model import UNet
    from losses import Grad, NCC, diffusion, jcob_det_3, ncc_local, curvature_regularization, mind_loss
    from dataset import AnhirPatches
    from utils import load_checkpoint, df_field_v2, he_init, get_images, make_grid, moving_average, create_pyramid, resample_tensor, register, freeze_nets
    from losses import ncc_loss_global
    from samplev2 import evaluate
    # %%

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Check for existing folder with resized data, if not presen make one and populate with resized images from original dataset
    init_data(root='raw/', resize=data_config['resample_rate'], summary_path=data_config['main_file_path'], save_root='')

    rTRE_list = []


    # %% Hyperparameters definitions
    lambda_reg = 6000
    learning_rate = 1e-4
    decay = 0.99
    batch_size = 4
    regularization = diffusion
    cost = ncc_local
    lambda_trans = None
    freeze_list = [2]
    levels = 2
    load_levels = [0, 1]


    # Tensorboard logger annotation and model checkpoint name
    exp_path = f'-levels_{levels}-freeze_{freeze_list}-cost_{cost.__name__}-reg_{regularization.__name__}-lr_{learning_rate}-decay_{decay}-bsize_{batch_size}-lmd_reg_{lambda_reg}-lmd_trans_{lambda_trans}'
    if cost.__name__ == 'ncc_local': loss_type = 'NCC'
    elif cost.__name__ == 'mind_loss': loss_type = 'MIND'

    print("Using name:")
    print(data_config['summary_path']+exp_path)
    # Dataset initialization
    AnhirSet = AnhirPatches(data_config['main_file_path'], data_config['train_root'], base_transforms=base_transforms)
    dataset = AnhirSet.get_training_subjects()
    # torchio queue for patch-based pipeline
    queue = AnhirSet.get_queue(patch_size=(256, 256, 1),
                                max_length=batch_size*8,
                                samples_per_volume=batch_size*8,
                                num_workers=4,
                                shuffle_subjects=True,
                                shuffle_patches=False,
                                start_background=True,)
    training_loader_patches = torch.utils.data.DataLoader(queue, batch_size=batch_size)
    params = {'patch_size':(256, 256, 1), 'patch_overlap':(20, 20, 0)} # parameters on test-time inference
    # %% Model init
    # transfer = UNet(6, 6).to(device).apply(he_init)
    if lambda_trans is not None: transfer = RegistrationNetwork(input_channels=3, ouput_channels=6).apply(he_init).to(device)
    else: transfer = None
    channels = 2
    # if int_num is not None:
    #     nonrigid = Nonrigid_Registration_Network(channels, vecint=True, num_int=int_num).to(device).apply(he_init)
    # else:
    #     nonrigid = Nonrigid_Registration_Network(channels, vecint=False, num_int=int_num).to(device).apply(he_init)

    nonrigid_0 = RegistrationNetwork(input_channels=2, ouput_channels=2).apply(he_init).to(device)
    nonrigid_1 = RegistrationNetwork(input_channels=2, ouput_channels=2).apply(he_init).to(device)
    nonrigid_2 = RegistrationNetwork(input_channels=2, ouput_channels=2).apply(he_init).to(device)
    # nonrigid_ema = copy.deepcopy(nonrigid)
    # nonrigid = nonrigid.apply(he_init).to(device)
    # nonrigid_ema = nonrigid.to(device)

    nets = []
    nets += [nonrigid_0, nonrigid_1, nonrigid_2]
    if transfer is not None: nets += [transfer]

    # parameters list for optimizer
    parameters = []
    for net in nets:
        parameters += list(net.parameters())

    # optimizer, scheduler and tensorboard logger initialization
    optimizer = optim.Adam(parameters, learning_rate, betas=model_config['betas'])
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: decay**step)
    # scheduler = None
    # epochs_oc = config['epochs']-(config['epochs']-lr_config['swa_kickin'])
    # print(epochs_oc)
    # scheduler_onecycle = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(training_loader_patches), epochs=epochs_oc)
    # scheduler = scheduler_onecycle
    # SWA
    # nonrigid_swa = AveragedModel(nonrigid)

    # scaler = amp.GradScaler()


    # Logging
    writer = SummaryWriter(data_config['summary_path']+exp_path)

    # pretrained model resuming if needed
    if model_config['resume']:
        if data_config['model_checkpoint'] is not None:
            epoch_resume = load_checkpoint(data_config['model_checkpoint'], nets, load_levels, transfer, optimizer, scheduler, model_config=model_config)
            print("loaded arbitrary model")
        else:
            epoch_resume = load_checkpoint(model_config['checkpoint_path']+exp_path+'.tar', nets, load_levels, transfer, optimizer, scheduler, model_config=model_config)
            print("loaded previous model")
    else: epoch_resume = 0

    freeze_nets(nets, freeze_list)

    # main trainloop
    for epoch in trange(epoch_resume, config['epochs']+1, desc='Epochs'):
        batch_loss_ncc = []
        batch_loss_grad = []
        batch_loss_trans = []
        for samples in tqdm(training_loader_patches, desc='Dataloader'):
            # loading images, transfering to gpu and adding noise
            content_image, style_image = get_images(samples, device, False)
            # conditional style transfer
            if transfer is not None:
                content_cp =  TF.rgb_to_grayscale(content_image.detach().clone())

                intense_tensor = transfer(content_image, style_image)
                content_image = ((content_image * intense_tensor[:, 0:3, :, :]) + intense_tensor[:, 3:, :, :])
            else:
                content_cp = None

            # transfer to grayscale
            content_image = TF.rgb_to_grayscale(content_image)
            style_image = TF.rgb_to_grayscale(style_image)

            # taking negative of images
            content_image = 1 - content_image
            style_image = 1 - style_image

            content_pyramid = create_pyramid(content_image, len(nets))
            style_pyramid = create_pyramid(style_image, len(nets))

            # non rigid transform

            output_images, n_grids, deformation_fields = register(nets, content_pyramid, style_pyramid, levels, device)
            # output_image = F.grid_sample(content_image, n_grid, mode='bilinear', padding_mode='zeros', align_corners=False)
            # loss

            loss_grad = 0
            for deformation in deformation_fields:
                loss_grad += lambda_reg*regularization(deformation, device)
            
            loss_ncc = 0
            if len(output_images) == 1:
                loss_ncc += cost(output_images[0], style_pyramid[-1], device)
            else:
                for i, image in enumerate(output_images):
                    loss_ncc += cost(image, style_pyramid[i], device)

            if transfer is not None: loss_transfer = lambda_trans*diffusion(intense_tensor, device)
            if transfer is not None: loss = loss + loss_transfer
            batch_loss_ncc += [loss_ncc.item()]
            batch_loss_grad += [loss_grad.item()]
            if transfer is not None: batch_loss_trans += [loss_transfer.item()]

            loss = loss_ncc + loss_grad

            # gradient calculations
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # if epoch < lr_config['swa_kickin']:
            #     scheduler_onecycle.step()
            #     # print('lr adjusted')

        # lr adjustments
        # if epoch == lr_config['swa_kickin']:
        #     print(f'lr for swa: {scheduler.get_last_lr()}')
        #     swa_scheduler = SWALR(optimizer, swa_lr=scheduler.get_last_lr())
        #     print(f'swa scheduler init, {epoch=}')
        # elif epoch > lr_config['swa_kickin']:
        #     nonrigid_swa.update_parameters(nonrigid)
        #     swa_scheduler.step()
        #     # print(f'changed swa, {epoch=}')
        # elif epoch >= lr_config['decay_kickin']:
        scheduler.step()
            # print(f'changed decay, {epoch=}')
        # moving_average(nonrigid, nonrigid_ema, beta=0.999)
        
        content_image = content_pyramid[-1]
        style_image = style_pyramid[-1]
        output_image = output_images[-1]
        n_grid = n_grids[-1]
        deformation_field = deformation_fields[-1]
        # tensorboard loging
        writer.add_scalar(f"Learning rate", optimizer.param_groups[0]['lr'], global_step=epoch)
        writer.add_scalar(f"{loss_type} loss", np.mean(batch_loss_ncc), global_step=epoch)
        writer.add_scalar("Grad loss (scaled)", np.mean(batch_loss_grad), global_step=epoch)
        if transfer is not None: writer.add_scalar(f"Trans loss", np.mean(batch_loss_trans), global_step=epoch)
        jacobians = jcob_det_3(n_grid)
        writer.add_scalar("Mean of sum of negative jacobians in the batch", jacobians, global_step=epoch)

        # tensorboard logging
        if epoch % config['sample_every'] == 0:
        
            # inversing negatives of images
            content_image = 1 - content_image
            style_image = 1 - style_image
            output_image = 1 - output_image

            # image examples and warped grids examples
            img_grid = make_grid(output_image, output_image.shape[0], content_image, style_image, src_cp=content_cp, n_grid=n_grid, device=device)
            writer.add_image("Registration (& transfer) samples", img_grid, global_step=epoch)
            writer.flush()

        # model checkpoint
        if epoch % config['checkpoint_every'] == 0:
            save_dict = {
                'epoch': epoch,
                'tranfer_model_state_dict': transfer.state_dict() if transfer is not None else False,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                }
            nets_dict = {f'nonrigid_{i}': nets[i].state_dict() for i in range(len(nets))}
            save_dict = {**save_dict, **nets_dict}
            torch.save(save_dict, model_config['checkpoint_path']+exp_path+'.tar')
            print('Saved checkpoint\n')

        # if epoch % 101 == 0 and epoch != 0:
        #     scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: decay**epoch)
        #     print("current state of scheduler:")
        #     print(scheduler.state_dict())

        # evaluation
        # if epoch % config['test_every'] == 0 and epoch != 0:

        # calculating whole images from patches and rTRE estimations
        if epoch % config['eval_every'] == 0:
            grid, rTRE = evaluate(AnhirSet, nets, levels, transfer, data_config['main_file_path'], device, data_config['train_root'], data_config['resample_rate'], **params) #NOTE: *4 for reduced dataset
            rTRE_list += [rTRE]

            # print(f'{rTRE=}')
            writer.add_scalar("rTRE metric", rTRE, global_step=epoch)
            if grid is not None:
                writer.add_image("Registration (& transfer) test", grid.float(), global_step=epoch)
            writer.add_hparams({'lr': learning_rate,
                                'decay': decay,
                                'batch_size': batch_size,
                                'resample_rate': data_config['resample_rate'],
                                'lambda_reg': lambda_reg},
                                {'hparam/loss_ncc': np.mean(batch_loss_ncc),
                                'hparam/rTRE': rTRE,
                                'hparam/jacobians': jacobians})

        writer.flush()

    save_dict = {
                'epoch': epoch,
                'tranfer_model_state_dict': transfer.state_dict() if transfer is not None else False,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                }
    nets_dict = {f'nonrigid_{i}': nets[i].state_dict() for i in range(len(nets))}
    save_dict = {**save_dict, **nets_dict}
    torch.save(save_dict, model_config['checkpoint_path']+exp_path+'.tar')
    print('Saved checkpoint\n')