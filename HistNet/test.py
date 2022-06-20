print('loading libraries...')
from utils import load_checkpoint
import torch
import torch.optim as optim
from NonRigidNet.nonrigidnet import Nonrigid_Registration_Network
from NonRigidNet.nonrigidnet_v2 import RegistrationNetwork

print('libraries loaded, starting...')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nonrigid = Nonrigid_Registration_Network(2, vecint=False, num_int=2).to(device)
nonrigid = RegistrationNetwork().to(device)
print('model loaded')
nets = []
nets += [nonrigid]

# parameters list for optimizer
parameters = []
for net in nets:
    parameters += list(net.parameters())

# optimizer, scheduler and tensorboard logger initialization
optimizer = optim.Adam(parameters, 1e-3)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.995**epoch)
path = 'checkpoints/registration_v2/hparams-nonrigid_v2-cost_mind_loss-vecint_None-reg_diffusion-lr_0.001-decay_0.995-bsize_8-lmd_reg_1000.tar'
print("reading...")
epoch_resume = load_checkpoint(path, nonrigid, None, optimizer, scheduler, None)
print('Done, reading')