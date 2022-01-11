# %% Imports
import numpy as np
from numpy.core.numeric import ones_like, zeros_like
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math

from torchvision.transforms.functional import affine
from utils import df_field_v2, tc_df_to_np_df

# %% Cost functions
def ncc_local(sources, targets, device="cpu", **params):
    """
    Implementation inspired by VoxelMorph (with some modifications).
    Courtesy of: dr. Marek Wodziński
    """
    ndim = len(sources.size()) - 2
    if ndim not in [2, 3]:
        raise ValueError("Unsupported number of dimensions.")
    try:
        win_size = params['win_size']
    except:
        win_size = 9
    window = (win_size, ) * ndim
    sum_filt = torch.ones([1, 1, *window]).to(device)
    pad_no = math.floor(window[0] / 2)
    stride = ndim * (1,)
    padding = ndim * (pad_no,)
    conv_fn = getattr(F, 'conv%dd' % ndim)
    sources_denom = sources**2
    targets_denom = targets**2
    numerator = sources*targets
    sources_sum = conv_fn(sources, sum_filt, stride=stride, padding=padding)
    targets_sum = conv_fn(targets, sum_filt, stride=stride, padding=padding)
    sources_denom_sum = conv_fn(sources_denom, sum_filt, stride=stride, padding=padding)
    targets_denom_sum = conv_fn(targets_denom, sum_filt, stride=stride, padding=padding)
    numerator_sum = conv_fn(numerator, sum_filt, stride=stride, padding=padding)
    size = np.prod(window)
    u_sources = sources_sum / size
    u_targets = targets_sum / size
    cross = numerator_sum - u_targets * sources_sum - u_sources * targets_sum + u_sources * u_targets * size
    sources_var = sources_denom_sum - 2 * u_sources * sources_sum + u_sources * u_sources * size
    targets_var = targets_denom_sum - 2 * u_targets * targets_sum + u_targets * u_targets * size
    ncc = cross * cross / (sources_var * targets_var + 1e-5)
    return -torch.mean(ncc)

def mind_loss(sources, targets, device="cpu", **params):
    '''Courtesy of: dr. Marek Wodziński'''
    sources = sources.view(sources.size(0), sources.size(1), sources.size(2), sources.size(3), 1)
    targets = targets.view(targets.size(0), targets.size(1), targets.size(2), targets.size(3), 1)
    try:
        dilation = params['dilation']
        radius = params['radius']
        return torch.mean((MINDSSC(sources, device=device, dilation=dilation, radius=radius) - MINDSSC(targets, device=device, dilation=dilation, radius=radius))**2)
    except:
        return torch.mean((MINDSSC(sources, device=device) - MINDSSC(targets, device=device))**2)
def pdist_squared(x):
    '''Courtesy of: dr. Marek Wodziński'''
    xx = (x**2).sum(dim=1).unsqueeze(2)
    yy = xx.permute(0, 2, 1)
    dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
    dist[dist != dist] = 0
    dist = torch.clamp(dist.float(), 0.0, np.inf)
    return dist
def MINDSSC(img, radius=2, dilation=2, device="cpu"):
    '''Courtesy of: dr. Marek Wodziński'''
    kernel_size = radius * 2 + 1
    six_neighbourhood = torch.Tensor([[0,1,1],
                                      [1,1,0],
                                      [1,0,1],
                                      [1,1,2],
                                      [2,1,1],
                                      [1,2,1]]).long()
    dist = pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)
    x, y = torch.meshgrid(torch.arange(6), torch.arange(6))
    mask = ((x > y).view(-1) & (dist == 2).view(-1))
    idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1,6,1).view(-1,3)[mask,:]
    idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6,1,1).view(-1,3)[mask,:]
    mshift1 = torch.zeros(12, 1, 3, 3, 3).to(device)
    mshift1.view(-1)[torch.arange(12) * 27 + idx_shift1[:,0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
    mshift2 = torch.zeros(12, 1, 3, 3, 3).to(device)
    mshift2.view(-1)[torch.arange(12) * 27 + idx_shift2[:,0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
    rpad1 = nn.ReplicationPad3d(dilation)
    rpad2 = nn.ReplicationPad3d(radius)
    ssd = F.avg_pool3d(rpad2((F.conv3d(rpad1(img), mshift1, dilation=dilation) - F.conv3d(rpad1(img), mshift2, dilation=dilation)) ** 2), kernel_size, stride=1)
    mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
    mind_var = torch.mean(mind, 1, keepdim=True)
    mind_var = torch.clamp(mind_var, mind_var.mean().item()*0.001, mind_var.mean().item()*1000)
    mind /= mind_var
    mind = torch.exp(-mind)
    mind = mind[:, torch.Tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]).long(), :, :, :]
    return mind
    
class NCC:
    """
    Local (over window) normalized cross correlation loss.
    Adapted from VoxelMorph.
    """

    def __init__(self, device='cpu', win=None):
        self.win = win
        self.device = device

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to(self.device)

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        ncc = -torch.mean(cc)
        if torch.isnan(ncc):
            return torch.Tensor([1.], requires_grad=True, device=self.device)
        return ncc

def ncc_loss(output, target):
    ...

def ncc_loss_global(source, target, device="cpu"):
    '''
    Adapted from DeepHistReg.
    '''
    return ncc_losses_global(source, target, device=device)

def ncc_losses_global(sources, targets, device="cpu"):
    '''
    Adapted from DeepHistReg.
    '''
    ncc = ncc_global(sources, targets, device=device)
    ncc = torch.mean(ncc)
    if ncc != ncc:
        return torch.autograd.Variable(torch.Tensor([1]), requires_grad=True).to(device)
    return -ncc

def ncc_global(sources, targets, device="cpu"):
    '''
    Adapted from DeepHistReg.
    '''
    size = sources.size(2)*sources.size(3)
    sources_mean = torch.mean(sources, dim=(1, 2, 3)).view(sources.size(0), 1, 1, 1)
    targets_mean = torch.mean(targets, dim=(1, 2, 3)).view(sources.size(0), 1, 1, 1)
    sources_std = torch.std(sources, dim=(1, 2, 3)).view(sources.size(0), 1, 1, 1)
    targets_std = torch.std(targets, dim=(1, 2, 3)).view(sources.size(0), 1, 1, 1)
    ncc = (1/size)*torch.sum((sources - sources_mean)*(targets-targets_mean) / (sources_std * targets_std), dim=(1, 2, 3))
    return ncc
# %% intensity transform regularization
class Grad:
    """
    2-D gradient loss.
    Adapted from VoxelMorph.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx

        d = torch.mean(dx) + torch.mean(dy)
        grad = d / 2.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad

def diffusion(displacement_field, device='cpu'):
    '''
    Courtesy of: Marek Wodziński
    '''
    dx = (displacement_field[:, 1:, :, :] - displacement_field[:, :-1, :, :])**2
    dy = (displacement_field[:, :, 1:, :] - displacement_field[:, :, :-1, :])**2
    diffusion_reg = (torch.mean(dx) + torch.mean(dy)) / 2
    return diffusion_reg

def curvature_regularization(displacement_fields, device="cpu"):
    '''
    Adapted from DeepHistReg
    '''
    u_x = displacement_fields[:, :, :, 0].view(-1, 1, displacement_fields.size(1), displacement_fields.size(2))
    u_y = displacement_fields[:, :, :, 1].view(-1, 1, displacement_fields.size(1), displacement_fields.size(2))
    x_laplacian = tensor_laplacian(u_x, device)[:, :, 1:-1, 1:-1]
    y_laplacian = tensor_laplacian(u_y, device)[:, :, 1:-1, 1:-1]
    x_term = x_laplacian**2
    y_term = y_laplacian**2
    curvature = torch.mean(1/2*(x_term + y_term))
    return curvature

def tensor_laplacian(tensor, device="cpu"):
    '''
    Adapted from DeepHistReg
    '''
    laplacian_filter = torch.Tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]).to(device)
    laplacian = F.conv2d(tensor, laplacian_filter.view(1, 1, 3, 3), padding=1) / 9
    return laplacian
# %% dynamic regularization parameter
def lambda_reg(epoch, car=10, kar=4, device='cpu', shift=0):
    n = torch.tensor([epoch/kar], device=device)
    loss = (car*kar)/(kar + torch.exp(n))
    if loss != loss:
        return torch.autograd.Variable(torch.Tensor([0]), requires_grad=True).to(device)
    return loss

# %% kinda metric
def jcob_det(n_grid):
    '''
    n_grid format: [batch, x, y, channels]
    returns mean of negative jacobians in a batch
    '''
    Jdet = np.zeros([n_grid.shape[0], n_grid.shape[1], n_grid.shape[2]])
    for i in range(n_grid.shape[0]):
        J = n_grid[i].detach().cpu().numpy()
        J = np.gradient(J)
        dx = J[0]
        dy = J[1]
        Jdet0 = dx[:, :, 0] * dy[:, :, 1]
        Jdet1 = dx[:, :, 1] * dy[:, :, 0]
        Jdet[i, :, :] = Jdet0 - Jdet1
    return np.sum(Jdet < 0)/n_grid.shape[0]

def jcob_det_2(displacement_field):
    '''
    Mean sum of negative jacobians
    '''
    dx = torch.abs(displacement_field[:, 1:, :, :] - displacement_field[:, :-1, :, :])
    dy = torch.abs(displacement_field[:, :, 1:, :] - displacement_field[:, :, :-1, :])

    Jdet0 = dx[:, :, 1:, 0] * dy[:, 1:, :, 1]
    Jdet1 = dy[:, 1:, :, 0] * dx[:, :, 1:, 1]
    Jdet = Jdet0 - Jdet1
    sum = torch.sum(Jdet < 0) / displacement_field.shape[0]
    return sum

def jcob_det_3(displacement_fields, spacing=(1.0, 1.0)):
    '''
    Courtesy of: Marek Wodziński
    '''
    jac_det = []
    for i in range(displacement_fields.shape[0]):
        displacement_field = tc_df_to_np_df(displacement_fields[i, ...].unsqueeze(0))
        u_x = displacement_field[0, :, :]
        u_y = displacement_field[1, :, :]
    
        y_size, x_size = np.shape(u_x)
        x_grid, y_grid = np.meshgrid(np.arange(x_size), np.arange(y_size))
    
        x_field, y_field = x_grid + u_x, y_grid + u_y
        x_field, y_field = x_field * spacing[0], y_field*spacing[1]
    
        u_x_grad, u_y_grad = np.gradient(x_field), np.gradient(y_field)
    
        u_xy, u_xx = u_x_grad[0], u_x_grad[1]
        u_yy, u_yx = u_y_grad[0], u_y_grad[1]
    
        jac = np.array([[u_xx, u_xy], [u_yx, u_yy]]).swapaxes(1, 2).swapaxes(0, 3).swapaxes(0, 1)
        det = np.linalg.det(jac)
        jac_det += [np.sum(det<0)]

    return np.sum(jac_det) / displacement_fields.shape[0]