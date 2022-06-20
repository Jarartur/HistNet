# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# %%
class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid, persistent=False)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=False, mode=self.mode)


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):
        super().__init__()
        
        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec

class Nonrigid_Registration_Network(nn.Module):
    def __init__(self, channels, vecint=False, num_int=7):
        super(Nonrigid_Registration_Network, self).__init__()

        self.vecint = vecint

        self.encoder_1 = nn.Sequential(
            nn.Conv2d(channels, 32, 3, stride=2, padding=0),
            nn.GroupNorm(32, 32),
            nn.PReLU(),
        )
        self.encoder_2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=0),
            nn.GroupNorm(64, 64),
            nn.PReLU(),
        )
        self.encoder_3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=0),
            nn.GroupNorm(128, 128),
            nn.PReLU(),
        )
        self.encoder_4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=0),
            nn.GroupNorm(256, 256),
            nn.PReLU(),
        )
        self.encoder_5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=0),
            nn.GroupNorm(512, 512),
            nn.PReLU(),
        )

        self.decoder_5 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=0, output_padding=0),
            nn.GroupNorm(256, 256),
            nn.PReLU(),
        )
        self.decoder_4 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 3, stride=2, padding=0, output_padding=0),
            nn.GroupNorm(128, 128),
            nn.PReLU(),
        )
        self.decoder_3 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 3, stride=2, padding=0, output_padding=0),
            nn.GroupNorm(64, 64),
            nn.PReLU(),
        )
        self.decoder_2 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, 3, stride=2, padding=0, output_padding=0),
            nn.GroupNorm(32, 32),
            nn.PReLU(),
        )
        self.decoder_1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=0, output_padding=1),
            nn.GroupNorm(32, 32),
            nn.PReLU(),
        )
        self.layer_1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.GroupNorm(32, 32),
            nn.PReLU(),
        )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(32, 16, 3, stride=1, padding=1),
            nn.GroupNorm(16, 16),
            nn.PReLU(),
        )
        self.layer_3 = nn.Sequential(
            nn.Conv2d(16, 2, 3, stride=1, padding=1),
        )
        if vecint:
            self.vecint = VecInt([256, 256], num_int)

        self.vecint = VecInt([256, 256], 5)

    def forward(self, source_patches, target_patches):
        x = torch.cat((source_patches, target_patches), dim=1)
        x1 = self.encoder_1(x)
        x2 = self.encoder_2(x1)
        x3 = self.encoder_3(x2)
        x4 = self.encoder_4(x3)
        x5 = self.encoder_5(x4)
        d5 = self.decoder_5(x5)
        d4 = self.decoder_4(torch.cat((d5, x4), dim=1))
        d3 = self.decoder_3(torch.cat((d4, x3), dim=1))
        d2 = self.decoder_2(torch.cat((d3, x2), dim=1))
        d1 = self.decoder_1(torch.cat((d2, x1), dim=1))
        result = self.layer_1(d1)
        result = self.layer_2(result)
        result = self.layer_3(result)
        if self.vecint: result = self.vecint(result)
        return result.permute(0, 2, 3, 1) # -> utils.df_field

def test_forward_pass_simple():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Nonrigid_Registration_Network(2)
    y_size = 256
    x_size = 256
    no_channels = 1

    batch_size = 10
    example_source = torch.rand((batch_size, no_channels, y_size, x_size)).to(device)
    example_target = torch.rand((batch_size, no_channels, y_size, x_size)).to(device)

    result = model(example_source, example_target)
    print(result.size())

if __name__ == "__main__":
    test_forward_pass_simple()