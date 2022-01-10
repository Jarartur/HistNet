import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
import math
import utils

class ResidualBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super(ResidualBlock, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(input_size, output_size, 3, stride=1, padding=1),
            nn.GroupNorm(output_size, output_size),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(output_size, output_size, 3, stride=1, padding=1),
            nn.GroupNorm(output_size, output_size),
            nn.LeakyReLU(0.01, inplace=True),        
        )
        self.conv = nn.Sequential(
            nn.Conv2d(input_size, output_size, 1)
        )
    def forward(self, x):
        return self.module(x) + self.conv(x)

class RegistrationNetwork(nn.Module):
    """
    Simple U-Net-like architecture.
    """
    def __init__(self, input_channels=1, ouput_channels=2):
        super(RegistrationNetwork, self).__init__()
        self.input_channels = input_channels
        
        self.encoder_1 = nn.Sequential(
            nn.Conv2d(input_channels*2, 16, 4, stride=2, padding=1),
            nn.GroupNorm(16, 16),
            nn.LeakyReLU(0.01, inplace=True),   
        )
        self.encoder_2 = nn.Sequential(
            ResidualBlock(16, 32),
            nn.Conv2d(32, 32, 4, stride=2, padding=1),
            nn.GroupNorm(32, 32),
            nn.LeakyReLU(0.01, inplace=True),   
        )
        self.encoder_3 = nn.Sequential(
            ResidualBlock(32, 64),
            nn.Conv2d(64, 64, 4, stride=2, padding=1),
            nn.GroupNorm(64, 64),
            nn.LeakyReLU(0.01, inplace=True), 
        )
        self.encoder_4 = nn.Sequential(
            ResidualBlock(64, 128),
            nn.Conv2d(128, 128, 4, stride=2, padding=1),
            nn.GroupNorm(128, 128),
            nn.LeakyReLU(0.01, inplace=True), 
        )
        self.decoder_4 = nn.Sequential(
            ResidualBlock(128, 256),
            ResidualBlock(256, 256),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, output_padding=0),
            nn.GroupNorm(128, 128),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.decoder_3 = nn.Sequential(
            ResidualBlock(128+64, 64),
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1, output_padding=0),
            nn.GroupNorm(64, 64),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.decoder_2 = nn.Sequential(
            ResidualBlock(64+32, 32),
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1, output_padding=0),
            nn.GroupNorm(32, 32),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.decoder_1 = nn.Sequential(
            ResidualBlock(32+16, 16),
            nn.ConvTranspose2d(16, 16, 4, stride=2, padding=1, output_padding=0),
            nn.GroupNorm(16, 16),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.output_1 = nn.Sequential(
            ResidualBlock(16, 16),
            nn.Conv2d(16, ouput_channels, 1),
            # nn.Sigmoid()
        )

    def pad(self, image, template):
        pad_x = math.fabs(image.size(3) - template.size(3))
        pad_y = math.fabs(image.size(2) - template.size(2))
        b_x, e_x = math.floor(pad_x / 2), math.ceil(pad_x / 2)
        b_y, e_y = math.floor(pad_y / 2), math.ceil(pad_y / 2)
        image = F.pad(image, (b_x, e_x, b_y, e_y))
        return image
        
    def forward(self, src, trg):

        i1 = tc.cat((src, trg), dim=1)

        x1 = self.encoder_1(i1)
        x2 = self.encoder_2(x1)
        x3 = self.encoder_3(x2)
        x4 = self.encoder_4(x3)
        d4 = self.decoder_4(x4)
        d4 = self.pad(d4, x3)
        d3 = self.decoder_3(tc.cat((d4, x3), dim=1))
        d3 = self.pad(d3, x2)
        d2 = self.decoder_2(tc.cat((d3, x2), dim=1))
        d2 = self.pad(d2, x1)
        d1 = self.decoder_1(tc.cat((d2, x1), dim=1))
        d1 = self.pad(d1, i1)
        r1 = self.output_1(d1)
        if self.input_channels == 1: out = r1.permute(0, 2, 3, 1)
        elif self.input_channels == 3: out = r1
        return out

def load_network(weights_path=None):
    """
    Utility function to load the network.
    """
    model = RegistrationNetwork()
    if weights_path is not None:
        model.load_state_dict(tc.load(weights_path))
        model.eval()
    return model

def test_forward_pass():
    # device = "cuda:0"
    device = "cpu"
    model = RegistrationNetwork()
    y_size, x_size = 800, 800
    no_channels = 1
    batch_size = 1
    example_input = tc.rand((batch_size, no_channels, y_size, x_size)).to(device)
    example_input_2 = tc.rand((batch_size, no_channels, y_size, x_size)).to(device)
    # pyramid = utils.create_pyramid(example_input, 3, device=device)
    r1 = model(example_input, example_input_2)
    ts.summary(model, [(no_channels, y_size, x_size), (no_channels, y_size, x_size)], device=device)
    print(f'output shape: {r1.shape}')
def run():
    test_forward_pass()
if __name__ == "__main__":
    import torchsummary as ts
    run()