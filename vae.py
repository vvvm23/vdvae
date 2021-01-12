"""
    Encoder Components:
        - Encoder, contains all the EncoderBlocks and manages data flow through them.
        - EncoderBlock, contains sub-blocks of residual units and a pooling layer.
        - ResidualBlock, contains a block of residual connections, as described in the paper (1x1,3x3,3x3,1x1)
            - We could slightly adapt, and make it a ReZero connection. Needs some testing.
        
    Decoder Components:
        - Decoder, contains all DecoderBlocks and manages data flow through them.
        - DecoderBlock, contains sub-blocks of top-down units and an unpool layer.
        - TopDownBlock, implements the topdown block from the original paper.

    All is encapsulated in the main VAE class.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    Some helper functions for common constructs
"""
class ConvBuilder:
    def _bconv(in_dim, out_dim, kernel_size, stride, padding):
        conv = nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding)
        return conv
    def b1x1(in_dim, out_dim):
        return ConvBuilder._bconv(in_dim, out_dim, 1, 1, 0)
    def b3x3(in_dim, out_dim):
        return ConvBuilder._bconv(in_dim, out_dim, 3, 1, 1)

"""
    Helper module to call super().__init__() for us
"""
class HelperModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.build(*args, **kwargs)

    def build(self, *args, **kwargs):
        raise NotImplementedError

"""
    Encoder Components
"""
class ResidualBlock(HelperModule):
    def build(self, in_width, hidden_width): # hidden_width should function as a bottleneck!
        self.conv = nn.ModuleList([
            ConvBuilder.b1x1(in_width, hidden_width),
            ConvBuilder.b3x3(hidden_width, hidden_width),
            ConvBuilder.b3x3(hidden_width, hidden_width),
            ConvBuilder.b1x1(hidden_width, in_width)
        ])
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        xh = x
        for l in self.conv:
            xh = l(F.gelu(xh))
        y = x + self.gate*xh
        return y

class EncoderBlock(HelperModule):
    def build(self, in_dim, nb_r_blocks, bottleneck_ratio, downscale_rate):
        self.downscale_rate = downscale_rate
        self.res_blocks = nn.ModuleList([
            ResidualBlock(in_dim, int(in_dim*bottleneck_ratio))
        for _ in range(nb_r_blocks)])
        
    def forward(self, x):
        y = x
        for l in self.res_blocks:
            y = l(y)
        a = y
        y = F.avg_pool2d(y, kernel_size=self.downscale_rate, stride=self.downscale_rate)
        return y, a # y is input to next block, a is activations to topdown layer

class Encoder(HelperModule):
    def build(self, in_dim, hidden_width, nb_encoder_blocks, nb_res_blocks=3, bottleneck_ratio=0.5, downscale_rate=2):
        self.in_conv = ConvBuilder.b3x3(in_dim, hidden_width)
        self.enc_blocks = nn.ModuleList([
            EncoderBlock(hidden_width, nb_res_blocks, bottleneck_ratio, downscale_rate)
        for _ in range(nb_encoder_blocks)])

    def forward(self, x):
        x = self.in_conv(x)
        activations = [x]
        for b in self.enc_blocks:
            x, a = b(x)
            activations.append(a)
        return activations

"""
    Decoder Components
"""
class TopDownBlock(HelperModule):
    def build(self):
        pass
    def forward(self, x):
        pass

class DecoderBlock(HelperModule):
    def build(self):
        pass
    def forward(self, x):
        pass

class Decoder(HelperModule):
    def build(self):
        pass
    def forward(self, x):
        pass

"""
    Main VAE class
"""
class VAE(HelperModule):
    def build(self):
        pass
    def forward(self, x):
        pass

if __name__ == "__main__":
    encoder = Encoder(3, 16, 4)
    x = torch.randn(1, 3, 128, 128)
    activations = encoder(x)
    for a in activations:
        print(a.shape)
