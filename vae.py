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
        self.conv = nn.Sequential(
            nn.GELU(), ConvBuilder.b1x1(in_width, hidden_width),
            nn.GELU(), ConvBuilder.b3x3(hidden_width, hidden_width),
            nn.GELU(), ConvBuilder.b3x3(hidden_width, hidden_width),
            nn.GELU(), ConvBuilder.b1x1(hidden_width, in_width)
        )
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        xh = self.conv(x)
        y = x + self.gate*xh
        return y

class EncoderBlock(HelperModule):
    def build(self, in_dim, nb_r_blocks, downscale_rate):
        pass
    def forward(self, x):
        pass

class Encoder(HelperModule):
    def build(self):
        pass
    def forward(self, x):
        pass

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
    res_block = ResidualBlock(8, 4)
    x = torch.randn(1, 8, 4, 4)
    print(res_block(x).shape)
