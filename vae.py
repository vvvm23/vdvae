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
    Diagonal Gaussian Distribution and loss.
    Taken directly from OpenAI implementation 
    Decorators means these functions will be compiled as TorchScript
"""
@torch.jit.script
def gaussian_analytical_kl(mu1, mu2, logsigma1, logsigma2):
    return -0.5 + logsigma2 - logsigma1 + 0.5 * (logsigma1.exp() ** 2 + (mu1 - mu2) ** 2) / (logsigma2.exp() ** 2)
@torch.jit.script
def draw_gaussian_diag_samples(mu, logsigma):
    eps = torch.empty_like(mu).normal_(0., 1.)
    return torch.exp(logsigma) * eps + mu

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
class Block(HelperModule):
    def build(self, in_width, hidden_width, out_width): # hidden_width should function as a bottleneck!
        self.conv = nn.ModuleList([
            ConvBuilder.b1x1(in_width, hidden_width),
            ConvBuilder.b3x3(hidden_width, hidden_width),
            ConvBuilder.b3x3(hidden_width, hidden_width),
            ConvBuilder.b1x1(hidden_width, out_width)
        ])

    def forward(self, x):
        for l in self.conv:
            x = l(F.gelu(x))
        return x

class TopDownBlock(HelperModule):
    def build(self, in_width, middle_width, z_dim):
        self.cat_conv = Block(in_width*2, middle_width, z_dim*2) # parameterises mean and variance
        self.prior = Block(in_width, middle_width, z_dim*2 + in_width) # parameterises mean, variance and xh
        self.out_res = ResidualBlock(in_width, middle_width)
        self.z_conv = ConvBuilder.b1x1(z_dim, in_width)
        self.z_dim = z_dim

    def forward(self, x, a):
        xa = torch.cat([x,a], dim=1)
        qm, qv = self.cat_conv(xa).chunk(2, dim=1) # Calculate q distribution parameters. Chunk into 2 (first z_dim is mean, second is variance)
        pfeat = self.prior(x)
        pm, pv, px = pfeat[:, :self.z_dim], pfeat[:, self.z_dim:self.z_dim*2], pfeat[:, self.z_dim*2:]
        x += px

        z = draw_gaussian_diag_samples(qm, qv)
        kl = gaussian_analytical_kl(qm, pm, qv, pv)

        z = self.z_conv(z)
        x += z
        x = self.out_res(x)

        return x, kl

class DecoderBlock(HelperModule):
    def build(self, in_dim, middle_width, z_dim, nb_td_blocks, upscale_rate):
        self.upscale_rate = upscale_rate
        self.td_blocks = nn.ModuleList([
            TopDownBlock(in_dim, middle_width, z_dim)
        for _ in range(nb_td_blocks)])
    def forward(self, x, a):
        x = F.interpolate(x, scale_factor=self.upscale_rate)
        block_kl = []
        for b in self.td_blocks:
            x, kl = b(x, a)
            block_kl.append(kl)
        return x, block_kl


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
    decoder_block = DecoderBlock(16, 8, 4, 3, 2)
    x = torch.randn(1, 16, 4, 4)
    a = torch.randn(1, 16, 8, 8)
    
    y, kl = decoder_block(x, a)
    print(y.shape)
    for k in kl:
        print(k.shape)
