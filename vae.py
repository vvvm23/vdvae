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
    def build(self):
        pass
    def forward(self, x):
        pass

class EncoderBlock(HelperModule):
    def build(self):
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
    print("Aloha, World!")
