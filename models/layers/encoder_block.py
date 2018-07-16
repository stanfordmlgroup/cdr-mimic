import torch.nn as nn


class EncoderBlock(nn.Module):
    def __init__(self):
        super(EncoderBlock, self).__init__()

    def forward(self, x):
        raise NotImplementedError
