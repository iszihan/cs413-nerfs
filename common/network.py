import torch.nn as nn
import torch

"""neural network building blocks, positional encoding"""

class PositionalEncoding(nn.Module):
    """Positional encoding for 3D points"""
    def __init__(self, num_freq_bands=6, include_input=True):
        super(PositionalEncoding, self).__init__()
        self.num_freq_bands = num_freq_bands
        self.include_input = include_input

    def forward(self, x):
        """x: (N, 3)"""
        ret = [x]
        if self.include_input:
            ret.append(x)
        for freq in range(1, self.num_freq_bands + 1):
            for func in [torch.sin, torch.cos]:
                ret.append(func(freq * x))
        ret = torch.cat(ret, dim=-1)
        return ret
    
