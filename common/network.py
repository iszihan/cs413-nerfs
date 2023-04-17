import torch.nn as nn
import torch

"""neural network building blocks, positional encoding"""
def create_freq_mask(l, t, T):
    freq_mask = torch.zeros(l)
    ptr = l / 3 * t / T + 1 
    ptr = ptr if ptr < l / 3 else l / 3 
    int_ptr = int(ptr)
    freq_mask[:int_ptr * 3] = 1.0 
    freq_mask[int_ptr * 3: int_ptr * 3 + 3] = (ptr - int_ptr)
    return torch.clip(freq_mask, 1e-8, 1-1e-8)

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
    
