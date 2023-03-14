import torch.nn as nn 

class DepthNet(nn.Module):
    '''Network to predict depth in PointNerf'''
    def __init__(self):
        return None 

class ImageNet(nn.Moudle):
    '''Network to predict image feature in PointNerf'''
    def __init__(self):
        return None 

class NerfModel(nn.Module):
    def __init__(self, dim_in):
        super(NerfModel, self).__init__()

    def forward(self, encodings):
        '''
        Modularized to take different type of encodings:
        - positional encoding for vanilla NeRF
        - encoding with point features in PointNeRF
        (just need to make input dimension an argument?)
        '''
        return None
    