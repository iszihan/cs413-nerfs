
import torch.nn as nn 

class NerfModel(nn.Module):
    """the MLP for NeRF""" 
    def __init__(self):
        super(NerfModel, self).__init__()

        # Kinjal 
        # self.mlp = self.build_network() 

    def build_network(self):
        return None 
    
    def forward(self, x):
        

