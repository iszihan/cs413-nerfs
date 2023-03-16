""" axis transformation, rotation, ..., any other utils"""
import torch 

def toNP(x):
    return x.detach().to(torch.device('cpu')).numpy()

def spherical_poses():
    return None 
