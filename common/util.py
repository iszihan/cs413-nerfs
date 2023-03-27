""" axis transformation, rotation, ..., any other utils"""
import torch 

def writable_image(img):
    img_min = torch.min(img)
    img_max = torch.max(img)
    img = (img - img_min) * (255 / (img_max - img_min))
    #img = (img-lo) * (255 / (hi-lo))
    img = torch.round(img).clip(0,255).to(torch.uint8)
    return img

def toNP(x):
    return x.detach().to(torch.device('cpu')).numpy()

def spherical_poses():
    return None 
