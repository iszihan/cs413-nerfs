from data.dataset import NerfDataset
from common.util import toNP

import torch 
from nerf.hierarchical_sampler import sample_coarse
train_dataset = NerfDataset(dataset='blender', mode='test')

rays_rgb = train_dataset.__getitem__(1)
rays = rays_rgb[:2]
rays = rays.permute(1,2,0,3).reshape(rays.shape[1],rays.shape[2],6) # h, w, 6
rays = torch.cat([rays, torch.tensor([train_dataset.near]).expand(rays.shape[0], rays.shape[1],1), 
                        torch.tensor([train_dataset.far]).expand(rays.shape[0], rays.shape[1],1)], dim=-1)

coarse_pts = sample_coarse(rays, 10) #h,w,n,3







