"Hierarchical Sampling"

import torch 
def sample_coarse(rays, n, perturb=False):
    ts = torch.linspace(0.0, 1.0, steps=n)
    # uniform sampling -- the original implementation provides the alternative of inverse depth sampling
    zs = rays[:,:,6:7] * (1. - ts) + rays[:,:,7:8] * ts
    
    # perturb if specified 
    if perturb:
        mid_zs = 0.5 * (zs[...,1:] + zs[...,:-1])
        upper_zs = torch.cat([mid_zs, zs[...,-1:]], -1)
        lower_zs = torch.cat([zs[...,:1], mid_zs], -1)
        t_rand = torch.rand(zs.shape)
        zs = lower_zs + (upper_zs - lower_zs) * t_rand
    
    pts = rays[:,:,None, :3] + rays[:,:,None,3:6] * zs[:,:,:,None]
    views = rays[:,:,None,3:6].repeat(1,1,10,1)
    return torch.cat([pts.reshape(-1,n,3), views.reshape(-1,n,3)],dim=-1) #[h,w,n_samples,6]
    
def sample_fine(self):
    pts = None 
    return pts 



