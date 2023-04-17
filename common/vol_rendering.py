import numpy as np
import torch
import torch.nn.functional as F
import nerf.nerf as nerf
import common.rays as rays_module
from common.util import printarr

def volumetric_rendering_per_ray(model, t_n, t_f, n_samples=10, rays=None, opt=None):
    """"
    @input:
    rays: [nr, 6]
    @return:
    imgs: [b, h, w, 3]
    """
    cam_rays = rays.reshape(rays.shape[0], 2, 3)
    rgba, weights, samples = expected_colour(model, cam_rays, t_n, t_f, n_samples, opt=opt)
    return rgba


def volumetric_rendering_per_image(model, t_n, t_f, n_samples=10, rays=None, h=None, w=None, focal=None, c2w=None, opt=None):
    """
    h, w: image height and width
    focal: focal length
    c2w: camera to world matrix - [b, 3, 4] matrix
    t_n, t_f: ray interval

    Return:
    @imgs: [b, h, w, 3]
    """
    # get rays for all cameras
    if rays is None:
        cam_rays = rays_module.get_rays_torch(h, w, focal, c2w) #[b, h, w, 2, 3]
    else:
        cam_rays = rays.reshape(rays.shape[0], rays.shape[1], rays.shape[2], 2, 3)
        h = rays.shape[1]
        w = rays.shape[2]

    imgs = torch.zeros((cam_rays.shape[0], h, w, 4)).to('cuda')

    # get expected colour for each ray for each camera
    for cam in range(cam_rays.shape[0]):
        # get rays for each camera
        cam_rays_i = cam_rays[cam] # [h, w, 2, 3]
        # get expected colour for each ray
        # two rows at a time to avoid OOM
        for i in range(cam_rays_i.shape[0]):
            cam_rays_i_batch = cam_rays_i[i, ...].reshape(-1, 2, 3)
            rgba, weights, samples = expected_colour(model, cam_rays_i_batch, t_n, t_f, n_samples, opt)
            # assign colour to image
            imgs[cam, i, :, :] = rgba.reshape(w, 4)
    return imgs


def expected_colour(model, rays, t_n, t_f, n_samples, opt=None):
    """
    rays = [origin, direction] of shape [n_rays, 2, 3]
    t_n, t_f = ray interval
    n_samples = number of samples
    opt = options
    course = True if coarse training, default True
    h_samples = [n_rays, n_samples], samples along the rays for fine training, default None

    Expected colour, alpha of the rays; weights of the samples for hierarchical sampling
    Returns:
        output: [n_rays, 4]
        weights: [n_rays, n_samples]
        samples: [n_rays, n_samples]
    """

    fine_sampling = opt.fine_sampling

    # coarse sampling always
    samples = stratified_sampling(t_n, t_f, n_samples).to(opt.device)
    samples = samples.transpose(0,1).repeat(rays.shape[0], 1)

    # get points along the rays
    pts = rays[:, 0, :].unsqueeze(1).repeat(1, n_samples, 1) + \
          rays[:, 1, :].unsqueeze(1).repeat(1, n_samples, 1) * samples.unsqueeze(-1).repeat(1, 1, 3)

    # get density and colour for each point from the model
    input = torch.cat([pts.reshape(-1, 3), rays[:, 1, :].repeat(1, n_samples).reshape(-1, 3)], dim=1) # [n_rays * n_samples, 6]
    
    # positional encoding
    encoded_pts, encoded_views = model.module.encode_input(input) #8000, 60; 8000, 24
    input = torch.cat([encoded_pts, encoded_views], dim=1) #8000, 84
    output = model(input.reshape(rays.shape[0], n_samples, -1).float())# [n_rays, n_samples, 4]

    # activation for density and colour
    rgb = torch.sigmoid(output[:, :, :3])  # [n_rays , n_samples, 3]
    density = torch.nn.functional.relu(output[:, :, 3]) # [n_rays , n_samples]

    # distance between samples
    dist = torch.cat([samples[:, 1:] - samples[:, :-1],
                        1e10*torch.ones(rays.shape[0], 1).to(opt.device)], dim=1)  # [n_rays, n_samples]
    ray_norms = torch.norm(rays[:, 1, :].unsqueeze(1).repeat(1, n_samples, 1), dim=-1) # [n_rays , n_samples]
    dist = dist * ray_norms

    # compute expected colour, alpha, weights
    weighted_density = (density * dist).reshape(rays.shape[0], n_samples)
    accumulated_transmittance = torch.exp(-torch.cumsum(weighted_density + 1e-10, dim=1))
    alphas = torch.ones_like(weighted_density) - torch.exp(-weighted_density)
    final_weights = alphas * accumulated_transmittance
    if not fine_sampling:
        # composite using coarse points only 
        colour_pred = torch.sum(final_weights[..., None].repeat(1, 1, 3)* rgb, dim=1)
        alpha = alphas[:, -1]
        output = torch.cat([colour_pred.reshape(-1, 3), alpha.reshape(-1).unsqueeze(1)], dim=1)
        return output, final_weights, samples
    else:
        # fine sampling 
        samples_mid = 0.5 * (samples[:, 1:] + samples[:, :-1])
        
        # np.save('./samples_mid.npy', samples_mid.detach().cpu().numpy())
        # np.save('./final_weights.npy', final_weights[:,1:-1].detach().cpu().numpy())
        fine_samples = inverse_transform_sampling(samples_mid, final_weights[:,1:-1], opt)
        # printarr(fine_samples)
        # np.save('./fine_samples.npy', fine_samples.detach().cpu().numpy())
        # exit()
         
        # cat and sort coarse samples with fine samples
        fine_samples = fine_samples.detach()
        samples, _ = torch.sort(torch.cat([samples, fine_samples], dim=-1),dim=-1)
        # np.save('./fine_sample_sorted.npy', samples.detach().cpu().numpy())
        # composite using coarse + fine points 
        # get points along the rays
        n_samples = samples.shape[1]
        pts = rays[:, 0, :].unsqueeze(1).repeat(1, n_samples, 1) + \
              rays[:, 1, :].unsqueeze(1).repeat(1, n_samples, 1) * samples.unsqueeze(-1).repeat(1, 1, 3)

        # get density and colour for each point from the model
        input = torch.cat([pts.reshape(-1, 3), rays[:, 1, :].repeat(1, n_samples).reshape(-1, 3)], dim=1) # [n_rays * n_samples, 6]
        
        # positional encoding
        encoded_pts, encoded_views = model.module.encode_input(input) #8000, 60; 8000, 24
        input = torch.cat([encoded_pts, encoded_views], dim=1) #8000, 84
        # np.save('./fine_samples_encoded.npy', input.detach().cpu().numpy())

        output = model(input.reshape(rays.shape[0], n_samples, -1).float())# [n_rays, n_samples, 84]
        # np.save('./output.npy', output.detach().cpu().numpy())

        # activation for density and colour
        rgb = torch.sigmoid(output[:, :, :3])  # [n_rays , n_samples, 3]
        density = torch.nn.functional.relu(output[:, :, 3]) # [n_rays , n_samples]

        # distance between samples
        dist = torch.cat([samples[:, 1:] - samples[:, :-1],
                            1e10*torch.ones(rays.shape[0], 1).to(opt.device)], dim=1)  # [n_rays, n_samples]
        ray_norms = torch.norm(rays[:, 1, :].unsqueeze(1).repeat(1, n_samples, 1), dim=-1) # [n_rays , n_samples]
        dist = dist * ray_norms

        # compute expected colour, alpha, weights
        weighted_density = (density * dist).reshape(rays.shape[0], n_samples)
        accumulated_transmittance = torch.exp(-torch.cumsum(weighted_density + 1e-10, dim=1))
        alphas = torch.ones_like(weighted_density) - torch.exp(-weighted_density)
        final_weights = alphas * accumulated_transmittance

        # composite using coarse points only 
        colour_pred = torch.sum(final_weights[..., None].repeat(1, 1, 3)* rgb, dim=1)
        alpha = alphas[:, -1]
        output = torch.cat([colour_pred.reshape(-1, 3), alpha.reshape(-1).unsqueeze(1)], dim=1)
        
        # np.save('./final_output.npy', output.detach().cpu().numpy())
        #exit()
        return output, final_weights, samples
    
def inverse_transform_sampling(coarse_samples_mid, weights, opt):
    '''
    Inverse transform sampling for fine sampling along a ray.
    @input
    coarse_samples_mid: [n_rays, n_samples-1]
    weights: [n_rays, n_samples-2]
    @output
    samples: [n_rays, n_fine_samples]
    '''
    # get pdf
    weights = weights + 1e-5 
    pdf = weights / torch.sum(weights, dim=1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], dim=-1) 

    # uniform samples first between [0, 1]
    u = torch.rand(cdf.shape[0], opt.n_importance).to(opt.device)
    np.save('./u.npy', u.detach().cpu().numpy())

    # invert cdf and find indices
    u = u.contiguous()
    indices = torch.searchsorted(cdf, u, right=True) 
    indices_below = torch.clamp(indices - 1, min=0)
    indices_above = torch.clamp(indices, max=cdf.shape[1] - 1)
    indices_both_sided = torch.stack([indices_below, indices_above], dim=-1)
    
    # get the values used for sampling  
    expand_shape = [cdf.shape[0], opt.n_importance, cdf.shape[1]] # 800, 18, 63
    cdf_both_sided = torch.gather(cdf.unsqueeze(1).expand(expand_shape), 2, indices_both_sided)
    coarse_samples_both_sided = torch.gather(coarse_samples_mid.unsqueeze(1).expand(expand_shape), 2, indices_both_sided)

    # compute the sampled value 
    denom = cdf_both_sided[:, :, 1] - cdf_both_sided[:, :, 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_both_sided[:, :, 0]) / denom
    samples = coarse_samples_both_sided[:, :, 0] + t * (coarse_samples_both_sided[:, :, 1] - coarse_samples_both_sided[:, :, 0])
    return samples

def stratified_sampling(t_n, t_f, n_samples):
    """
    Stratified sampling of the ray interval [t_n, t_f]
    Returns: [n_samples, 1]
    """
    ts = np.linspace(0.0, 1.0, n_samples+1)
    zs = t_n * (1. - ts) + t_f * ts
    samples = np.array([np.random.uniform(zs[i], zs[i+1], 1) for i in range(n_samples)])
    return torch.tensor(samples)