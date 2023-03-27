import numpy as np
import torch
import nerf.nerf as nerf
import common.rays as rays_module

def volumetric_rendering_per_ray(model, t_n, t_f, n_samples=10, rays=None):
    """"
    @input:
    rays: [nr, 6]
    @return:
    imgs: [b, h, w, 3]
    """
    cam_rays = rays.reshape(rays.shape[0], 2, 3)
    rgba = expected_colour(model, cam_rays, t_n, t_f, n_samples)
    return rgba

def volumetric_rendering_per_image(model, t_n, t_f, n_samples=10, rays=None, h=None, w=None, focal=None, c2w=None):
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

    imgs = torch.zeros((cam_rays.shape[0], h, w, 4))

    # get expected colour for each ray for each camera
    for cam in range(cam_rays.shape[0]):
        # get rays for each camera
        cam_rays_i = cam_rays[cam] # [h, w, 2, 3]
        # get expected colour for each ray
        rgba = expected_colour(model, cam_rays_i.reshape(-1, 2, 3), t_n, t_f, n_samples)
        # assign colour to image
        imgs[cam, :, :, :] = rgba.reshape(h, w, 4)
    return imgs


def expected_colour(model, rays, t_n, t_f, n_samples):
    """
    rays = [origin, direction] of shape [n_rays, 2, 3]
    t_n, t_f = ray interval
    n_samples = number of samples

    Expected colour, alpha of the rays
    Returns: [n_rays, 4]
    """
    samples = stratified_sampling(t_n, t_f, n_samples)
    pts = rays[:, 0, :].repeat(1, n_samples).reshape(-1, 3) + \
          rays[:, 1, :].repeat(1, n_samples).reshape(-1, 3) * samples.repeat(rays.shape[0], 3)
    input = torch.cat([pts, rays[:, 1, :].repeat(1, n_samples).reshape(-1, 3)], dim=1) # [n_rays * n_samples, 6]
    output = model(input.reshape(rays.shape[0], n_samples, 6).float()).reshape(-1, 4)         # [n_rays*n_samples, 4]
    rgb = output[:, :3]
    density = output[:, 3]
    temp = torch.cat([samples[1:] - samples[:-1], torch.tensor([[t_f - samples[-1]]])], dim = 0).squeeze().repeat(rays.shape[0])
    weighted_density = density * temp
    accumulated_transmittance = torch.exp(-torch.cumsum(weighted_density, dim=0))
    weights = torch.ones_like(weighted_density) - torch.exp(-weighted_density)
    colour_pred = torch.sum(((weights * accumulated_transmittance)[:, None].repeat(1, 3) * rgb).reshape(rays.shape[0], n_samples, 3), dim=1)
    #print(colour_pred.shape)
    alpha = accumulated_transmittance.reshape(rays.shape[0], n_samples)[:, -1].reshape(rays.shape[0], 1)
    #print(alpha.shape)
    output = torch.cat([colour_pred, alpha], dim = 1)
    #print(output.shape)
    return output


def stratified_sampling(t_n, t_f, n_samples):
    """
    Stratified sampling of the ray interval [t_n, t_f]
    Returns: [n_samples, 1]
    """
    ts = np.linspace(0.0, 1.0, n_samples+1)
    zs = t_n * (1. - ts) + t_f * ts
    samples = np.array([np.random.uniform(zs[i], zs[i+1], 1) for i in range(n_samples)])
    return torch.tensor(samples)