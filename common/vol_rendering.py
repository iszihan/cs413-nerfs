import numpy as np
import torch
import nerf.nerf as nerf


def expected_colour(rays, t_n, t_f, n_samples):
    """
    rays = [origin, direction]
    t_n, t_f = ray interval
    n_samples = number of samples

    Expected colour of a ray, given the ray interval [t_n, t_f] and the number of samples
    Returns: [1, 3]
    """
    samples = torch.tensor(stratified_sampling(t_n, t_f, n_samples))
    pts = rays[0] + rays[1] * samples
    input = torch.cat([pts, rays[1].repeat(n_samples, 1)], dim=1)
    output = nerf.nerf(input)
    rgb = output[:, :3]
    density = output[:, 3:]
    weighted_density = density * (samples[1:] - samples[:-1])
    accumulated_transmittance = torch.exp(-torch.cumsum(weighted_density, dim=0))
    weights = 1. - torch.exp(-weighted_density)
    colour_pred = torch.sum(weights * accumulated_transmittance * rgb, dim=0)
    return colour_pred


def stratified_sampling(t_n, t_f, n_samples):
    """
    Stratified sampling of the ray interval [t_n, t_f]
    Returns: [n_samples, 1]
    """
    ts = np.linspace(0.0, 1.0, steps=n_samples)
    zs = t_n * (1. - ts) + t_f * ts
    samples = np.array([np.random.uniform(zs[i], zs[i+1], 1) for i in range(n_samples-1)])
    return samples