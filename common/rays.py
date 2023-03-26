" rendering, ray marching, query rays"
import numpy as np
import torch

def get_rays(h,w,K,p,imgs):
    '''
    Get rays from an image grid with intrinsic and extrinsic matrix.
    @h, w: image height and width
    @K: intrinsic matrix - [3, 3] matrix
    @p: extrinsic matrix - [num_images, 3, 4] matrix
    @imgs: images - [num_images, h, w, 3]

    Returns:
        ray origin, ray direction, rgb of the pixels in the image grid
        [num_images, 3, h, w, 3]
    '''
    # create image grid 
    i, j = np.meshgrid(np.arange(w, dtype=np.float32),
                       np.arange(h, dtype=np.float32))
    dirs = np.stack([(i-K[0][2])/K[0][0], 
                    -(j-K[1][2])/K[1][1], 
                        -np.ones_like(i)], -1) # [800, 800, 3]
    rays = []
    for _p in p:
        raysd = np.dot(dirs[..., :].reshape(-1,3), _p[:3,:3]).reshape(dirs.shape[0], dirs.shape[1],3)
        rayso = np.broadcast_to(_p[:3,-1], np.shape(raysd))
        rays.append([rayso, raysd])
    rays = np.stack(rays, 0) #200, 800, 800, 3 
    rays_rgb = np.concatenate([rays, imgs[:,None,:,:,:3]],1)
    return rays_rgb


def get_coord(origin, direction, t):
    '''
    Get coordinate for a given ray at parametric distance t.
    '''
    return origin + direction * t


def get_rays_torch(h, w, focal, c2w):
    '''
    @h, w: image height and width
    @focal: focal length
    @c2w: camera to world matrix - [b, 3, 4] matrix

    Return:
    @rays: [b, h, w, 2, 3]
    '''
    # create image grid
    i, j = torch.meshgrid(torch.arange(w, dtype=torch.float32),
                            torch.arange(h, dtype=torch.float32))

    # create ray directions
    # camera w.r.t itself is 0, 0, 0
    # rays hit focal length distance away from camera
    # z = 1 because camera is shooting rays in z direction
    dirs = torch.stack([(i - w * 0.5) / focal,
                        -(j - h * 0.5) / focal,
                        -torch.ones_like(i)], -1)        # [h, w, 3]
    dirs = dirs / torch.linalg.norm(dirs, dim=-1, keepdim=True) # [h, w, 3]

    # transform ray directions to world space
    # c2w is a 3x4 matrix
    # c2w[:3, :3] is the rotation matrix
    # c2w[:3, -1] is the translation vector
    dirs_world = (dirs[..., None, :] @ c2w[:, None, None, :, :]).squeeze(-2)
    # [h, w, 1, 3] @ [b, 1, 1, 3, 3] -> [b, h, w, 1, 3] -> [b, h, w, 3]

    origin = c2w[:, :, -1]
    oris = origin[:, None, None, :].repeat(1, h, w, 1)

    return torch.concatenate([oris[..., None, :], dirs_world[..., None, :]], -2) # [b, h, w, 2, 3]

