" rendering, ray marching, query rays"
import numpy as np 

def get_rays(h,w,K,p,imgs):
    '''
    Get rays from an image grid with intrinsic and extrinsic matrix.
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

# Kinjal