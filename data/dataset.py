"""
    DTU dataset.
    Selena.
    common/utils.py or common/rays.py will have the transformation
"""
from common.util import spherical_poses
from common.rays import get_rays
import os 
import json 
import imageio
import torch
import numpy as np 
class NerfDataset():
    def __init__(self, dataset='blender', mode='train'):
        self.mode = mode
        self.dataset = dataset 
        if dataset=='blender':
            # referencing https://github.com/yenchenlin/nerf-pytorch/blob/223fe62d87d641e2bb0fb5bdbbcb6dad5efb2af3/run_nerf.py
            self.imgs, self.poses, self.h, self.w, self.f = self.get_blender_dataset('./dataset/nerf_synthetic/lego')        
            # self.poses: 100,4,4
            # Construct Intrinsic 
            self.K = np.array([
                            [self.f, 0, 0.5 * self.w],
                            [0, self.f, 0.5 * self.h],
                            [0, 0, 1]])
            self.near = 2
            self.far = 6
        
        self.rays_rgb = {k: get_rays(self.h, self.w, self.K, self.poses[k], self.imgs[k]) for k in self.imgs}

    def get_blender_dataset(self, basedir):
        data = {}
        categories = ['train', 'val', 'test']
        for c in categories:
            with open(os.path.join(basedir, f'transforms_{c}.json'),'r') as fp:
                data[c] = json.load(fp)
        allimgs = {}
        allposes = {}
        for c in categories:
            d = data[c]
            imgs = [] 
            poses = []
            for f in d['frames']:
                filename = os.path.join(basedir, f['file_path'] + '.png')
                imgs.append(imageio.imread(filename))
                poses.append(np.array(f['transform_matrix']))
            imgs = (np.array(imgs) / 255.).astype(np.float32)
            poses = np.array(poses).astype(np.float32)
            allimgs[c] = imgs 
            allposes[c] = poses
        
        h,w = allimgs[categories[0]][0].shape[:2]
        camera_angle_x = float(d['camera_angle_x'])
        focal = .5 * w / np.tan(.5 * camera_angle_x)
        return allimgs, allposes, h, w, focal

    def __len__(self):
        return len(self.imgs[self.mode])

    def __getitem__(self, i):
        return torch.from_numpy(self.rays_rgb[self.mode][i]).float() #[3: rayo + rayd + rgb, h, w, 3] 
