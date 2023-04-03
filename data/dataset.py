"""
    DTU dataset.
    Selena.
    common/utils.py or common/rays.py will have the transformation
"""

from common.util import spherical_poses
from common.rays import get_rays
import os 
import json 
import cv2
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
        
        # self.rays_rgb = {k: get_rays(self.h, self.w, self.K, self.poses[k], self.imgs[k]) for k in self.imgs}

    def get_blender_dataset(self, basedir, half_res=True):
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
            #use white background
            imgs = imgs[...,:3]*imgs[...,-1:] + (1.-imgs[...,-1:])
            poses = np.array(poses).astype(np.float32)
            allimgs[c] = imgs 
            allposes[c] = poses
        h,w = allimgs[categories[0]][0].shape[:2]
        camera_angle_x = float(d['camera_angle_x'])
        focal = .5 * w / np.tan(.5 * camera_angle_x)

        if half_res:
            h = h//2
            w = w//2
            focal = focal/2.
            imgs_half_res = {}
            for i, c in enumerate(allimgs):
                imgs_half_res[c] = []
                for img in allimgs[c]:
                    imgs_half_res[c].append(cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA))
                imgs_half_res[c] = np.array(imgs_half_res[c])
            allimgs = imgs_half_res
        return allimgs, allposes, h, w, focal

    def __len__(self):
        return len(self.imgs[self.mode])

    def getConstants(self):
        return self.h, self.w, self.f, self.near, self.far

    def __getitem__(self, i):
        # print(self.imgs[self.mode][i])
        # print(self.imgs[self.mode][i].shape)
        # print(self.poses[self.mode][i])
        # exit()
        rays_rgb = get_rays(self.h, self.w, self.K, self.poses[self.mode][i:i + 1], self.imgs[self.mode][i:i + 1])
        rays_rgb = torch.tensor(rays_rgb.squeeze(0))
        rays_rgb = rays_rgb.permute(1,2,0,3).reshape(rays_rgb.shape[1],rays_rgb.shape[2],9) # h, w, 9 (rayo + rayd + rgb)
        # print(rays_rgb[...,:3])
        # print(rays_rgb[...,3:6])
        # exit()
        return rays_rgb
