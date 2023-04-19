"""
    DTU dataset.
    Selena.
    common/utils.py or common/rays.py will have the transformation
"""
from common.rays import get_rays
import os 
import json 
import cv2
import imageio
import torch
import numpy as np
from data.llff import load_llff_data

class NerfDataset():
    def __init__(self, dataset='blender', mode='train'):
        self.mode = mode
        self.dataset = dataset 
        if dataset=='blender':
            # referencing https://github.com/yenchenlin/nerf-pytorch/blob/223fe62d87d641e2bb0fb5bdbbcb6dad5efb2af3/run_nerf.py
            self.imgs, self.poses, self.h, self.w, self.f = self.get_blender_dataset('./data/nerf_synthetic/lego')
            # self.poses: 100,4,4
            # Construct Intrinsic 
            self.K = np.array([
                            [self.f, 0, 0.5 * self.w],
                            [0, self.f, 0.5 * self.h],
                            [0, 0, 1]])
            self.near = 2
            self.far = 6
        
        elif dataset =='llff':
            self.imgs, self.poses, bds, hwf = self.get_llff_data('./data/nerf_llff_data/fern')
            self.h, self.w, self.f = hwf
            self.near, self.far = bds
            self.K = np.array([
                            [self.f, 0, 0],
                            [0, self.f, 0],
                            [0, 0, 1]])

    def get_blender_dataset(self, basedir, half_res=True):
        data = {}
        categories = ['train', 'val', 'test']

        few_shot_indices = ['1', '7', '10', '13', '15']
        few_shot_imgs = [] 
        few_shot_poses = []
        
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
                img = imageio.imread(filename)
                pose = np.array(f['transform_matrix'])
                if c == 'train' and f['file_path'].split('_')[-1].split('.')[0] in few_shot_indices:
                    few_shot_imgs.append(img)
                    few_shot_poses.append(pose)
                imgs.append(img)
                poses.append(pose)
            imgs = (np.array(imgs) / 255.).astype(np.float32)
            #use white background
            imgs = imgs[...,:3]*imgs[...,-1:] + (1.-imgs[...,-1:])
            poses = np.array(poses).astype(np.float32)
            allimgs[c] = imgs 
            allposes[c] = poses
        
        few_shot_imgs = (np.array(few_shot_imgs) / 255.).astype(np.float32)
        #use white background
        few_shot_imgs = few_shot_imgs[...,:3]*few_shot_imgs[...,-1:] + (1.-few_shot_imgs[...,-1:])
        few_shot_poses = np.array(few_shot_poses).astype(np.float32)
        allimgs['few shot'] = few_shot_imgs 
        allposes['few shot'] = few_shot_poses
        
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

    def get_llff_data(self, basedir, factor=8):
        imgs, poses, bds, i_test = load_llff_data(basedir, factor)

        #use white background
        imgs = imgs[...,:3]*imgs[...,-1:] + (1.-imgs[...,-1:])

        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]

        allimgs = {}
        allposes = {}
        allimgs['test'] = np.expand_dims(imgs[i_test, ...], 0)
        allposes['test'] = np.expand_dims(poses[i_test, ...], 0)
        allimgs['train'] = np.delete(imgs, i_test, 0)
        allposes['train'] = np.delete(poses, i_test, 0)

        return allimgs, allposes, bds, hwf

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
