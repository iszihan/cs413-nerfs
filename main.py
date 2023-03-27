import numpy as np

from data.dataset import NerfDataset
import common.losses as losses
import torch
from torch.utils.data import DataLoader
from nerf.nerf import NerfModel
from nerf.train import train 
from common.vol_rendering import volumetric_rendering_per_image as render
from torchvision.utils import save_image
import click 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_rays', type=int, default=800, help="ray batch size")
parser.add_argument('--batch_imgs', type=int, default=1, help="image batch size")
parser.add_argument('--epoch', type=int, default=200, help="epoch size")
parser.add_argument('--outdir', type=str, default='./output', help="output directory")
parser.add_argument('--expname', type=str, default='trial', help="experiment name")
opt = parser.parse_args()

opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Construct dataset
train_dataset = NerfDataset(dataset='blender', mode='test')
opt.h, opt.w, opt.focal, opt.near, opt.far = train_dataset.getConstants()

# Construct dataloader for coarse training
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# Construct nerf model
model = NerfModel(use_viewdirs=True).to(opt.device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Train
train(train_dataloader, model, optimizer, opt)

exit()
rays_rgb_image = next(iter(train_dataloader))  # 1, h, w, 9
rays = rays_rgb_image[:, :, :, :6]  # 1, h, w, 6
rgb = rays_rgb_image[:, :, :, 6:]  # 1, h, w, 3
pred = render(model, near, far, 10, rays)[0]
np.save('pred.npy', pred.detach().numpy())
print(torch.max(pred))
print(torch.min(pred))
print(pred.shape)
print(pred.dtype)
pred = torch.round(255*pred).clip(0,255).to(torch.uint8)
print(pred.dtype)
loss = losses.loss_mse(pred, rgb)
print(loss)
save_image(pred, 'pred.png')

if __name__ == '__main__':
    main()









