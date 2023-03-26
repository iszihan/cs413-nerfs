import numpy as np

from data.dataset import NerfDataset
import common.losses as losses
import torch
from torch.utils.data import DataLoader
from nerf.nerf import NerfModel
from common.vol_rendering import volumetric_rendering as render
from torchvision.utils import save_image

# Construct dataset
train_dataset = NerfDataset(dataset='blender', mode='test')
h, w, focal, near, far = train_dataset.getConstants()

# Construct dataloader for coarse training
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# Construct nerf model
model = NerfModel(use_viewdirs=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Train
batch_size = 20

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










