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
parser.add_argument('--n_samples', type=int, default=64, help='number of point samples along a ray')
opt = parser.parse_args()

opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Construct dataset
train_dataset = NerfDataset(dataset='blender', mode='train')
# train_dataset.__getitem__(0)
# exit()
opt.h, opt.w, opt.focal, opt.near, opt.far = train_dataset.getConstants()

# Construct dataloader for coarse training
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)


# Construct nerf model
model = NerfModel(use_viewdirs=True).to(opt.device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

# Train
train(train_dataloader, model, optimizer, opt)

if __name__ == '__main__':
    main()









