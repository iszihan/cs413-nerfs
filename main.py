from data.dataset import NerfDataset
from common.util import toNP
import torch 
from torch.utils.data import DataLoader
from nerf.hierarchical_sampler import sample_coarse
from nerf.nerf import NerfModel

# Construct dataset 
train_dataset = NerfDataset(dataset='blender', mode='test')

# Construct dataloader for coarse training 
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# Construct nerf model 
model = NerfModel(use_viewdirs=True)

# Train 
rays_per_image = next(iter(train_dataloader))
coarse_input = sample_coarse(rays_per_image.squeeze(), 10) #h*w,n,6

# Batchify rays 
batch_size = 800
for i in range(0,coarse_input.shape[0],batch_size):
    batched_input = coarse_input[i:i+batch_size]
    output = model(batched_input) #nb, 10, 6
    #Kinjal: render rays 











