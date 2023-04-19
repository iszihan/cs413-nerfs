import numpy as np

from data.dataset import NerfDataset
import common.losses as losses
import torch
from torch.utils.data import DataLoader
from nerf.nerf import NerfModel
from nerf.train import train 
from nerf.test import test
from common.vol_rendering import volumetric_rendering_per_image as render
from torchvision.utils import save_image
import click 
import argparse
from common.util import str2bool


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default='train', help='dataset', choices=['train', 'test', 'few shot'])
    parser.add_argument('--op', type=str, default='train', help='operation', choices=['train', 'test'])
    parser.add_argument('--batch_rays', type=int, default=200, help="ray batch size")
    parser.add_argument('--batch_imgs', type=int, default=1, help="image batch size")
    parser.add_argument('--epoch', type=int, default=3, help="epoch size")
    parser.add_argument('--total_steps', type=int, default=200000, help="total steps")
    parser.add_argument('--outdir', type=str, default='./output/lego/', help="output directory")
    parser.add_argument('--expname', type=str, default='occ_reg', help="experiment name")
    parser.add_argument('--n_samples', type=int, default=64, help='number of point samples along a ray for stratefied sampling')
    parser.add_argument('--n_importance', type=int, default=128, help='number of points sampled along a ray for importance sampling')
    parser.add_argument('--iter_center', type=int, default=500, help='number of iterations for center cropping')
    parser.add_argument('--iter_coarse', type=int, default=800, help='number of iterations for coarse training')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint path')
    # logging 
    parser.add_argument('--log_img', type=str2bool, default=True, help='log image during training.')
    # test 
    parser.add_argument('--save_img', type=str2bool, default=False, help='save image only during testing.')
    # regularization 
    parser.add_argument('--freq_mask', type=str2bool, default=False, help='FREENeRF frequency regularization.')
    parser.add_argument('--freq_reg_steps', type=int, default=160000, help='FREENeRF frequency regularization steps \
                         ( 0.8 * total steps = 0.8 * 200000).')
    parser.add_argument('--occ_reg_weight', type=float, default=0.2, help='occlusion regularization weight')
    parser.add_argument('--occ_index', type=int, default=10, help='occlusion reg indices with weight 1 upto occ_index')
    parser.add_argument('--dataset_name', type=str, default='llff', help='dataset name')

    opt = parser.parse_args()
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # some config bindings 
    if opt.op == 'test': opt.dataset = 'test'
    if opt.freq_mask: opt.dataset = 'few shot'
    
    # Construct dataset 
    dataset = NerfDataset(dataset=opt.dataset_name, mode=opt.dataset)
    opt.h, opt.w, opt.focal, opt.near, opt.far = dataset.getConstants()

    # Construct dataloader for coarse training
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Construct nerf model
    model = NerfModel(use_viewdirs=True).to(opt.device)
    model = torch.nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    # Load checkpoint
    if opt.checkpoint is not None:
        checkpoint = torch.load(opt.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        opt.global_step = checkpoint['global_step']
        opt.epoch = checkpoint['epoch']
        print('Loaded checkpoint from epoch', opt.epoch)

    # Train
    if opt.op == 'train':
        train(dataloader, model, optimizer, opt)
    elif opt.op == 'test':
        test(dataloader, model, opt)


if __name__ == '__main__':
    main()









