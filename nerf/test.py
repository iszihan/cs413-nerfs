#from tqdm.auto import tqdm
import tensorboardX
import os 
from common.vol_rendering import volumetric_rendering_per_image as render_image
from common.vol_rendering import volumetric_rendering_per_ray as render_ray
import numpy as np
import torch 
import tqdm 
from common.util import *
import lpips 
import json 
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.0]).to(x.device))
lpips_fn = lpips.LPIPS(net='vgg')
lpips_fn.to('cuda')

def test(dataloader, model, opt):

    #model.eval()
    opt.fine_sampling = True
    
    ret = {}
    psnrs = [] 
    lpipss = []
    
    img_to_save = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180]
    for i, img in enumerate(dataloader):
        if opt.save_img:
            if i not in img_to_save: 
                pbar.update(1)
                continue
        
        nb, h, w = img.shape[:3]
        coords = torch.stack(torch.meshgrid(torch.linspace(0, h-1, h), 
                                            torch.linspace(0, w-1, w)), -1) # (H, W, 2)
        coords = torch.reshape(coords, [-1,2]) #h*w, 2
        if i==0:
            pbar = tqdm.tqdm(total=len(dataloader), 
                                bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        with torch.no_grad():
            rays = img[:1, :, :, :6].to(opt.device)
            pred = render_image(model, opt.near, opt.far, 64, rays, opt=opt)[0]
            gt = img[:, :, :, 6:].to(opt.device)[0]
            if opt.save_img:
                save_image(toNP(writable_image(pred[...,:3])), f'{opt.outdir}/{opt.expname}/pred_{i}.png')
                save_image(toNP(writable_image(gt)), f'{opt.outdir}/{opt.expname}/gt_{i}.png')
            else:
                # compute psnr 
                psnr = mse2psnr(torch.mean((pred[...,:3] - gt) ** 2))
                # compute lpips  
                lpips = lpips_fn(pred[...,:3].permute(2,0,1).unsqueeze(0), gt.permute(2,0,1).unsqueeze(0))
                psnrs.append(psnr)
                lpipss.append(lpips)
        pbar.update(1)
    
    if not opt.save_img:
        psnrs = torch.stack(psnrs)
        lpipss = torch.stack(lpipss)
        mean_psnr = torch.mean(psnrs)
        mean_lpips = torch.mean(lpipss)
    
        ret['mean psnr'] = mean_psnr.cpu().numpy()
        ret['mean lpips'] = mean_lpips.cpu().numpy()
        ret['psnrs'] = psnrs.cpu().numpy()
        ret['lpipss'] = lpipss.cpu().numpy()
    
        # write results
        np.save(f'{opt.outdir}/{opt.expname}/psnrs.npy', ret['psnrs'])
        np.save(f'{opt.outdir}/{opt.expname}/lpipss.npy', ret['lpipss'])
        np.save(f'{opt.outdir}/{opt.expname}/psnr.npy', ret['mean psnr'])
        np.save(f'{opt.outdir}/{opt.expname}/lpips.npy', ret['mean lpips'])