#from tqdm.auto import tqdm
import tensorboardX
import os 
from common.vol_rendering import volumetric_rendering_per_image as render_image
from common.vol_rendering import volumetric_rendering_per_ray as render_ray
from common.losses import loss
import numpy as np
import torch 
import tqdm 
from common.network import create_freq_mask
from common.util import writable_image, printarr

def train_one_epoch(loader, model, optimizer, opt):
    
    for i, img in enumerate(loader):
        nb, h, w = img.shape[:3]

        # center cropping for first 500 images 
        if opt.global_step < opt.iter_center and h>256 and w>256:
            # center cropping 
            dh = int(h//2 * 0.5)
            dw = int(w//2 * 0.5)
            coords = torch.stack(torch.meshgrid(torch.linspace(h//2-dh, h//2+dh-1, 2*dh), 
                                                torch.linspace(w//2-dw, w//2+dw-1, 2*dw)),
                                                -1)
        else:
            coords = torch.stack(torch.meshgrid(torch.linspace(0, h-1, h), 
                                            torch.linspace(0, w-1, w)), -1) # (H, W, 2)
            
        coords = torch.reshape(coords, [-1,2]) #h*w, 2
        n_rays = coords.shape[0]
        ray_indices = np.array(range(coords.shape[0]))
        np.random.shuffle(ray_indices)
        n_batches = n_rays // opt.batch_rays
        if i==0:
            pbar = tqdm.tqdm(total=len(loader) * n_batches, 
                             bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for i_batch in range(n_batches):
            optimizer.zero_grad()
            opt.global_step += 1

            batch_indices = ray_indices[i_batch * opt.batch_rays : (i_batch + 1) * opt.batch_rays]
            batch_indices = coords[batch_indices].long()
            batch_rays = img[0, batch_indices[:, 0], batch_indices[:, 1], :6].to(opt.device)
            batch_rgb = img[0, batch_indices[:, 0], batch_indices[:, 1], 6:].to(opt.device) 

            if i_batch / n_batches < 0.3:
                 opt.fine_sampling = False 
                 batch_pred, density = render_ray(model, opt.near, opt.far, 64, batch_rays, opt) #nb,4
            else:
                opt.fine_sampling = True
                batch_pred, density = render_ray(model, opt.near, opt.far, 64, batch_rays, opt) #nb,4

            l = l(batch_pred[:, :3], batch_rgb, density, opt.occ_reg_weight, opt.occ_index)#torch.mean((batch_pred[:, :3] - batch_rgb) ** 2)
            l.backward()
            optimizer.step()

            if i_batch % 10 == 0:
                opt.writer.add_scalar('loss', loss, opt.global_step)
            
            # log image
            if i_batch % 100 == 0:
                with torch.no_grad():
                    rays = img[:1, :, :, :6].to(opt.device)
                    pred = render_image(model, opt.near, opt.far, 64, rays, opt=opt)[0]
                    gt = img[:, :, :, 6:].to(opt.device)[0]
                    opt.writer.add_image('pred', writable_image(pred.permute(2, 0, 1)), opt.global_step)
                    opt.writer.add_image('gt', writable_image(gt.permute(2, 0, 1)), opt.global_step)

            if opt.global_step % 10000 == 0 or opt.global_step == opt.total_steps:
                save_checkpoint(model, optimizer, opt.global_epoch, opt)
                print(f'saved checkpoint at global step {opt.global_step}')
                
            pbar.set_description(f"loss={loss:.4f}")
            pbar.update(1)

def train(train_dataloader, model, optimizer, opt):
    
    # on keyboard interrupt, save model
    try:
        model.train()
        # logging
        opt.writer = tensorboardX.SummaryWriter(os.path.join(opt.outdir, opt.expname, 'logs'))
        opt.global_step = 0
        opt.global_epoch = 0
        while opt.global_step < opt.total_steps: 
            train_one_epoch(train_dataloader, model, optimizer, opt)
            opt.global_epoch += 1
            if opt.global_epoch % 1 == 0:
                save_checkpoint(model, optimizer, opt.global_epoch, opt)
                print(f'saved checkpoint at epoch {opt.global_epoch} and global step {opt.global_step}')

    except KeyboardInterrupt:
        save_checkpoint(model, optimizer, opt.global_epoch, opt)
        print('saved checkpoint at step {}'.format(opt.global_step))
        print('training interrupted')

def save_checkpoint(model, optimizer, epoch, opt):
    # save model
    torch.save({
        'epoch': epoch,
        'global_step': opt.global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(opt.outdir, opt.expname, f'checkpoint_e{epoch}_s{opt.global_step}.pt'))
