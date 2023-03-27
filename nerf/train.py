
import tensorboardX
import os 
from common.vol_rendering import volumetric_rendering_per_image as render_image
from common.vol_rendering import volumetric_rendering_per_ray as render_ray
import numpy as np
import torch 
from common.util import writable_image

def eval():
    return None

def train_one_epoch(loader, model, optimizer, opt):

    for i, img in enumerate(loader):
        # batchify rays 
        n_rays = img.shape[0] * img.shape[1] * img.shape[2] 
        ray_indices = np.array(range(n_rays))
        np.random.shuffle(ray_indices)
        n_batches = n_rays // opt.batch_rays
        for i_batch in range(n_batches):
            optimizer.zero_grad()
            opt.global_step += 1
            batch_indices = ray_indices[i_batch * opt.batch_rays : (i_batch + 1) * opt.batch_rays]
            batch_indices = np.unravel_index(batch_indices, (img.shape[0], img.shape[1], img.shape[2]))
            batch_indices = np.stack(batch_indices, axis=1)
            batch_rays = img[batch_indices[:, 0], batch_indices[:, 1], batch_indices[:, 2], :6].to(opt.device)
            batch_rgb = img[batch_indices[:, 0], batch_indices[:, 1], batch_indices[:, 2], 6:].to(opt.device)            
            batch_pred = render_ray(model, opt.near, opt.far, 64, batch_rays, opt) #nb,4
            loss = torch.mean((batch_pred[:, :3] - batch_rgb) ** 2)
            opt.writer.add_scalar('loss', loss, opt.global_step)
            print('loss:', loss)
            loss.backward()
            optimizer.step()
        
            if i_batch % 1 == 0:
                # save image
                with torch.no_grad():
                    rays = img[:1, :, :, :6].to(opt.device)
                    pred = render_image(model, opt.near, opt.far, 64, rays, opt=opt)[0]
                    gt = img[:, :, :, 6:].to(opt.device)[0]
                    opt.writer.add_image('pred', writable_image(pred.permute(2,0,1)), opt.global_step)
                    opt.writer.add_image('gt', writable_image(gt.permute(2,0,1)), opt.global_step)
            

def train(train_dataloader, model, optimizer, opt):
    model.train()
    # logging 
    opt.writer = tensorboardX.SummaryWriter(os.path.join(opt.outdir, opt.expname, 'logs'))
    opt.global_step = 0
    for i_epoch in range(opt.epoch):
        train_one_epoch(train_dataloader, model, optimizer, opt)
        save_checkpoint(model, optimizer, opt, i_epoch)
    
def save_checkpoint():
    return None
