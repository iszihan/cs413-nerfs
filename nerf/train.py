from tqdm.auto import tqdm
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
        nb, h, w = img.shape[:3]

        # center cropping for first 500 images 
        if opt.global_step < opt.iter_coarse and h>256 and w>256:
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
        for i_batch in range(n_batches):
            optimizer.zero_grad()
            opt.global_step += 1

            batch_indices = ray_indices[i_batch * opt.batch_rays : (i_batch + 1) * opt.batch_rays]
            batch_indices = coords[batch_indices].long()
            batch_rays = img[0, batch_indices[:, 0], batch_indices[:, 1], :6].to(opt.device)
            batch_rgb = img[0, batch_indices[:, 0], batch_indices[:, 1], 6:].to(opt.device) 

            batch_pred = render_ray(model, opt.near, opt.far, 64, batch_rays, opt) #nb,4

            loss = torch.mean((batch_pred[:, :3] - batch_rgb) ** 2)
            loss.backward()
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


def train(train_dataloader, model, optimizer, opt):
    # on keyboar interrupt, save model
    try:
        model.train()
        # logging
        opt.writer = tensorboardX.SummaryWriter(os.path.join(opt.outdir, opt.expname, 'logs'))
        opt.global_step = 0
        # use tqdm to show progress bar
        for i_epoch in tqdm(range(opt.epoch)):
            train_one_epoch(train_dataloader, model, optimizer, opt)
            if i_epoch % 25 == 0 or i_epoch == opt.epoch - 1:
                save_checkpoint(model, optimizer, opt, i_epoch)
                print('saved checkpoint at epoch {}'.format(i_epoch))

    except KeyboardInterrupt:
        save_checkpoint(model, optimizer, opt, opt.epoch)
        print('saved checkpoint at epoch {}'.format(opt.epoch))
        print('training interrupted')


def save_checkpoint(model, optimizer, opt, epoch):
    # save model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(opt.outdir, opt.expname, 'checkpoint_{}.pt'.format(epoch)))
