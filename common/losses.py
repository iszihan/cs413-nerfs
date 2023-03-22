import torch

def rendering_loss(data_imgs, pred_course, pred_fine):
    """
    data_imgs, pred_course, pred_fine: [b, h, w, 3]
    """
    return torch.mean(torch.sum(torch.norm(data_imgs - pred_course, dim=3)**2, dim=(1, 2))
                      + torch.sum(torch.norm(data_imgs - pred_fine, dim=3)**2, dim=(1, 2)))
