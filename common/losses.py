import torch


def loss(pred, target, pred_density=None, occ_reg_weight=0, occ_index=10):
    """
    pred: [n, 4] rgb + density
    target: [n, 3] rgb
    """
    l = loss_mse(pred, target)
    if occ_reg_weight > 0:
        occ_reg = occlusion_regularization(pred_density, occ_index)
        l += occ_reg_weight * occ_reg
    return l


def loss_mse(pred, target):
    return torch.mean((pred - target) ** 2)


def rendering_loss(data_imgs, pred_course, pred_fine):
    """
    data_imgs, pred_course, pred_fine: [b, h, w, 3]
    """
    return torch.mean(torch.sum(torch.norm(data_imgs - pred_course, dim=3)**2, dim=(1, 2))
                      + torch.sum(torch.norm(data_imgs - pred_fine, dim=3)**2, dim=(1, 2)))


def occlusion_regularization(pred_density, occ_index=10):
    """
    pred_density: [n, n_samples]
    """
    mask = torch.zeros_like(pred_density)
    mask[:, :occ_index] = 1
    return torch.mean(mask*pred_density)
