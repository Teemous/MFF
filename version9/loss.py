import torch
import torch.nn.functional as F

# 极端化损失
def Entropy_Loss(M1):
    L_entropy = -torch.mean(
        M1 * torch.log(M1 + 1e-8) + (1 - M1) * torch.log(1 - M1 + 1e-8)
    )
    '''L_entropy += -torch.mean(
        M2 * torch.log(M2 + 1e-8) + (1 - M2) * torch.log(1 - M2 + 1e-8)
    )'''
    return L_entropy
def ssim_loss(x, y):
    """简易结构相似性损失"""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)
    sigma_x = F.avg_pool2d(x * x, 3, 1, 1) - mu_x ** 2
    sigma_y = F.avg_pool2d(y * y, 3, 1, 1) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
           ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
    return 1 - ssim.mean()