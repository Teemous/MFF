import torch
import torch.nn.functional as F
import torch.nn as nn

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

# 计算梯度
def grad(img):
    dx = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
    dy = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
    # pad补齐到原尺寸
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return dx + dy

# 光照平滑损失
class TVloss(nn.Module):
    def __init__(self, sigma=0.1):
        """
        高斯加权的 Total Variation Loss。
        sigma: 控制边缘宽度的标准差 (默认0.1, 可调 0.05~0.3)
        """
        super(TVloss, self).__init__()
        self.sigma = sigma

    def forward(self, x, weight_map=None):
        self.h_x = x.size()[2]
        self.w_x = x.size()[3]
        self.batch_size = x.size()[0]

        if weight_map is None:
            # 无引导图 → 普通TV
            self.TVLoss_weight = (1, 1)
        else:
            # 使用高斯型权重
            self.TVLoss_weight = self.compute_weight(weight_map)

        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])

        h_tv = (self.TVLoss_weight[0] * torch.abs(x[:, :, 1:, :] - x[:, :, :self.h_x - 1, :])).sum()
        w_tv = (self.TVLoss_weight[1] * torch.abs(x[:, :, :, 1:] - x[:, :, :, :self.w_x - 1])).sum()

        return (h_tv / count_h + w_tv / count_w) / self.batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

    def compute_weight(self, img):
        """
        计算高斯型边缘权重：
        W = exp(-(grad^2)/(2*sigma^2))
        """
        gradx = torch.abs(img[:, :, 1:, :] - img[:, :, :self.h_x - 1, :])
        grady = torch.abs(img[:, :, :, 1:] - img[:, :, :, :self.w_x - 1])

        # 高斯型边缘抑制权重
        TVLoss_weight_x = torch.exp(- (gradx ** 2) / (2 * self.sigma ** 2))
        TVLoss_weight_y = torch.exp(- (grady ** 2) / (2 * self.sigma ** 2))

        return TVLoss_weight_x, TVLoss_weight_y