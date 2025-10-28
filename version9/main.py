import torch
import random
import numpy as np
import os
import time
import cv2 as cv
import yaml
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
from torch import nn, optim
import argparse
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from dataset import Microscopy_dataset
from Unet import UNet
from noise import *
from common import *
from loss import *

# 参数解析
def parse():
    parser = argparse.ArgumentParser(description="Train UNet")
    parser.add_argument('--config', default='./config/config.yaml', help='Path to config file')
    args = parser.parse_args()
    return args

# 随机种子
def set_random_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(config):
    # 固定随机种子
    set_random_seed(config["rand_seed"])
    # 学习率
    lr = config["lr"]
    # 噪声
    reg_noise_std = config["reg_noise_std"]
    # 循环次数
    num_epochs = config['num_epochs']
    # 阈值
    thresh = config['thresh']
    alpha = config['alpha']
    # 输出位置
    output_path = config['output_path']
    # 加载数据集
    test_set = Microscopy_dataset(config["data_path"])
    print(f"Dataset length: {len(test_set)}")
    loader = DataLoader(test_set, batch_size=1, shuffle=False)
    # 加载设备
    device = config["device"]

    for (y1,y2,cr,cb,img_name) in loader:
        # 发送到设备
        y1 = y1.to(device)
        y2 = y2.to(device)
        cr = cr.to(device)
        cb = cb.to(device)
        # 获取图像长宽
        _, _, H, W = y1.shape

        # —————————————————————————— 图像融合部分 —————————————————————————— #
        # 输入拼接图像
        net_inputx = torch.cat([y1,y2],dim=1).detach().to(device) # 拼接两张输入图像
        # 输入噪声
        net_inputm = get_noise(like_image=net_inputx).to(device)
        # 噪声形状
        print(net_inputm.shape)
        # 图像融合网络
        netx = UNet(num_input_channels=2,
               num_output_channels=1,
               num_channels_down=[16, 32, 64, 128,128],
               num_channels_up=[16, 32, 64, 128,128],
               num_channels_skip=[4, 4, 4, 4,4],
               upsample_mode='bilinear',
               need_sigmoid=True).to(device)
        # 掩膜生成网络
        netm = UNet(num_input_channels=2,
               num_output_channels=1,
               num_channels_down=[16, 32, 64, 128],
               num_channels_up=[16, 32, 64, 128],
               num_channels_skip=[4, 4, 4, 4],
               upsample_mode='bilinear',
               need_sigmoid=True).to(device)
        ############################################### optimizer ######################################################
        parameters = []
        parameters.extend([{'params': netm.parameters()},
                           {'params': net_inputm},
                           {'params': netx.parameters()},
                           {'params': net_inputx}])

        optimizer = torch.optim.Adam(parameters, lr=lr)
        scheduler = MultiStepLR(optimizer, milestones=[200, 400, 800], gamma=0.5)

        net_input_savedm = net_inputm.detach().clone()
        noisem = net_inputm.detach().clone()

        score_map = []
        # --- 生成初始焦点图 (引导项) ---
        mask1_init = get_score_map(y1, y2, mode='blur2th')
        mask2_init = 1 - mask1_init
        score_map.append(mask1_init)
        score_map.append(mask2_init)

        pbar = tqdm(range(num_epochs), desc="DIP-MFF processing", ncols=200)
        for step in pbar:
            optimizer.zero_grad()

            if reg_noise_std > 0:
                net_inputm = net_input_savedm + noisem.normal_() * reg_noise_std

            fuse = netx(net_inputx) # 融合图像

            mask1 = netm(net_inputm) # 掩膜图像1
            mask2 = 1 - mask1 # 掩膜图像2
            ############################################### LOSS ######################################################

            # 初始化损失函数
            loss_percep = 0
            loss_recon = 0
            loss_prior = 0
            loss_ssim = 0

            # 先验损失
            loss_prior = F.l1_loss(mask1, score_map[0]) + F.l1_loss(mask2, score_map[1])
            # 熵损失
            loss_entropy = Entropy_Loss(mask1)
            # 重建损失
            recon = mask1 * y1 + mask2 * y2
            loss_recon = F.l1_loss(fuse, recon)
            # 结构损失
            loss_ssim = ssim_loss(fuse, recon)

            if step < thresh:
                total_loss = alpha * loss_prior + loss_recon + loss_ssim
                pbar.set_postfix({
                    'Loss': f'{total_loss.item():.4f}',
                    'prior': f'{loss_prior.item():.4f}',
                    'recon': f'{loss_recon.item():.4f}',
                })
            else:
                total_loss = loss_recon + loss_ssim
                pbar.set_postfix({
                    'Loss': f'{total_loss.item():.4f}',
                    'prior': f'{loss_prior.item():.4f}',
                    'recon': f'{loss_recon.item():.4f}',
                })
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            # ---------------------- 每 10 轮保存一次掩膜 ----------------------
            if (step + 1) % 100 == 0:
                with torch.no_grad():
                    save_path_mask1 = os.path.join(output_path, f'mask1/mask1_epoch_{step + 1:04d}.png')
                    save_path_mask2 = os.path.join(output_path, f'mask2/mask2_epoch_{step + 1:04d}.png')
                    save_path_fuse = os.path.join(output_path, f'fuse/fuse_epoch_{step + 1:04d}.png')
                    save_image(fuse.clamp(0, 1), save_path_fuse)
                    save_image(mask1.clamp(0, 1), save_path_mask1)
                    save_image(mask2.clamp(0, 1), save_path_mask2)

        with torch.no_grad():
            final_fuse = fuse.clamp(0, 1)  # [B,1,H,W] in [0,1]
            Y_8u = (final_fuse.squeeze().cpu().numpy() * 255.0).round().astype(np.uint8)  # [H,W]
            Cr_8u = cr.squeeze().cpu().numpy().astype(np.uint8)
            Cb_8u = cb.squeeze().cpu().numpy().astype(np.uint8)
            ycrcb8 = np.stack([Y_8u, Cr_8u, Cb_8u], axis=-1)
            fused_bgr = cv.cvtColor(ycrcb8, cv.COLOR_YCrCb2BGR)

            save_path_rgb = os.path.join(output_path, "fuse/fused_color.png")
            cv.imwrite(save_path_rgb, fused_bgr)
            print(f"彩色融合结果已保存到: {save_path_rgb}")

        # —————————————————————————— 图像增强部分 —————————————————————————— #

if __name__ == "__main__":
    args = parse()

    with open(args.config, mode='r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    set_random_seed(config['rand_seed'])

    main(config)
