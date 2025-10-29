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
from torchvision.utils import save_image
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
    lr_fuse = config["lr_fuse"]
    lr_enhance = config['lr_enhance']
    # 噪声
    reg_noise_std = config["reg_noise_std"]
    # 融合循环次数
    num_epochs_fuse = config['num_epochs_fuse']
    # 增强循环次数
    num_epochs_enhance = config['num_epochs_enhance']
    # 阈值
    thresh = config['thresh']
    alpha = config['alpha']
    # 保存周期
    save_interval = config['save_interval']
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

        # ———————————————————————————————————————————————————— 图像融合部分 ———————————————————————————————————————————————————— #
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

        optimizer = torch.optim.Adam(parameters, lr=lr_fuse)
        scheduler = MultiStepLR(optimizer, milestones=[200, 400, 800], gamma=0.5)

        net_input_savedm = net_inputm.detach().clone()
        noisem = net_inputm.detach().clone()

        score_map = []
        # --- 生成初始焦点图 (引导项) ---
        mask1_init = get_score_map(y1, y2, mode='blur2th')
        mask2_init = 1 - mask1_init
        score_map.append(mask1_init)
        score_map.append(mask2_init)

        pbar = tqdm(range(num_epochs_fuse), desc="DIP-MFF processing", ncols=200)
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
            if (step + 1) % save_interval == 0:
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

        # ———————————————————————————————————————————————————— 图像增强部分 ———————————————————————————————————————————————————— #
        # 冻结掩膜与融合
        for p in netm.parameters(): p.requires_grad = False
        for p in netx.parameters(): p.requires_grad = False
        # 原始融合输出
        fused_rgb = cv.cvtColor(fused_bgr, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
        original_fuse = torch.from_numpy(fused_rgb).permute(2, 0, 1).unsqueeze(0).to(device)  # [1,3,H,W]
        # 光照图输出网络
        neti = UNet(num_input_channels=3,
                    num_output_channels=1,
                    num_channels_down=[16, 32, 64, 128],
                    num_channels_up=[16, 32, 64, 128],
                    num_channels_skip=[4, 4, 4, 4],
                    upsample_mode='bilinear',
                    need_sigmoid=True).to(device)
        # 反射图输出网络
        netr = UNet(num_input_channels=3,
                    num_output_channels=3,
                    num_channels_down=[16, 32, 64, 128,128],
                    num_channels_up=[16, 32, 64, 128,128],
                    num_channels_skip=[4, 4, 4, 4,4],
                    upsample_mode='bilinear',
                    need_sigmoid=True).to(device)
        # 初始光照图
        I0 = get_max_rgb(original_fuse)
        # TV损失
        tv_loss = TVloss()
        ############################################### optimizer ######################################################
        parameters = []
        parameters.extend([{'params': neti.parameters()},
                           {'params': netr.parameters()},
                           ])

        optimizer = torch.optim.Adam(parameters, lr=lr_enhance)
        scheduler = MultiStepLR(optimizer, milestones=[200, 400, 800], gamma=0.5)
        pbar = tqdm(range(num_epochs_enhance), desc="enhance processing", ncols=200)
        for step in pbar:
            optimizer.zero_grad()

            # 光照图输出
            out_i = neti(original_fuse)
            # 反射图输出
            out_r = netr(original_fuse)
            # 重建图像
            enhanced_recon = out_i * out_r
            # 损失函数--------------------------------------------------------------
            loss_enhance_recon = 0 # retinex基本约束
            loss_enhance_illu_consis = 0 # 光照一致性损失
            loss_enhance_illumination_smoothness = 0 # 光照平滑损失
            loss_enhance_reflectance = 0 # 反射图平滑损失

            # 重建损失
            loss_enhance_recon = F.l1_loss(enhanced_recon , original_fuse)
            # 光照一致性损失
            loss_enhance_illu_consis = F.l1_loss(out_i,I0) # 保证光照图趋近原始光照图
            loss_enhance_illumination_smoothness = tv_loss(out_i,out_r) # 保证光照图在反射图细节部分保留边缘，在反射图平滑区域强制平滑
            loss_enhance_reflectance = tv_loss(out_r)
            # 总损失
            total_loss = (
                    5 * loss_enhance_recon
                    + loss_enhance_illu_consis
                    + loss_enhance_reflectance
                    + loss_enhance_illumination_smoothness)
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            pbar.set_postfix({
                "loss": f"{total_loss.item():.4f}",
                "recon": f"{loss_enhance_recon.item():.4f}",
                "illum_consis": f"{loss_enhance_illu_consis.item():.4f}",
                "reflect": f"{loss_enhance_reflectance.item():.4f}",
                "illum_smooth": f"{loss_enhance_illumination_smoothness.item():.4f}",
            })
            # 每隔 N 个周期输出图像
            if (step + 1) % save_interval == 0 or step == num_epochs_enhance - 1:
                with torch.no_grad():
                    # 反射图
                    save_image(out_r.clamp(0, 1), f"{output_path}/enhance/reflect/reflect_{step + 1}.png")
                    # 光照图
                    save_image(out_i.clamp(0, 1), f"{output_path}/enhance/illum/illum_{step + 1}.png")
                    # 重建图
                    save_image(enhanced_recon.clamp(0, 1), f"{output_path}/enhance/recon/recon_{step + 1}.png")
        # 最终增强输出（伽马校正）
        with torch.no_grad():
            # 转为 numpy
            illum_np = out_i.detach().cpu().clamp(0, 1)[0].permute(1, 2, 0).numpy()
            refl_np = out_r.detach().cpu().clamp(0, 1)[0].permute(1, 2, 0).numpy()

            # 光照取最大通道
            illum_max = np.max(illum_np, axis=2)
            illum_max = np.clip(illum_max, 1e-6, 1.0)

            # γ校正
            illum_gamma = np.power(illum_max, 0.5)

            # 原图像除以光照图
            img_np = original_fuse.detach().cpu().clamp(0, 1)[0].permute(1, 2, 0).numpy()
            enhanced = img_np / illum_gamma[..., None]
            enhanced = np.clip(enhanced, 0, 1)

            # 保存结果
            enhanced_uint8 = (enhanced * 255).astype(np.uint8)
            os.makedirs(output_path, exist_ok=True)
            cv.imwrite(f"{output_path}/enhance/enhanced_final.png", cv.cvtColor(enhanced_uint8, cv.COLOR_RGB2BGR))

        print(f"✅ 训练完成！最终结果保存在：{output_path}/enhance/enhanced_final.png")


if __name__ == "__main__":
    args = parse()

    with open(args.config, mode='r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    set_random_seed(config['rand_seed'])

    main(config)
