import cv2 as cv
import torch
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim

# 读取图像并分离
def read_img(img_path):
    img = cv.imread(img_path)

    y, cr, cb = cv.split(cv.cvtColor(img, cv.COLOR_BGR2YCrCb))

    # 归一化到 [0,1] 并转 float32
    y = torch.from_numpy(y.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    cr = torch.from_numpy(cr.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    cb = torch.from_numpy(cb.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)

    return y, cr, cb
def pad_to_32(x):
    h, w = x.shape[-2:]
    new_h = int(np.ceil(h / 32) * 32)
    new_w = int(np.ceil(w / 32) * 32)
    pad_h = new_h - h
    pad_w = new_w - w
    x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    return x_padded, (h, w)

def crop_to_original(x, original_size):
    h, w = original_size
    return x[..., :h, :w]

def crop_to_even(tensor, base=32):
    """
    将输入 tensor 裁剪到最近的 base 的整数倍尺寸（默认 32）
    例如 513×517 → 512×512
    tensor: [B,C,H,W]
    """
    _, _, H, W = tensor.shape
    H_new = (H // base) * base
    W_new = (W // base) * base
    return tensor[:, :, :H_new, :W_new]


def get_score_map(y1, y2, mode='blur2th'):
    score_map_ = torch.sign(torch.abs(blur_2th(y1)) - torch.min(torch.abs(blur_2th(y1)), torch.abs(blur_2th(y2))))
    if mode == 'blur2th':
        score_map = score_map_
    elif mode == 'max_select':
        score_map = torch.max(torch.abs(blur_2th(y1)), torch.abs(blur_2th(y2)))
    elif mode == 'gradient':
        score_map = torch.sign(torch.abs(gradient(y1)) - torch.min(torch.abs(gradient(y1)), torch.abs(gradient(y2))))
    elif mode == 'guassian':
        score_map = guassian(score_map_)
    else:
        raise NotImplementedError
    return score_map.cuda()

def blur_2th(img):
    filtr = torch.tensor([[0.0947, 0.1183, 0.0947], [0.1183, 0.1478, 0.1183], [0.0947, 0.1183, 0.0947]], device=img.device)
    assert img.ndim == 4 and (img.shape[1] == 1 or img.shape[1] == 3)
    filtr = filtr.expand(img.shape[1], img.shape[1], 3, 3)
    blur = F.conv2d(img, filtr, bias=None, stride=1, padding=1)
    blur = F.conv2d(blur, filtr, bias=None, stride=1, padding=1)
    diff = torch.abs(img - blur)
    return diff

def guassian(input1):
    filtr = torch.tensor([[0.0947, 0.1183, 0.0947], [0.1183, 0.1478, 0.1183], [0.0947, 0.1183, 0.0947]]).type(torch.cuda.FloatTensor)
    filtr = filtr.expand(input1.shape[1], input1.shape[1], 3, 3)
    blur = F.conv2d(input1, filtr, bias=None, stride=1, padding=1)
    return blur

def gradient(input1):
    n, c, w, h = input1.shape
    filter1 = torch.reshape(torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).type(torch.cuda.FloatTensor), [1, 1, 3, 3])
    filter1 = filter1.repeat_interleave(c, dim=1)
    d = torch.nn.functional.conv2d(input1, filter1, bias=None, stride=1, padding=1)
    return d
# 生成噪声图像
def get_niose():
    return torch.rand((1,128,12,12))

def save_tensor_image(tensor, save_path):
    """
    保存网络输出图像张量为图片文件。
    tensor: torch.Tensor [B, C, H, W]
    save_path: str, 输出路径
    """
    # 取第一个batch
    img = tensor[0].detach().cpu()

    # 如果是单通道 -> [H,W]
    if img.shape[0] == 1:
        img = img.squeeze(0)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = (img * 255).numpy().astype(np.uint8)
        cv.imwrite(save_path, img)

    # 如果是3通道 -> [C,H,W] -> [H,W,C]
    elif img.shape[0] == 3:
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = (img * 255).permute(1, 2, 0).numpy().astype(np.uint8)
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        cv.imwrite(save_path, img)
    else:
        raise ValueError("Unsupported channel number in tensor.")

def entropy_score(img_gray):
    """信息熵"""
    # 转成 uint8 类型
    if img_gray.dtype != np.uint8:
        img_gray = (img_gray * 255).clip(0,255).astype(np.uint8)

    hist = cv.calcHist([img_gray], [0], None, [256], [0,256])
    hist_norm = hist / (hist.sum() + 1e-8)
    entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-8))
    return float(entropy)

def mean_gradient(img_gray):
    """平均梯度，反映整体锐度"""
    gx = cv.Sobel(img_gray, cv.CV_64F, 1, 0, ksize=3)
    gy = cv.Sobel(img_gray, cv.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(gx**2 + gy**2)
    return np.mean(grad)
def mean_val(img_gray):
    u = np.mean(img_gray)
    return u
def std_val(img_gray):
    u = np.std(img_gray)
    return u
def snr_val(img_gray):
    snr  = 20 * np.log10( mean_val(img_gray) / std_val(img_gray))
    return snr
def variance_score(img_gray):
    """灰度方差，对比度度量"""
    return np.var(img_gray)

def tenengrad_score(img_gray):
    """Tenengrad 焦点评价算子"""
    gx = cv.Sobel(img_gray, cv.CV_64F, 1, 0, ksize=3)
    gy = cv.Sobel(img_gray, cv.CV_64F, 0, 1, ksize=3)
    grad = gx**2 + gy**2
    return np.mean(grad)

def brenner_score(img_gray):
    """Brenner 焦点评价算子"""
    shifted = np.zeros_like(img_gray)
    shifted[2:, :] = img_gray[:-2, :]
    diff = (img_gray - shifted)**2
    return np.mean(diff)

def laplacian_energy(img_gray):
    """Laplacian 能量"""
    lap = cv.Laplacian(img_gray, cv.CV_64F)
    return np.mean(lap**2)

def gray_level_variance(img_gray, block=16):
    """局部灰度方差(GLV)，反映细节均匀性"""
    H, W = img_gray.shape
    glv_values = []
    for y in range(0, H-block, block):
        for x in range(0, W-block, block):
            patch = img_gray[y:y+block, x:x+block]
            glv_values.append(np.var(patch))
    return np.mean(glv_values)

def niqe_like(img_gray):
    """
    简化版自然图像质量统计(NIQE-like)：
    根据均值和标准差的偏差衡量自然性
    """
    mu = np.mean(img_gray)
    sigma = np.std(img_gray)
    # 正常自然图像亮度均值应在 [0.3,0.7] 范围，方差适中
    score = abs(mu - 0.5) + abs(sigma - 0.25)
    return score

def evaluate_image_quality(img_path):
    """综合评价函数"""
    img = cv.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) / 255.0

    results = {}
    #results["Entropy"] = entropy_score(img_gray)
    results["MeanGrad"] = mean_gradient(img_gray)
    results["Variance"] = variance_score(img_gray)
    #results["Tenengrad"] = tenengrad_score(img_gray)
    #results["Brenner"] = brenner_score(img_gray)
    #results["LapEnergy"] = laplacian_energy(img_gray)
    #results["GLV"] = gray_level_variance(img_gray)
    results["mean"] = mean_val(img_gray)
    results["std"] = std_val(img_gray)
    results["SNR"] = snr_val(img_gray)
    results["NIQE_like"] = niqe_like(img_gray)

    print(f"图像质量指标 ({img_path})")
    for k,v in results.items():
        print(f"{k:12s}: {v:.4f}")
    return results

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    return 20 * np.log10(1.0 / np.sqrt(mse + 1e-8))

def compare_images(gt_path, fused_path):
    gt = cv.imread(gt_path)
    fused = cv.imread(fused_path)
    gt = cv.resize(gt, (fused.shape[1], fused.shape[0]))
    gt = cv.cvtColor(gt, cv.COLOR_BGR2GRAY) / 255.0
    fused = cv.cvtColor(fused, cv.COLOR_BGR2GRAY) / 255.0
    psnr_val = psnr(gt, fused)
    ssim_val = ssim(gt, fused, data_range=1.0)
    print(f"PSNR: {psnr_val:.2f} dB  |  SSIM: {ssim_val:.4f}")

if __name__ == "__main__":
    # 示例
    #evaluate_image_quality("fused_color.png")
    compare_images("68.jpg", "fused_color.png")