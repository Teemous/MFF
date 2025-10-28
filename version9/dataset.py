import os
import glob
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
import cv2 as cv

class Microscopy_dataset(Dataset):
    def __init__(self,root_dir):
        self.images1 = glob.glob(os.path.join(root_dir, '*-1.jpg'))
        self.images1.sort()
        self.images2 = glob.glob(os.path.join(root_dir, '*-2.jpg'))
        self.images2.sort()
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.images1)

    def __getitem__(self, idx):
        img1 = cv.imread(self.images1[idx])
        img2 = cv.imread(self.images2[idx])
        y1, cr, cb = cv.split(cv.cvtColor(img1, cv.COLOR_BGR2YCrCb))
        y2, _, _ = cv.split(cv.cvtColor(img2, cv.COLOR_BGR2YCrCb))

        y1 = self.transform(y1)
        y2 = self.transform(y2)

        return y1, y2, cr, cb, os.path.basename(self.images1[idx]).split('.')[0][:-2]

if __name__ == "__main__":
    root_dir = "data/test"
    dataset = Microscopy_dataset(root_dir)

    # 2️⃣ 打印基本信息
    print(f"样本数量: {len(dataset)}")
    print(f"第一对图像路径: {dataset.images1[0]}, {dataset.images2[0]}")

    # 3️⃣ 取一条样本测试
    y1, y2, cr, cb, name = dataset[0]
    print(f"样本名称: {name}")
    print(f"y1 形状: {y1.shape}, y2 形状: {y2.shape}")
    print(f"cr 形状: {cr.shape}, cb 形状: {cb.shape}")