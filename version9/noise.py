import os
import PIL
import sys
import cv2
import torch
import random
import numpy as np
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms

def fill_noise(x,type):
    torch.manual_seed(0)
    if type == 'u':
        x.uniform_()
    elif type == 'n':
        x.normal_()
    else:
        assert False

def get_noise(spatial_size=None, input_channel=None, noise_type='u', var=1./10,like_image=None):
    if like_image is not None:
        # 自动匹配输入图像的通道和分辨率
        _, input_channel, h, w = like_image.shape
        spatial_size = (h, w)

    elif isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)

    shape = [1,input_channel,spatial_size[0],spatial_size[1]]
    net_input = torch.zeros(shape)

    fill_noise(net_input,noise_type)
    net_input *= var

    return net_input

if __name__ == "__main__":
    noise = get_noise(256,3)
    print(noise.shape)