import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F


# 卷积块
def conv(in_c, out_c, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride'):
    downsampler = None
    if stride != 1 and downsample_mode != 'stride':
        if downsample_mode == 'avg':
            downsampler = nn.AvgPool2d(stride, stride)
        elif downsample_mode == 'max':
            downsampler = nn.MaxPool2d(stride, stride)
        else:
            assert False

        stride = 1

    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0

    convolver = nn.Conv2d(in_c, out_c, kernel_size, stride, padding=to_pad, bias=bias)

    layers = filter(lambda x: x is not None, [padder, convolver, downsampler])  # 如果不改pad和stride就只会用conv2d
    return nn.Sequential(*layers)


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

def get_max_rgb(img_tensor):
    img = torch.max(img_tensor,dim=1,keepdim=True)[0]
    return img

if __name__ == "__main__":
    x = torch.randn(1, 3, 256, 256)
    y = get_max_rgb(x)
    print(y.shape)