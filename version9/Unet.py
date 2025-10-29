import torch.nn as nn
import torch
import torch.nn.functional as F
from common import *

# UNet卷积块
class UNetConv2d(nn.Module):
    def __init__(self, in_size, out_size,kernel_size=3,need_norm_layer=True, need_bias=True, pad='zero'):
        super(UNetConv2d, self).__init__()

        if need_norm_layer is not None: # 是否使用归一化
            self.conv1 = nn.Sequential(
                conv(in_size,out_size,kernel_size,bias=need_bias,pad=pad),
                nn.InstanceNorm2d(out_size),
                nn.LeakyReLU()
            )
            self.conv2 = nn.Sequential(
                conv(out_size, out_size, kernel_size, bias=need_bias, pad=pad),
                nn.InstanceNorm2d(out_size),
                nn.LeakyReLU()
            )
        else:
            self.conv1 = nn.Sequential(
                conv(in_size,out_size,kernel_size,bias=need_bias,pad=pad),
                nn.LeakyReLU()
            )
            self.conv2 = nn.Sequential(
                conv(out_size, out_size, kernel_size, bias=need_bias, pad=pad),
                nn.LeakyReLU()
            )
    def forward(self,input):
        output = self.conv1(input)
        output = self.conv2(output)

        return output


class unetdown(nn.Module):
    def __init__(self,in_c,out_c,kernel_szie=3,norm_layer=True,need_bias=True,pad='zero'):
        super(unetdown, self).__init__()
        self.conv = UNetConv2d(in_c,out_c,kernel_szie,norm_layer,need_bias,pad)
        self.down = nn.MaxPool2d(2,2)
    def forward(self,input):
        out = self.down(input)
        out = self.conv(out)
        return out

class unetup(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 kernel_size=3,
                 upsample_mode='nearest',
                 need_norm_layer=True,
                 need_bias=True,
                 pad='zero'):
        super().__init__()
        assert upsample_mode in ['nearest', 'bilinear']
        self.upsample_mode = upsample_mode
        self.block = UNetConv2d(in_ch, out_ch,kernel_size, need_norm_layer, need_bias, pad)

    def forward(self, x, skip=None):
        # 上采样
        x = F.interpolate(x, scale_factor=2, mode=self.upsample_mode,
                          align_corners=(self.upsample_mode == 'bilinear'))
        if skip is not None:
            x = torch.cat([skip, x], dim=1)
        x = self.block(x)
        return x

class UNet(nn.Module):
    def __init__(self,num_input_channels=3,
                 num_output_channels=1,
                 num_channels_down=[16, 32, 64, 128, 128],
                 num_channels_up=[16, 32, 64, 128, 128],
                 num_channels_skip=[4, 4, 4, 4, 4],
                 filter_size_down=3,
                 filter_size_up=3,
                 filter_skip_size=1,
                 pad='zero',
                 upsample_mode='nearest',
                 need_norm_layers = True,
                 need_bias = True,
                 need_sigmoid=True
                 ):
        super(UNet, self).__init__()

        # --- 检查 ---
        assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip), \
            "down/up/skip channel lists must have same length"
        self.depth = len(num_channels_down)
        self.num_input_channels = num_input_channels # 输入通道数
        self.num_output_channels = num_output_channels # 输出通道数

        # 编码器第一层
        self.start_block = UNetConv2d(num_input_channels,num_channels_down[0],filter_size_down,need_norm_layers,need_bias,pad)
        # 编码器列表
        self.Unetdown = nn.ModuleList()
        for i in range(len(num_channels_down) - 1):
            self.Unetdown.append(
                unetdown(num_channels_down[i],num_channels_down[i+1],filter_size_down,need_norm_layers,need_bias,pad)
            )

        # 跳连
        self.skip_convs = nn.ModuleList()
        for i in range(self.depth):
            ch_in = num_channels_down[i] if i > 0 else num_channels_down[0]
            ch_skip = num_channels_skip[i]
            if ch_skip > 0:
                self.skip_convs.append(
                    nn.Sequential(
                        conv(ch_in, ch_skip, kernel_size=filter_skip_size,
                             bias=need_bias, pad=pad),
                        nn.LeakyReLU(inplace=True)
                    )
                )
            else:
                self.skip_convs.append(nn.Identity())

        # 解码端定义（只建 depth-1 个 up）
        self.Unetup = nn.ModuleList()
        for i in range(self.depth - 2, -1, -1):  # i: 3,2,1,0  对应 skip[3], skip[2], skip[1], skip[0]
            if i == self.depth - 2:
                # 第一个 up：输入 = 最深层主干 + 对应的 skip（i 层）
                in_ch = num_channels_down[-1] + num_channels_skip[i]  # 128 + 4 = 132
            else:
                # 其余 up：输入 = 上一 up 输出 + 对应的 skip（i 层）
                in_ch = num_channels_up[i + 1] + num_channels_skip[i]
            out_ch = num_channels_up[i]  # 依次 128, 64, 32, 16
            self.Unetup.append(
                unetup(in_ch, out_ch, filter_size_up, upsample_mode, need_norm_layers, need_bias, pad)
            )
            # --- 输出头 ---
        self.out_conv = nn.Conv2d(num_channels_up[0], num_output_channels, kernel_size=1, bias=True)
        self.out_act = nn.Sigmoid() if need_sigmoid else nn.Identity()

    def forward(self, inputs):
        skips = []
        e = self.start_block(inputs)
        skips.append(self.skip_convs[0](e))

        # 编码阶段
        for i, block in enumerate(self.Unetdown, start=1):
            e = block(e)
            skips.append(self.skip_convs[i](e))

        # 用编码器最后一层输出做解码起点
        y = e

        # 解码：与上面构造一致的“层号 i”顺序（3,2,1,0）
        for j, up in enumerate(self.Unetup):
            skip_i = self.depth - 2 - j  # j=0→skip[3], j=1→skip[2], j=2→skip[1], j=3→skip[0]
            skip = skips[skip_i]
            #print(f"Up[{j}]  y={y.shape}, skip={skip.shape}")
            y = up(y, skip)

        y = self.out_conv(y)
        y = self.out_act(y)
        return y


# -------------------------

if __name__ == "__main__":
    x = torch.randn(1, 3, 256, 256)
    net = UNet(num_input_channels=3,
               num_output_channels=3,
               num_channels_down=[16, 32, 64, 128],
               num_channels_up=[16, 32, 64, 128],
               num_channels_skip=[4, 4, 4, 4],
               upsample_mode='bilinear',
               need_sigmoid=True)
    y = net(x)
    print("Output:", y.shape)