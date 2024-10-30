import math
#详细的各类改进方法和流程操作，请关注B站博主：AI学术叫叫兽 
import numpy as np
import torch
import torch.nn as nn



class Concat_BiFPN(nn.Module):
    def __init__(self, dimension=1):
        super(Concat_BiFPN, self).__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
 
    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        x = [weight[0] * x[0], weight[1] * x[1]]
        return torch.cat(x, self.d)


class Concat_BiFPN(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, c1, c2):
        super(Concat_BiFPN, self).__init__()
        self.w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        # self.w3 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.conv = Conv(c1, c2, 1, 1, 0)
        self.act = nn.ReLU()

    def forward(self, x):  # mutil-layer 1-3 layers #ADD or Concat
        # print("bifpn:",x.shape)
        if len(x) == 2:
            w = self.w1
            weight = w / (torch.sum(w, dim=0) + self.epsilon)
            x = self.conv(self.act(weight[0] * x[0] + weight[1] * x[1]))
        elif len(x) == 3:
            w = self.w2
            weight = w / (torch.sum(w, dim=0) + self.epsilon)
            x = self.conv(self.act(weight[0] * x[0] + weight[1] * x[1] + weight[2] * x[2]))
        # elif len(x) == 4:
        #     w = self.w3
        #     weight = w / (torch.sum(w, dim=0) + self.epsilon)
        #     x = self.conv(self.act(weight[0] * x[0] + weight[1] * x[1] + weight[2] *x[2] + weight[3]*x[3] ))
        return x

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
            super(ConvBlock, self).__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
            return x

    class BiFPNBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(BiFPNBlock, self).__init__()
            self.conv6_up = ConvBlock(in_channels, out_channels, 1, 1, 0)
            self.conv5_up = ConvBlock(in_channels, out_channels, 1, 1, 0)
            self.conv4_up = ConvBlock(in_channels, out_channels, 1, 1, 0)
            self.conv3_up = ConvBlock(in_channels, out_channels, 1, 1, 0)

            self.conv4_down = ConvBlock(in_channels, out_channels, 1, 1, 0)
            self.conv5_down = ConvBlock(in_channels, out_channels, 1, 1, 0)
            self.conv6_down = ConvBlock(in_channels, out_channels, 1, 1, 0)
            self.conv7_down = ConvBlock(in_channels, out_channels, 1, 1, 0)

            self.p6_upsample = nn.Upsample(scale_factor=2, mode='nearest')
            self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
            self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')

            self.p4_downsample = nn.MaxPool2d(2)
            self.p5_downsample = nn.MaxPool2d(2)
            self.p6_downsample = nn.MaxPool2d(2)

        def forward(self, p3, p4, p5, p6):
            # Upward pathway
            p6_up = self.conv6_up(p6)
            p5_up = self.conv5_up(p5 + self.p6_upsample(p6_up))
            p4_up = self.conv4_up(p4 + self.p5_upsample(p5_up))
            p3_out = self.conv3_up(p3 + self.p4_upsample(p4_up))

            # Downward pathway
            p4_out = self.conv4_down(p4 + p4_up + self.p4_downsample(p3_out))
            p5_out = self.conv5_down(p5 + p5_up + self.p5_downsample(p4_out))
            p6_out = self.conv6_down(p6 + p6_up + self.p6_downsample(p5_out))

            return p3_out, p4_out, p5_out, p6_out

    class BiFPN(nn.Module):
        def __init__(self, in_channels_list, out_channels, num_blocks=1):
            super(BiFPN, self).__init__()
            self.num_blocks = num_blocks
            self.blocks = nn.ModuleList([BiFPNBlock(out_channels, out_channels) for _ in range(num_blocks)])

            # Conv layers to align the input feature maps to the same number of channels
            self.p3_conv = ConvBlock(in_channels_list[0], out_channels, 1, 1, 0)
            self.p4_conv = ConvBlock(in_channels_list[1], out_channels, 1, 1, 0)
            self.p5_conv = ConvBlock(in_channels_list[2], out_channels, 1, 1, 0)
            self.p6_conv = ConvBlock(in_channels_list[3], out_channels, 1, 1, 0)

        def forward(self, inputs):
            p3, p4, p5, p6 = inputs
            p3 = self.p3_conv(p3)
            p4 = self.p4_conv(p4)
            p5 = self.p5_conv(p5)
            p6 = self.p6_conv(p6)

            for i in range(self.num_blocks):
                p3, p4, p5, p6 = self.blocks[i](p3, p4, p5, p6)

            return [p3, p4, p5, p6]

    # Example usage
    # Assume input feature maps from backbone network have 64, 128, 256, and 512 channels respectively
    bifpn = BiFPN(in_channels_list=[64, 128, 256, 512], out_channels=128, num_blocks=3)

    # Dummy inputs with batch size of 1
    p3 = torch.randn(1, 64, 64, 64)
    p4 = torch.randn(1, 128, 32, 32)
    p5 = torch.randn(1, 256, 16, 16)
    p6 = torch.randn(1, 512, 8, 8)

    # Forward pass through BiFPN
    outputs = bifpn([p3, p4, p5, p6])
    for i, output in enumerate(outputs):
        print(f"Output {i + 1} shape: {output.shape}")
