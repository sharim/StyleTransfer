import torch
import torch.nn as nn
from tensorboardX import SummaryWriter


class conv(nn.Module):
    # 卷积模块，分上采样和下采样
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super().__init__()
        self.up = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        y = x
        if self.up is not None:
            y = nn.functional.interpolate(y, mode='nearest', scale_factor=self.up)
        y = self.conv(self.reflection_pad(y))
        return y


class residual(nn.Module):
    # 残差模块
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = conv(in_channels, in_channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(in_channels, affine=True)
        self.conv2 = conv(in_channels, in_channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(in_channels, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        resi = x
        y = self.relu(self.in1(self.conv1(x)))
        y = self.in2(self.conv2(y))
        y = y + resi
        return y


class transformNet(nn.Module):
    # 按Johnson转换结构构建模型
    def __init__(self):
        super().__init__()
        # 初始化下采样模块
        self.down1 = conv(3, 32, 9, 1)
        self.in1 = nn.InstanceNorm2d(32, affine=True)
        self.down2 = conv(32, 64, 3, 2)
        self.in2 = nn.InstanceNorm2d(64, affine=True)
        self.down3 = conv(64, 128, 3, 2)
        self.in3 = nn.InstanceNorm2d(128, affine=True)
        # 初始化残差模块
        self.res1 = residual(128)
        self.res2 = residual(128)
        self.res3 = residual(128)
        self.res4 = residual(128)
        self.res5 = residual(128)
        # 初始化上采样模块
        self.up1 = conv(128, 64, 3, 1, 2)
        self.in4 = nn.InstanceNorm2d(64, affine=True)
        self.up2 = conv(64, 32, 3, 1, 2)
        self.in5 = nn.InstanceNorm2d(32, affine=True)
        self.up3 = conv(32, 3, 9, 1)
        # 初始化ReLU模块
        self.relu = nn.ReLU()

    def forward(self, x):
        # 下采样
        y = self.relu(self.in1(self.down1(x)))
        y = self.relu(self.in2(self.down2(y)))
        y = self.relu(self.in3(self.down3(y)))
        # 残差
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        # 上采样
        y = self.relu(self.in4(self.up1(y)))
        y = self.relu(self.in5(self.up2(y)))
        y = self.up3(y)
        return y


if __name__ == "__main__":
    dummy_input_1 = torch.rand(13, 3, 28, 28)
    dummy_input_2 = torch.rand(13, 1, 28, 28)
    dummy_input_3 = torch.rand(13, 1, 28, 28)
    model = transformNet()
    with SummaryWriter(comment='transformNet') as w:
        w.add_graph(model, (dummy_input_1,))
