import torch.nn as nn
from torchvision import models
from collections import namedtuple


class vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        # 初始化参数
        super().__init__()
        pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        relu1_2 = h
        h = self.slice2(h)
        relu2_2 = h
        h = self.slice3(h)
        relu3_3 = h
        h = self.slice4(h)
        relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        return vgg_outputs(relu1_2, relu2_2, relu3_3, relu4_3)
