import torch
import torch.nn as nn
import functools
from loss import *


class Res_Enhancement(nn.Module):
    def __init__(self):
        super(Res_Enhancement, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))
        self.conv2_0 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16))

        self.conv2_1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16))

        self.conv2_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16))

        self.delta = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1, padding=0),
            nn.Sigmoid())

        self.en_loss = LossFunction()

    def forward(self, x):
        block1 = self.conv1(x)
        block2 = self.conv2_0(block1)
        block2 = block1 + block2
        block3 = self.conv2_1(block2)
        block3 = block2 + block3
        block4 = self.conv2_2(block3)
        block4 = block3 + block4
        delta = self.delta(block4)
        illu = delta + x

        illu = torch.clamp(illu, 0.001, 1.0)
        enhance = x / illu
        enhance = torch.clamp(enhance, 0.0, 1.0)
        return enhance, illu

    def loss(self, x):
        illu, enhance = self(x)
        loss = self.en_loss(x, illu)
        return loss
