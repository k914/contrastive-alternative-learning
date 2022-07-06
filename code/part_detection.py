# -*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

from layers import *
from data.config import cfg


# from genotypes import *
# from operations import *


class FEM(nn.Module):
    """docstring for FEM"""

    def __init__(self, in_planes):
        super(FEM, self).__init__()
        inter_planes = in_planes // 3
        inter_planes1 = in_planes - 2 * inter_planes
        self.branch1 = nn.Conv2d(
            in_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3)

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_planes, inter_planes, kernel_size=3,
                      stride=1, padding=3, dilation=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_planes, inter_planes, kernel_size=3,
                      stride=1, padding=3, dilation=3)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_planes, inter_planes1, kernel_size=3,
                      stride=1, padding=3, dilation=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_planes1, inter_planes1, kernel_size=3,
                      stride=1, padding=3, dilation=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_planes1, inter_planes1, kernel_size=3,
                      stride=1, padding=3, dilation=3)
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x1, x2, x3), dim=1)
        out = F.relu(out, inplace=True)
        return out


class DetectionNetwork(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, num_classes):
        super(DetectionNetwork, self).__init__()

        self.base = vgg(vgg_cfg, 3)
        # self.genotype = genotype
        self.add_extras = add_extras(extras_cfg, 1024)
        self.head1 = multibox(self.base, self.add_extras, num_classes)
        # self.head2 = multibox(self.base, self.add_extras, num_classes)
        self.fem = fem_module(fem_cfg)

        self.phase = phase
        self.num_classes = num_classes
        self.vgg = nn.ModuleList(self.base)
        if self.phase != 'test':
            self._init_vgg()

        self.L2Normof1 = L2Norm(256, 10)
        self.L2Normof2 = L2Norm(512, 8)
        self.L2Normof3 = L2Norm(512, 5)

        self.extras = nn.ModuleList(self.add_extras)
        self.fpn = FeaturePyramidNetwork()
        # self.fpn_topdown = nn.ModuleList(self.fem[0])
        # self.fpn_latlayer = nn.ModuleList(self.fem[1])
        #
        # self.fpn_fem = nn.ModuleList(self.fem[2])

        # self.L2Normef1 = L2Norm(256, 10)
        # self.L2Normef2 = L2Norm(512, 8)
        # self.L2Normef3 = L2Norm(512, 5)

        self.loc_pal1 = nn.ModuleList(self.head1[0])
        self.conf_pal1 = nn.ModuleList(self.head1[1])

        # self.loc_pal2 = nn.ModuleList(self.head2[0])
        # self.conf_pal2 = nn.ModuleList(self.head2[1])

        self.criterion = MultiBoxLoss(cfg, True)

        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(cfg)

    def _upsample_prod(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') * y

    def forward(self, x, contras=False):
        # print("---------------",x.shape)
        size = x.size()[2:]
        pal1_sources = list()
        pal2_sources = list()
        loc_pal1 = list()
        conf_pal1 = list()
        loc_pal2 = list()
        conf_pal2 = list()
        feature = []

        # apply vgg up to conv4_3 relu
        for k in range(16):
            x = self.vgg[k](x)
        of1 = x
        s = self.L2Normof1(of1)
        pal1_sources.append(s)
        feature.append(of1)

        # apply vgg up to fc7
        for k in range(16, 23):
            x = self.vgg[k](x)
        of2 = x
        s = self.L2Normof2(of2)
        pal1_sources.append(s)
        feature.append(of2)

        for k in range(23, 30):
            x = self.vgg[k](x)
        of3 = x
        s = self.L2Normof3(of3)
        pal1_sources.append(s)
        feature.append(of3)

        for k in range(30, len(self.vgg)):
            x = self.vgg[k](x)
        of4 = x
        pal1_sources.append(of4)
        feature.append(of4)

        for k in range(2):
            x = F.relu(self.extras[k](x), inplace=True)

        of5 = x
        pal1_sources.append(of5)

        for k in range(2, 4):
            x = F.relu(self.extras[k](x), inplace=True)

        of6 = x
        pal1_sources.append(of6)

        out_fea = self.fpn(pal1_sources)

        for (x, l, c) in zip(out_fea, self.loc_pal1, self.conf_pal1):
            loc_pal1.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf_pal1.append(c(x).permute(0, 2, 3, 1).contiguous())

        # for (x, l, c) in zip(pal2_sources, self.loc_pal2, self.conf_pal2):
        #     loc_pal2.append(l(x).permute(0, 2, 3, 1).contiguous())
        #     conf_pal2.append(c(x).permute(0, 2, 3, 1).contiguous())

        features_maps = []
        for i in range(len(loc_pal1)):
            feat = []
            feat += [loc_pal1[i].size(1), loc_pal1[i].size(2)]
            features_maps += [feat]

        # print(features_maps)
        # for i in loc_pal1:
        #     print(i.shape)
        # for i in conf_pal1:
        #     print(i.shape)

        loc_pal1 = torch.cat([o.view(o.size(0), -1)
                              for o in loc_pal1], 1)
        conf_pal1 = torch.cat([o.view(o.size(0), -1)
                               for o in conf_pal1], 1)

        # loc_pal2 = torch.cat([o.view(o.size(0), -1)
        #                       for o in loc_pal2], 1)
        # conf_pal2 = torch.cat([o.view(o.size(0), -1)
        #                        for o in conf_pal2], 1)

        priorbox = PriorBox(size, features_maps, cfg, pal=1)
        with torch.no_grad():
            self.priors_pal1 = Variable(priorbox.forward())

        # priorbox = PriorBox(size, features_maps, cfg, pal=2)
        # with torch.no_grad():
        #     self.priors_pal2 = Variable(priorbox.forward())

        if contras:
            return feature
        else:
            if self.phase == 'test':

                output = self.detect.forward(
                    loc_pal1.view(loc_pal1.size(0), -1, 4),
                    self.softmax(conf_pal1.view(conf_pal1.size(0), -1,
                                                self.num_classes)),                # conf preds
                    self.priors_pal1.type(type(x.data))
                )

            else:
                output = (
                    loc_pal1.view(loc_pal1.size(0), -1, 4),
                    conf_pal1.view(conf_pal1.size(0), -1, self.num_classes),
                    self.priors_pal1,)
                    # loc_pal2.view(loc_pal2.size(0), -1, 4),
                    # conf_pal2.view(conf_pal2.size(0), -1, self.num_classes),
                    # self.priors_pal2

            return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            mdata = torch.load(base_file,
                               map_location=lambda storage, loc: storage)
            weights = mdata['weight']
            epoch = mdata['epoch']
            self.load_state_dict(weights)
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')
        return epoch

    def xavier(self, param):
        init.xavier_uniform(param)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            self.xavier(m.weight.data)
            m.bias.data.zero_()

        if isinstance(m, nn.ConvTranspose2d):
            self.xavier(m.weight.data)
            if 'bias' in m.state_dict().keys():
                m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data[...] = 1
            m.bias.data.zero_()

    def _init_vgg(self):
        model_dict = torch.load('E:/hj/coSSD/weights/vgg16_reducedfc.pth')
        self.vgg.load_state_dict(model_dict)

class FeaturePyramidNetwork(nn.Module):
    # def __init__(self, feature_size=512):
    def __init__(self, feature_size=512):
        super(FeaturePyramidNetwork, self).__init__()
        self.fem_cfg = [0, 256, 512, 512, 1024, 512, 256]
        self.feature_size = feature_size
        # upsample C5 to get P5 from the FPN paper
        # 256 256 512 512
        # self.P6_1 = nn.Conv2d(self.fem_cfg[6], self.feature_size, kernel_size=1, stride=1, padding=0)
        self.P6_1 = nn.Conv2d(self.fem_cfg[6], self.feature_size, kernel_size=3, stride=1, padding=1)
        self.P6_2 = nn.Conv2d(self.feature_size, 512, kernel_size=3, stride=1, padding=1)

        # 512 256 512 512
        self.P5_1 = nn.Conv2d(self.fem_cfg[5], self.feature_size, kernel_size=3, stride=1, padding=1)
        self.P5_2 = nn.Conv2d(self.feature_size, 512, kernel_size=3, stride=1, padding=1)
        # self.P5_m = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)

        self.P4_1 = nn.Conv2d(self.fem_cfg[4], self.feature_size, kernel_size=3, stride=1, padding=1)
        self.P4_2 = nn.Conv2d(self.feature_size, 512, kernel_size=3, stride=1, padding=1)

        self.P3_1 = nn.Conv2d(self.fem_cfg[3], self.feature_size, kernel_size=3, stride=1, padding=1)
        self.P3_2 = nn.Conv2d(self.feature_size, 512, kernel_size=3, stride=1, padding=1)

        self.P2_1 = nn.Conv2d(self.fem_cfg[2], self.feature_size, kernel_size=3, stride=1, padding=1)
        self.P2_2 = nn.Conv2d(self.feature_size, 512, kernel_size=3, stride=1, padding=1)

        self.P1_1 = nn.Conv2d(self.fem_cfg[1], self.feature_size, kernel_size=3, stride=1, padding=1)
        self.P1_2 = nn.Conv2d(self.feature_size, 512, kernel_size=3, stride=1, padding=1)

        self.Upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.act = nn.ReLU()

    def _upsample(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear')

    def forward(self, pal1_sources):
        of1, of2, of3, of4, of5, of6 = pal1_sources

        of6 = self.act(self.P6_1(of6))
        mf6_up = self._upsample(of6, of5)
        out_f6 = self.act(self.P6_2(of6))

        of5 = self.act(self.P5_1(of5))
        mf5 = of5 + mf6_up
        mf5_up = self._upsample(mf5, of4)
        out_f5 = self.act(self.P5_2(mf5))

        of4 = self.act(self.P4_1(of4))
        mf4 = of4 + mf5_up
        mf4_up = self._upsample(mf4, of3)
        out_f4 = self.act(self.P4_2(mf4))

        of3 = self.act(self.P3_1(of3))
        mf3 = of3 + mf4_up
        mf3_up = self._upsample(mf3, of2)
        out_f3 = self.act(self.P3_2(mf3))

        of2 = self.act(self.P2_1(of2))
        mf2 = of2 + mf3_up
        mf2_up = self._upsample(mf2, of1)
        out_f2 = self.act(self.P2_2(mf2))

        of1 = self.act(self.P1_1(of1))
        mf1 = of1 + mf2_up
        out_f1 = self.act(self.P1_2(mf1))

        return [out_f1, out_f2, out_f3, out_f4, out_f5, out_f6]

vgg_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
           512, 512, 512, 'M']

extras_cfg = [256, 'S', 512, 128, 'S', 256]

fem_cfg = [256, 512, 512, 1024, 512, 256]


def fem_module(cfg):
    topdown_layers = []
    lat_layers = []
    fem_layers = []

    topdown_layers += [nn.Conv2d(cfg[-1], cfg[-1],
                                 kernel_size=1, stride=1, padding=0)]
    for k, v in enumerate(cfg):
        fem_layers += [FEM(v)]
        cur_channel = cfg[len(cfg) - 1 - k]
        if len(cfg) - 1 - k > 0:
            last_channel = cfg[len(cfg) - 2 - k]
            topdown_layers += [nn.Conv2d(cur_channel, last_channel,
                                         kernel_size=1, stride=1, padding=0)]
            lat_layers += [nn.Conv2d(last_channel, last_channel,
                                     kernel_size=1, stride=1, padding=0)]
    return (topdown_layers, lat_layers, fem_layers)


def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=3, dilation=3)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                                     kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


# def multibox(vgg, extra_layers, num_classes):
#     loc_layers = []
#     conf_layers = []
#     vgg_source = [14, 21, 28, -2]
#
#     for k, v in enumerate(vgg_source):
#         loc_layers += [nn.Conv2d(vgg[v].out_channels,
#                                  4, kernel_size=3, padding=1)]
#         conf_layers += [nn.Conv2d(vgg[v].out_channels,
#                                   num_classes, kernel_size=3, padding=1)]
#     for k, v in enumerate(extra_layers[1::2], 2):
#         loc_layers += [nn.Conv2d(v.out_channels,
#                                  4, kernel_size=3, padding=1)]
#         conf_layers += [nn.Conv2d(v.out_channels,
#                                   num_classes, kernel_size=3, padding=1)]
#     return (loc_layers, conf_layers)

def multibox(vgg, extra_layers, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [14, 21, 28, -2]

    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(512, 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(512, num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(512, 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(512, num_classes, kernel_size=3, padding=1)]
    return (loc_layers, conf_layers)