# -*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import time
import torch
import argparse
import numpy as np
from torch.autograd import Variable
import cv2
from data.config import cfg
from PIL import Image
from utils.augmentations import to_chw_bgr
from model import Network

parser = argparse.ArgumentParser(description='demo')
parser.add_argument('--save_dir',
                    type=str, default='E:/hj/SSD-exdark/en_test/',
                    help='Directory for detect result')
args = parser.parse_args()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)



torch.set_default_tensor_type('torch.cuda.FloatTensor')


def detect(net, img_path):
    img = Image.open(img_path)
    img = np.array(img)
    height, width, _ = img.shape
    # max_im_shrink = np.sqrt(
    #     600 * 400 / (img.shape[0] * img.shape[1]))
    # image = cv2.resize(img, None, None, fx=max_im_shrink,
    #                    fy=max_im_shrink, interpolation=cv2.INTER_LINEAR)
    image = img
    x = to_chw_bgr(image)
    x = x.astype('float32')
    x -= cfg.img_mean
    x = x[[2, 1, 0], :, :]

    x = Variable(torch.from_numpy(x).unsqueeze(0))
    x = x.cuda()
    net.cuda()

    enhance, _, _ = net(x, False)

    save_enhance_img_path = args.save_dir
    os.makedirs(save_enhance_img_path, exist_ok=True)
    print(save_enhance_img_path)
    name = img_path.split('/')[-1]
    save_enhance_file = save_enhance_img_path + name
    save_images(enhance, save_enhance_file)



def save_images(tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    # max_im_shrink = np.sqrt(
    #     600 * 400 / (image_numpy.shape[0] * image_numpy.shape[1]))
    # image_numpy = cv2.resize(image_numpy, None, None, fx=max_im_shrink,
    #                    fy=max_im_shrink, interpolation=cv2.INTER_LINEAR)
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    im.save(path, 'png')


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    net = Network('test', 2)
    net.cuda()
    # print(net)
    # net.detection_net.load_state_dict(torch.load(args.model))
    dir = 'E:/hj/SSD-exdark/epoch_199.pth'
    enhance_model = dir
    enhance_net = net.enhance_net
    enhance_net.load_state_dict(torch.load(enhance_model))
    enhance_net.cuda()

    enhance_net.eval()
    # path1 = 'E:/hj/deeplabv3/DeepLabV3Plus/datasets/data/cityscapes/val/night/'
    # path = os.listdir(path1)
    # with torch.no_grad():
    #     for filelist in path:
    #         file = path1 + filelist
    #         # print(file)
    #         detect(net, file)
    img_path = 'E:/hj/SSD-exdark/data/test--copy.txt'
    img_list = []
    with open(img_path) as f:
        f_lines = f.readlines()
        for i in range(0, len(f_lines)):
            f_lines[i] = f_lines[i].split(' ', 1)[0]
            img_list.append(f_lines[i])

    with torch.no_grad():
        for path in img_list:
            # print(path)
            detect(net, path)
