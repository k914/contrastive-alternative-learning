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
import torch.backends.cudnn as cudnn
import cv2
from data.config import cfg
from PIL import Image
from utils.augmentations import to_chw_bgr
from model import Network

parser = argparse.ArgumentParser(description='demo')
parser.add_argument('--save_dir',
                    type=str, default='tmp/0401',
                    help='Directory for detect result')
parser.add_argument('--model',
                    type=str,
                    default='E:/hj/coSSD/models/EXP/models-detection-loss-EXP-20220323-110918/model_epochs/epoch_385.pth',
                    help='trained model')
parser.add_argument('--enhance_model',
                    type=str,
                    default='E:/hj/coSSD/models/EXP/models-detection-loss-EXP-20220323-110918/image_epochs/epoch_400.pth',
                    help='trained model')
parser.add_argument('--thresh',
                    default=0.4, type=float,
                    help='Final confidence threshold')
parser.add_argument('--test_data',
                    default='E:/hj/coSSD/models/data/face_val.txt', type=str,
                    help='Test data')
args = parser.parse_args()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

torch.set_default_tensor_type('torch.FloatTensor')

def detect(net, img_path, thresh):
    img = Image.open(img_path)
    if img.mode == 'L':
        img = img.convert('RGB')

    img = np.array(img)

    height, width, _ = img.shape
    max_im_shrink = np.sqrt(
        1500 * 1000 / (img.shape[0] * img.shape[1]))
    image = cv2.resize(img, None, None, fx=max_im_shrink,
                       fy=max_im_shrink, interpolation=cv2.INTER_LINEAR)

    x = to_chw_bgr(image)
    x = x.astype('float32')
    x -= cfg.img_mean
    x = x[[2, 1, 0], :, :]

    x = Variable(torch.from_numpy(x).unsqueeze(0))
    t1 = time.time()
    enhance, illu, y = net(x, False)

    save_enhance_img_path = args.save_dir + '/img_enhance/'
    os.makedirs(save_enhance_img_path, exist_ok=True)
    name = path.split('/')[-1]
    save_enhance_file = save_enhance_img_path + name
    save_images(enhance, save_enhance_file)


    detections = y.data
    scale = torch.Tensor([img.shape[1], img.shape[0],
                          img.shape[1], img.shape[0]])

    # img = cv2.imread(save_enhance_file, cv2.IMREAD_COLOR)
    save_txt_path = args.save_dir + '/txt/'
    os.makedirs(save_txt_path, exist_ok=True)
    txt_name = save_txt_path + img_path.split('/')[-1].split('.')[0] + '.txt'

    f = open(txt_name, 'w')
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= thresh:
            score = detections[0, i, j, 0]
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy().astype(int)

            # if pt[0]<0 or pt[1]<0 or pt[2]<0 or pt[3]<0:
            #     # continue
            #     break

            left_up, right_bottom = (pt[0], pt[1]), (pt[2], pt[3])
            j += 1
            cv2.rectangle(img, left_up, right_bottom, (0, 0, 255), 2)
            # conf = "{:.6}".format(score)
            # text_size, baseline = cv2.getTextSize(
            #     conf, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
            #
            # # print(text_size)
            # p1 = (left_up[0], left_up[1] - text_size[1])
            # cv2.rectangle(img, (p1[0] - 2 // 2, p1[1] - 2 - baseline),
            #               (p1[0] + text_size[0], p1[1] + text_size[1]), [255, 0, 0], -1)
            #
            # cv2.putText(img, conf, (p1[0], p1[
            #     1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, 8)

            f.write('face {:.6f} {} {} {} {}\n'.
                    format(score, int(np.floor(pt[0])), int(np.floor(pt[1])), int(np.floor(pt[2])),
                           int(np.floor(pt[3]))))
            if j==750:
                break

    t2 = time.time()
    print('detect:{} timer:{}'.format(img_path, t2 - t1))

    save_img_path = args.save_dir + '/img/'
    os.makedirs(save_img_path, exist_ok=True)
    cv2.imwrite(os.path.join(save_img_path, os.path.basename(img_path)), img)




def write_txt(net, img_path, thresh):
    img = Image.open(img_path)
    if img.mode == 'L':
        img = img.convert('RGB')

    img = np.array(img)
    height, width, _ = img.shape
    max_im_shrink = np.sqrt(
        1500 * 1000 / (img.shape[0] * img.shape[1]))
    image = cv2.resize(img, None, None, fx=max_im_shrink,
                       fy=max_im_shrink, interpolation=cv2.INTER_LINEAR)

    x = to_chw_bgr(image)
    x = x.astype('float32')
    x -= cfg.img_mean
    x = x[[2, 1, 0], :, :]

    x = Variable(torch.from_numpy(x).unsqueeze(0))
    x = x.cuda()
    t1 = time.time()
    targets = 0
    enhance, illu, y = net(x, targets, False)
    detections = y.data
    scale = torch.Tensor([img.shape[1], img.shape[0],
                          img.shape[1], img.shape[0]])

    save_txt_path = args.save_dir + '/txt/'
    os.makedirs(save_txt_path, exist_ok=True)
    txt_name = save_txt_path + img_path.split('/')[-1].split('.')[0] + '.txt'

    f = open(txt_name, 'w')
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= thresh:
            score = detections[0, i, j, 0]
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy().astype(int)
            j += 1
            # f.write('face {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.
            #         format(int(np.floor(pt[0])), int(np.floor(pt[1])), int(np.floor(pt[2])), int(np.floor(pt[3])),
            #                score))
            f.write('face {:.6f} {} {} {} {}\n'.
                    format(score, int(np.floor(pt[0])), int(np.floor(pt[1])), int(np.floor(pt[2])),
                           int(np.floor(pt[3]))))
            if j == 750:
                break

    t2 = time.time()
    print('detect:{} timer:{}'.format(img_path, t2 - t1))


def save_images(tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    max_im_shrink = np.sqrt(
        1080 * 720 / (image_numpy.shape[0] * image_numpy.shape[1]))
    image_numpy = cv2.resize(image_numpy, None, None, fx=max_im_shrink,
                       fy=max_im_shrink, interpolation=cv2.INTER_LINEAR)
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    im.save(path, 'png')

# def save_images(tensor, path):
#     image_numpy = tensor[0].cpu().float().numpy()
#     image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
#     im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
#     im.save(path, 'png')


if __name__ == '__main__':
    # net = build_net('test', cfg.NUM_CLASSES, args.network)
    net = Network('test', 2)
    print(net)
    net.detection_net.load_state_dict(torch.load(args.model))
    net.enhance_net.load_state_dict(torch.load(args.enhance_model))
    net.eval()

    # with open(args.test_data) as f:
    #     lines = f.readlines()
    #
    # img_list = []
    # for line in lines:
    #     line = line.strip().split()
    #     img_path = line[0]
    #     img_list.append(img_path)

    img_path = r'E:/hj/coSSD/image/'
    img_list = [os.path.join(img_path, x)
                for x in os.listdir(img_path) if x.endswith('png')]

    with torch.no_grad():
        for path in img_list:
            detect(net, path, args.thresh)
            # write_txt(net, path, args.thresh)
