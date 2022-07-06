# -*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import sys

sys.path.append(".")
import time
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import myfunc
import cv2
import glob
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import logging
import sys
from data.config import cfg
from PIL import Image
from data.widerface import WIDERDetection, detection_collate
from model import Network
c


parser = argparse.ArgumentParser(description='DSFD face Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--batch_size',
                    default=2, type=int,
                    help='Batch size for training')
parser.add_argument('--num_workers',
                    default=0, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda',
                    default=True, type=bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate',
                    default=5e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum',
                    default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay',
                    default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma',
                    default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--gpu', type=str,
                    default="0", help='gpu device id')
parser.add_argument('--multigpu',
                    default=False, type=bool,
                    help='Use mutil Gpu training')
parser.add_argument('--save',
                    default='EXP',
                    help='Directory for saving checkpoint models')
parser.add_argument('--grad_clip',
                    type=float, default=5,
                    help='gradient clipping')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def save_log():
    path = os.path.split(__file__)[0].split('/')[-1]
    args.save = args.save + '/' + path + '-detection-loss-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    myfunc.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
    global model_path
    model_path = args.save + '/model_epochs/'
    os.makedirs(model_path, exist_ok=True)
    global image_path
    image_path = args.save + '/image_epochs/'
    os.makedirs(image_path, exist_ok=True)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info("train file name = %s", os.path.split(__file__))


# if not args.multigpu:
#     os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

train_dataset = WIDERDetection(cfg.FACE.TRAIN_FILE, mode='train')

val_dataset = WIDERDetection(cfg.FACE.VAL_FILE, mode='val')

train_loader = data.DataLoader(train_dataset, args.batch_size,
                               num_workers=args.num_workers,
                               shuffle=True,
                               collate_fn=detection_collate,
                               pin_memory=False,
                               generator=torch.Generator(device='cuda'))
val_batchsize = args.batch_size  # // 2
val_loader = data.DataLoader(val_dataset, val_batchsize,
                             num_workers=args.num_workers,
                             shuffle=False,
                             collate_fn=detection_collate,
                             pin_memory=False)
per_epoch_size = len(train_dataset) // args.batch_size

min_loss = np.inf
iteration = 0

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

img_path = 'E:/hj/detec/darkface1000/input/1002.png'


def main():
    save_log()
    global iteration
    start_epoch = 1

    step_index = 0

    model = Network('train', 2)

    # for p, v in model.named_parameters():
    #     if v.requires_grad == True:
    #         print(p)

    if args.cuda:
        # torch.cuda.set_device(args.gpu)
        # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        model = model.cuda()
        cudnn.benckmark = True

    optimizer_En = optim.Adam(model.enhance_net.parameters(), lr=args.lr, betas=(0.5, 0.999))
    


    print('Loading wider dataset...')
    print('Using the specified args:')
    print(args)
    logging.info("args = %s", args)

    log_dir = 'D:/hj/newSSD/EXP/Base_FPN-EXP-20211104-183228/model_epochs/dsfd.pth'
    if os.path.exists(log_dir):
        # checkpoint = torch.load(log_dir)
        # model.detection_net.load_state_dict(torch.load({k.replace('detection_net.', ''): v for k, v in checkpoint.items()})['weight'])
        model.detection_net.load_state_dict(torch.load(log_dir), False)

    model.train()
    for epoch in range(start_epoch, cfg.EPOCHES):
        train(train_loader, model, optimizer_En, epoch)
        if epoch % 1 == 0:
            file = 'epoch_' + repr(epoch) + '.pth'
            torch.save(model.enhance_net.state_dict(), os.path.join(image_path, file))
        model.eval()
        with torch.no_grad():
            img = Image.open(img_path)
            img = np.array(img)
            height, width, _ = img.shape
            max_im_shrink = np.sqrt(
                1500 * 1000 / (img.shape[0] * img.shape[1]))
            image = cv2.resize(img, None, None, fx=max_im_shrink,
                               fy=max_im_shrink, interpolation=cv2.INTER_LINEAR)

            x = to_chw_bgr(image)
            x = x.astype('float32')
            # print(cfg.img_mean)
            # print(x.shape)
            x -= cfg.img_mean
            x = x[[2, 1, 0], :, :]

            x = Variable(torch.from_numpy(x).unsqueeze(0))
            x = x.cuda()
            enhance, _, _ = model(x, False)

            save_enhance_img_path = 'D:/hj/coSSD/tmp/test--/img_enhance1/'
            os.makedirs(save_enhance_img_path, exist_ok=True)
            save_enhance_file = save_enhance_img_path + str(epoch) + '_1002.png'
            save_images(enhance, save_enhance_file)



def train(train_loader, model, optimizer_En, epoch):
    en_losses = 0
    co_losses = 0
    global iteration
    for batch_idx, (images, images2, _) in enumerate(train_loader):
        t0 = time.time()
        if args.cuda:
            images = Variable(images.cuda())
            images2 = Variable(images2.cuda())

        else:
            images = Variable(images)
            images2 = Variable(images2)


        if epoch == 1 and batch_idx == 0:
            E = model._enhcence_loss(images)
            C = model._contras_loss(images, images2)
            EC = E + C
            EC.backward(retain_graph=True)
            optimizer_F = optim.Adam(model.netF.parameters(), lr=0.0001, betas=(0.5, 0.999))
        optimizer_F = optim.Adam(model.netF.parameters(), lr=0.0001, betas=(0.5, 0.999))


        optimizer_En.zero_grad()
        optimizer_F.zero_grad()
        en_loss = model._enhcence_loss(images)
        ce_loss = model._contras_loss(images, images2)
        Eloss = en_loss + ce_loss
        Eloss.backward()
        #nn.utils.clip_grad_norm_(model.enhance_net.parameters(), args.grad_clip)
        optimizer_En.step()
        optimizer_F.step()
        co_losses += ce_loss.item()


        en_losses += en_loss.item()
        co_losses += ce_loss.item()
        t1 = time.time()
        if iteration % 100 == 0:
            eloss = en_losses / (batch_idx + 1)
            closs = co_losses / (batch_idx + 1)
            logging.info('Timer: %.4f' % (t1 - t0))
            logging.info('epoch:' + repr(epoch) + ' || iter:' +
                         repr(iteration) + ' || eLoss:%.8f' % (eloss) + ' || ')
            logging.info('epoch:' + repr(epoch) + ' || iter:' +
                         repr(iteration) + ' || cLoss:%.8f' % (closs) + ' || ')

        iteration += 1





def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_images(tensor, path):
    image_numpy = tensor[0].cpu().detach().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    im.save(path, 'png')


if __name__ == '__main__':
    main()
