import torch
import torch.nn as nn
from part_enhancement import *
from part_detection import *
from loss import *
from patchnce import PatchNCELoss
from part_F import *


class Network(nn.Module):

    def __init__(self, phase, num_classes):
        super(Network, self).__init__()

        self.phase = phase
        self.num_classes = num_classes
        self.iem_nums = 3
        self.enhance_channel = 3
        self.enhance_net = Res_Enhancement()
        self._enhance_criterion = LossFunction()
        self._contras_criterion = []
        self.nce_layers = [1,2,3,4]
        for nce_layer in self.nce_layers:
            self._contras_criterion.append(PatchNCELoss())
        self.netF = PatchSampleF(True, 'xavier', 0.02, 256)

        self.detection_net = DetectionNetwork(self.phase, self.num_classes)
        self._detection_criterion = MultiBoxLoss(cfg, True)
        self.constant = torch.Tensor([123, 117, 104]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        # self.constant = Variable(self.constant)

    def forward(self, inputs, contras=False):
        new_input = (inputs + self.constant) / 255
        enhance, illu = self.enhance_net(new_input)
        detect_img = enhance * 255 - self.constant
        if contras:
            feature = self.detection_net(new_input, True)
            return feature
        else:
            out = self.detection_net(detect_img, False)
            return enhance, illu, out


    def _enhcence_loss(self, inputs):
        enhance, illu, out = self.forward(inputs, False)
        new_input = (inputs + self.constant) / 255
        enhance_loss = self._enhance_criterion(new_input, illu)
        return enhance_loss

    def _detection_loss(self, inputs, targets):
        u_list, t_list, out = self.forward(inputs, False)
        loss_l_pa1l, loss_c_pal1 = self._detection_criterion(out[:3], targets)
        loss = loss_l_pa1l + loss_c_pal1
        return loss

    def _contras_loss(self, inputs, inputs2):
        enhance, _, _ = self.forward(inputs, False)
        feat_q = self.forward(enhance, True)
        feat_k = self.forward(inputs, True)
        feat_g = self.forward(inputs2, True)
        feat_k_pool, sample_ids = self.netF(feat_k, 256, None)
        feat_q_pool, _ = self.netF(feat_q, 256, sample_ids)
        feat_g_pool, _ = self.netF(feat_g, 256, sample_ids)
        # feat_k_pool, _ = self.netF(feat_k, 0, None)
        # feat_q_pool, _ = self.netF(feat_q, 0, None)
        # feat_g_pool, _ = self.netF(feat_g, 0, None)

        total_nce_loss = 0.0
        for f_q, f_k, f_g, crit in zip(feat_q_pool, feat_k_pool, feat_g_pool,
                                       self._contras_criterion):
            loss = crit(f_q, f_k, f_g)
            total_nce_loss = total_nce_loss + loss.mean()

        return total_nce_loss
