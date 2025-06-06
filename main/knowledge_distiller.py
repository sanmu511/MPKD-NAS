# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

'''
Author: Jiajie Chen, Helong Zhou.

Implemented the following paper:
Helong Zhou, Liangchen Song, Jiajie Chen, Ye Zhou, Guoli Wang, Junsong Yuan, Qian Zhang. "Rethinking Soft Labels for Knowledge Distillation: A Bias-Variance Tradeoff Perspective" (ICLR2021)
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
# from imagenet_train_cfg import cfg as config
# from tools import utils


class WSLDistiller(nn.Module):
    def __init__(self, t_net, s_net):
        super(WSLDistiller, self).__init__()

        self.t_net = t_net
        self.s_net = s_net

        self.T = 2
        self.alpha = 2.5

        self.softmax = nn.Softmax(dim=1).cuda()
        self.logsoftmax = nn.LogSoftmax().cuda()

        # if config.optim.label_smooth:
        #     self.hard_loss = utils.cross_entropy_with_label_smoothing
        # else:
        self.hard_loss = nn.CrossEntropyLoss()
        self.hard_loss = self.hard_loss.cuda()

    def caculate_wsl(self, s_logits, t_logits, label):
        s_input_for_softmax = s_logits / self.T
        t_input_for_softmax = t_logits / self.T

        t_soft_label = self.softmax(t_input_for_softmax)

        softmax_loss = - torch.sum(t_soft_label * self.logsoftmax(s_input_for_softmax), 1, keepdim=True)  # Lkd

        fc_s_auto = s_logits.detach()
        fc_t_auto = t_logits.detach()
        log_softmax_s = self.logsoftmax(fc_s_auto)
        log_softmax_t = self.logsoftmax(fc_t_auto)
        one_hot_label = F.one_hot(label.to(torch.int64), num_classes=5).float()
        softmax_loss_s = - torch.sum(one_hot_label * log_softmax_s, 1, keepdim=True)
        softmax_loss_t = - torch.sum(one_hot_label * log_softmax_t, 1, keepdim=True)

        focal_weight = softmax_loss_s / (softmax_loss_t + 1e-7)
        ratio_lower = torch.zeros(1).cuda()
        focal_weight = torch.max(focal_weight, ratio_lower)
        focal_weight = 1 - torch.exp(- focal_weight)
        softmax_loss = focal_weight * softmax_loss
        return softmax_loss


    def forward(self, x, label):

        fc_t,ss_t_logits = self.t_net(x)
        fc_s,ss_logits = self.s_net(x)

        softmax_loss_total = self.caculate_wsl(fc_s,fc_t,label)
        softmax_loss_extra = []
        for i in range(len(ss_logits)):
            wsl_extra = self.caculate_wsl(ss_logits[i],ss_t_logits[i],label)
            softmax_loss_extra.append(wsl_extra)

        soft_loss = (self.T ** 2) * torch.mean(softmax_loss_total)
        for i in range(len(softmax_loss_extra)):
            soft_loss += (self.T ** 2) * torch.mean(softmax_loss_extra[i])

        hard_loss = self.hard_loss(fc_s, label.long())
        loss = hard_loss + self.alpha * soft_loss

        return fc_s, loss
