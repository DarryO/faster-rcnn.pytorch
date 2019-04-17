# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models
from model.faster_rcnn.faster_rcnn import _fasterRCNN
import pdb

class vgg16(_fasterRCNN):
    def __init__(self, classes, pretrained=False, class_agnostic=False):
        self.model_path = 'data/pretrained_model/vgg16_caffe.pth'
        self.dout_base_model = 512
        self.pretrained = pretrained
        self.class_agnostic = class_agnostic

        _fasterRCNN.__init__(self, classes, class_agnostic)

    def _init_modules(self):
        vgg = models.vgg16()
        if self.pretrained:
            print("Loading pretrained weights from %s" % (self.model_path))
            state_dict = torch.load(self.model_path)
            print("111")
            vgg.load_state_dict({k: v for k, v in state_dict.items() if k in vgg.state_dict()}, strict=False)
            print("111")

        # not using the last Linear layer: in 4096, out 1000
        vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])

        self.RCNN_base = nn.Sequential(*list(vgg.features._modules.values())[:-1])

        # Fix the layers before conv3:
        for layer in range(10):
            for p in self.RCNN_base[layer].parameters():
                p.requires_grad = False

        # self.RCNN_base = _RCNN_base(vgg.features, self.classes, self.dout_base_model)

        self.RCNN_top = vgg.classifier

        # not using the last maxpool layer
        self.RCNN_cls_score = nn.Linear(4096, self.n_classes)

        if self.class_agnostic:
            self.RCNN_bbox_pred = nn.Linear(4096, 4)
            self.RCNN_dim_offset_pred = nn.Linear(4096, 3)
            self.RCNN_loc_offset_pred = nn.Linear(4096, 3)
        else:
            self.RCNN_bbox_pred = nn.Linear(4096, 4 * self.n_classes)
            # dimension offsets prediction(height, width, length)
            self.RCNN_dim_offset_pred = nn.Linear(4096, 3 * self.n_classes)
            self.RCNN_dim_offset_pred.requires_grad = False
            self.RCNN_loc_offset_pred = nn.Linear(4096, 3 * self.n_classes)
            self.RCNN_loc_offset_pred.requires_grad = False
        self.RCNN_alpha_pred = nn.Linear(4096, 1)
        self.RCNN_ry_pred = nn.Linear(4096, 1)
        for p in self.RCNN_dim_offset_pred.parameters():
            p.requires_grad = False
        for p in self.RCNN_loc_offset_pred.parameters():
            p.requires_grad = False
        for p in self.RCNN_alpha_pred.parameters():
            p.requires_grad = False
        for p in self.RCNN_ry_pred.parameters():
            p.requires_grad = False

    def _head_to_tail(self, pool5):

        pool5_flat = pool5.view(pool5.size(0), -1)
        fc7 = self.RCNN_top(pool5_flat)

        return fc7

