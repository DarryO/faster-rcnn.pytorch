from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.utils.config import cfg
from .proposal_layer import _ProposalLayer
from .anchor_target_layer import _AnchorTargetLayer
from model.utils.net_utils import _smooth_l1_loss


class _RPN(nn.Module):
    """ region proposal network """
    def __init__(self, din):
        super(_RPN, self).__init__()

        self.din = din  # get depth of input feature map, e.g., 512
        self.anchor_scales = cfg.ANCHOR_SCALES
        self.anchor_ratios = cfg.ANCHOR_RATIOS
        self.feat_stride = cfg.FEAT_STRIDE[0]

        # define the convrelu layers processing input feature map
        self.RPN_Conv = nn.Conv2d(
                in_channels=self.din,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True)

        # define bg/fg classifcation score layer
        # 2(bg/fg) * 9 (anchors)
        self.nc_score_out = len(self.anchor_scales) * len(self.anchor_ratios) * 2
        self.RPN_cls_score = nn.Conv2d(512, self.nc_score_out, 1, 1, 0)

        # define anchor box offset prediction layer
        # 4(coords) * 9 (anchors)
        self.nc_bbox_out = len(self.anchor_scales) * len(self.anchor_ratios) * 4
        self.RPN_bbox_pred = nn.Conv2d(512, self.nc_bbox_out, 1, 1, 0)

        # define proposal layer
        self.RPN_proposal = _ProposalLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        # define anchor target layer
        self.RPN_anchor_target = _AnchorTargetLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x

    def forward(self, base_feat, im_info, gt_boxes, num_boxes):

        batch_size = base_feat.size(0)

        # return feature map after convrelu layer
        # shape not change
        rpn_conv1 = F.relu(self.RPN_Conv(base_feat), inplace=True)

        # -------------- rpn bg/fg classification score ---------
        # get rpn classification score
        # outchannel: 2 * anchor
        # [batch, 2 * anchor, feat_h, feat_w]
        rpn_cls_score = self.RPN_cls_score(rpn_conv1)
        # rpn_cls_score_reshape: [batch, 2, anchor * feat_h, feat_w]
        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        # softmax on bg/fg
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)
        # [batch, 2 * anchor, feat_h, feat_w]
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)

        # ------------ rpn offsets -------------------------
        # get rpn *offsets* to the anchor boxes
        # [batch, 4 * anchor, feat_h, feat_w]
        rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv1)
        # ------------------------------------------------

        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'
        # rois: [batch_size, RPN_POST_NMS_TOP_N, 5] (batch_index, x, y, x, y)
        rois = self.RPN_proposal((rpn_cls_prob.data, rpn_bbox_pred.data,
                                 im_info, cfg_key))

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

        # generating training labels and build the rpn loss
        if self.training:
            assert gt_boxes is not None

            # rpn_data:
            #   labels: [batch, 1, A * feat_h, feat_w]
            #   bbox_targets: [batch, feat_h, feat_w, A * 4]
            #   bbox_inside_wieights: [batch, feat_height, feat_width, A * 4]
            #   bbox_outsize_wieights: [batch, feat_height, feat_width, A * 4]
            rpn_data = self.RPN_anchor_target((rpn_cls_score.data, gt_boxes, im_info, num_boxes))

            # compute classification loss
            # rpn_cls_score_reshape: [batch, 2, anchor * feat_h, feat_w]
            # rpn_cls_score: [batch, anchor * feat_h * feat_w,2]
            rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(
                    batch_size, -1, 2)

            # rpn_label: [batch, anchor * feat_h * feat_w]
            rpn_label = rpn_data[0].view(batch_size, -1)

            rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1))
            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1, 2), 0, rpn_keep)
            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data)
            rpn_label = Variable(rpn_label.long())
            self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label)
            # fg_cnt = torch.sum(rpn_label.data.ne(0))

            rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]

            # compute bbox regression loss
            rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)
            rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
            rpn_bbox_targets = Variable(rpn_bbox_targets)

            self.rpn_loss_box = _smooth_l1_loss(
                    rpn_bbox_pred, rpn_bbox_targets,
                    rpn_bbox_inside_weights, rpn_bbox_outside_weights,
                    sigma=3, dim=[1, 2, 3])

        return rois, self.rpn_loss_cls, self.rpn_loss_box
