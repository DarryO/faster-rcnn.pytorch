import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model.utils.config import cfg
from model.rpn.rpn import _RPN

from model.roi_layers import ROIAlign, ROIPool

from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from model.utils.net_utils import _smooth_l1_loss


class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN, self).__init__()
        print(classes)
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)

        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / 16.0)
        self.RCNN_roi_align = ROIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / 16.0, 0)

    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        # im_data: [batch, channel, height, width]
        # im_info: [batch, 3] (height, width, scale)
        # num_boxes: [batch]
        # gt_boxes: [batch, MAX_NUM_GTBOX, 5] (x1, y1, x2, y2, cls)
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)

        # feed base feature map tp RPN to obtain rois
        # rois: [batch_size, RPN_POST_NMS_TOP_N, 5] (batch_index, x, y, x, y)
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(
                base_feat, im_info, gt_boxes[:, :, :5] if gt_boxes.dim == 3 else gt_boxes[:, :5], num_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(
                    rois, gt_boxes[:, :, :5] if gt_boxes.dim == 3 else gt_boxes[:, :5], num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        # do roi pooling based on predicted rois

        # ROI feature
        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5))

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        dim_offset_pred = self.RCNN_dim_offset_pred(pooled_feat)
        loc_offset_pred = self.RCNN_loc_offset_pred(pooled_feat)

        def target_select(all_class_pred, label, num_feat):
            all_class_view = all_class_pred.view(all_class_pred.size(0), int(all_class_pred.size(1) / num_feat), num_feat)
            target_pred_select = torch.gather(all_class_view, 1, label.view(label.size(0), 1, 1).expand(label.size(0), 1, num_feat))
            target_pred = target_pred_select.squeeze(1)
            return target_pred

        # if class_agnostic is True, then
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred = target_select(bbox_pred, rois_label, 4)
            dim_offset_pred = target_select(dim_offset_pred, rois_label, 3)
            loc_offset_pred = target_select(loc_offset_pred, rois_label, 3)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        alpha_pred = self.RCNN_alpha_pred(pooled_feat)
        ry_pred = self.RCNN_ry_pred(pooled_feat)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                # not a perfect approximation
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

        normal_init(self.RCNN_dim_offset_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_loc_offset_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_alpha_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_ry_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
