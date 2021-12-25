#encoding=utf-8
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN

from model.roi_layers import ROIAlign, ROIPool

# from model.roi_pooling.modules.roi_pool import _RoIPooling
# from model.roi_align.modules.roi_align import RoIAlignAvg

from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta

class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)

        # self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        # self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)
        self.RCNN_roi_align = ROIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0, 0)

        self.lamda = 0.8

        self.G_cls_feat = torch.zeros(self.n_classes, 4096).cuda()
        self.not_first = False
        self.label = torch.eye(self.n_classes, self.n_classes).cuda()
        # for i in range(self.n_classes):
        #     self.label[i] = i

    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)
        # base_feat = base_feat1.detach()


        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            # feed base feature map tp RPN to obtain rois
            self.RCNN_rpn.train()
            rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)
            self.RCNN_rpn.eval()
            rois1, rpn_loss_cls1, rpn_loss_bbox1 = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)
            # pdb.set_trace()

            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)
            rois1, rpn_loss_cls1, rpn_loss_bbox1 = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        rois1 = Variable(rois1)
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == 'align':
            pooled_feat1 = self.RCNN_roi_align(base_feat, rois1.view(-1, 5).detach())
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5).detach())
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat1 = self.RCNN_roi_pool(base_feat, rois1.view(-1,5).detach())
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5).detach())
        # pdb.set_trace()
        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)
        pooled_feat1 = self._head_to_tail(pooled_feat1)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)

        if self.training and not self.class_agnostic:
            # pdb.set_trace()
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        cls_score1 = self.RCNN_cls_score(pooled_feat1)
        cls_prob1 = F.softmax(cls_score1, 1)

        cls_loss_cls, cls_entropy_loss, sim_loss_cls = self.fun(cls_prob1.detach(), cls_score1, pooled_feat1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:

            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)


        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, cls_loss_cls, cls_entropy_loss, sim_loss_cls#, pooled_feat

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()


    def fun(self, cls_prob, cls_score, pooled_feat):
        region_prob_sum = 1/(torch.t(cls_prob).sum(1))
        region_prob_weight = region_prob_sum.unsqueeze(1).expand(cls_prob.size(1), cls_prob.size(0)).mul(torch.t(cls_prob))
        cls_feat = torch.mm(region_prob_weight, pooled_feat)

        if self.not_first:
            alpha = ((cls_feat.mul(self.G_cls_feat).sum(1)/(torch.norm(cls_feat, 2, 1).mul(torch.norm(self.G_cls_feat, 2, 1)))).unsqueeze(1)+1)/2
        else:
            alpha = torch.ones(self.n_classes,1).cuda()
            self.not_first = True
        self.G_cls_feat = cls_feat.mul(alpha.detach())+(self.G_cls_feat.detach()).mul(1-alpha.detach())
        # pdb.set_trace()

        sim_cls_score = torch.mm(pooled_feat, torch.t(self.G_cls_feat.detach()))/torch.mm(torch.norm(pooled_feat, 2, 1).unsqueeze(1), torch.norm(self.G_cls_feat.detach(), 2, 1).unsqueeze(0))

        sim_label = (10 * sim_cls_score + cls_prob).max(1)[1]
        sim_loss_cls = F.cross_entropy(cls_score, sim_label.detach())

        cls_cls_score = self.RCNN_cls_score(self.G_cls_feat)
        cls_cls_prob = (F.softmax(cls_cls_score, 1)>(1.0/self.n_classes)).float()

        cls_loss_cls = -F.log_softmax(cls_cls_score, 1).mul(self.label).mul(cls_cls_prob.detach()).sum(0).sum(0) / cls_cls_score.size(0)#goodcls.sum(0)
        # pdb.set_trace()

        cls_entropy_loss = self.entropy_loss(cls_score)

        return cls_loss_cls, cls_entropy_loss, sim_loss_cls

    def entropy_loss(self, score):
        #pdb.set_trace()
        num = -F.log_softmax(score, 1).mul(F.softmax(score, 1)).sum(0).sum(0)/score.size(0)

        return num




    #reason_function
    def reason_run(self, pooled_feat, cls_score, cls_prob, rois):
        reason_feat, mask = self.reason(pooled_feat.detach(), cls_score.detach(), rois.detach())
        reason_score = self.RCNN_cls_score(reason_feat)
        reason_prob = F.softmax(reason_score, 1)

        right_region = mask.byte().squeeze(1).mul(reason_prob.max(1)[1] == cls_prob.max(1)[1])
        cls_score = cls_score + right_region.float().unsqueeze(1).expand(reason_score.size(0), reason_score.size(1)).mul(reason_score)
        right_inds = torch.nonzero(right_region > 0).view(-1).detach()

        right_score = cls_score[right_inds]
        fake_label = cls_prob.max(1)[1][right_inds]



    def reason(self, pooled_feat, cls_score, rois):

        # 提取batch大小，候选区域个数，类别数目； B，R，C
        # train：4,256,101； test：1,300,101
        batch_size = rois.size(0)
        region_num = rois.size(1)
        fea_num = pooled_feat.size(1)
        # pdb.set_trace()

        mask = self.back_goodinds(cls_score, batch_size*region_num)
        mask_cls_score = cls_score.mul(mask.expand(batch_size*region_num, cls_score.size(1)))

        raw_region_box = torch.cuda.FloatTensor(rois.data[:, :, 1:5])

        trs_region_box = self.__transform_box(raw_region_box)  # raw_region_box)


        # 分离B,R，调整大小为B*R*C
        temp_feat = pooled_feat.view(batch_size, region_num, -1)
        temp_score = mask_cls_score.view(batch_size, region_num, -1)
        temp_mask = mask.view(batch_size, region_num, -1)
        # pdb.set_trace()
        # 新建一个全零B*R*C，用于存储每张图片的推理结果
        temp0 = torch.zeros(batch_size, region_num, fea_num).cuda()

        for i in range(batch_size):
            # 使用nms筛选重复框得到一个R*1的mask
            # pdb.set_trace()
            region_fea = temp_feat[i, :, :]
            # region_score = temp_score[i, :, :]
            region_mask = temp_mask[i, :, :]

            A_spread = self.__Build_spread(trs_region_box[i, :, :], region_num, region_mask)

            spread_reason_fea = torch.mm(A_spread, region_fea)

            # norm = 1.0 / (spread_reason_fea.sum(1) + 0.0001).unsqueeze(1).expand(region_num, fea_num)  # 50
            # spread_reason_fea = spread_reason_fea.mul(norm)
            choose_fea = spread_reason_fea  # 0.07

            # pdb.set_trace()
            # 将每张图片的推理结果保存到temp3中
            temp0[i, :, :] = temp0[i, :, :] + choose_fea

        reason_feat = temp0.view(batch_size*region_num, -1)

        return reason_feat, mask

    def __Build_spread(self, region_box, region_num, mask):

        # 坐标扩展为3维,并转置，用于计算区域之间的距离
        expand1_region_box = region_box.unsqueeze(2)
        expand_region_box = expand1_region_box.expand(region_num, 4, region_num)
        transpose_region_box = expand_region_box.transpose(0, 2)

        # 跟据每个区域的w和h，计算每个区域的传播范围
        spread_distance = torch.sqrt(torch.pow(region_box[:, 2], 2) + torch.pow(region_box[:, 3], 2))
        expand_spread_distance = (self.lamda * spread_distance).expand(region_num, region_num)

        # 根据每个区域的x和y，计算区域之间的距离
        region_distance = torch.sqrt(
            torch.pow((expand_region_box[:, 0, :] - transpose_region_box[:, 0, :]), 2) + torch.pow(
                (expand_region_box[:, 1, :] - transpose_region_box[:, 1, :]), 2))

        # A = F.relu(expand_spread_distance-region_distance)

        # 根据传播范围和距离，计算传播矩阵的权值
        A = F.relu(1 - region_distance / expand_spread_distance)

        # pdb.set_trace()
        #mask_w = torch.t(mask).expand(region_num, region_num)
        mask_w = torch.mm(mask, torch.t(mask))
        # 不接受来自自己的推理信息
        self_w = 1 - torch.eye(region_num, region_num).cuda()

        A = A.mul(self_w).mul(mask_w)

        weight = 1.0/(A.sum(1)+0.001).unsqueeze(1).expand(region_num, region_num)

        return A.mul(weight)

    def __transform_box(self, region_box):

        new_region_box = torch.zeros((region_box.size(0), region_box.size(1), 4)).cuda()
        new_region_box[:, :, 0] = 0.5 * (region_box[:, :, 0] + region_box[:, :, 2])
        new_region_box[:, :, 1] = 0.5 * (region_box[:, :, 1] + region_box[:, :, 3])
        new_region_box[:, :, 2] = region_box[:, :, 2] - region_box[:, :, 0]
        new_region_box[:, :, 3] = region_box[:, :, 3] - region_box[:, :, 1]

        return new_region_box

    def back_goodinds(self, region_fea, region_num):
        mask = (region_fea.max(1)[1]>0).float().unsqueeze(1)
        return mask