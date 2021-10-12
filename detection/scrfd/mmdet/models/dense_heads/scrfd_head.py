import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale, bias_init_with_prob, normal_init, DepthwiseSeparableConvModule
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import force_fp32

from mmdet.core import (anchor_inside_flags, bbox2distance, bbox_overlaps,
                        build_assigner, build_sampler, distance2bbox, distance2kps, kps2distance,
                        images_to_levels, multi_apply, multiclass_nms,
                        reduce_mean, unmap)
from ..builder import HEADS, build_loss
from .anchor_head import AnchorHead
from .base_dense_head import BaseDenseHead
from .dense_test_mixins import BBoxTestMixin


def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
    else:
        constant_init(m, val=0)


class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution.

    This layer calculates the target location by :math: `sum{P(y_i) * y_i}`,
    P(y_i) denotes the softmax vector that represents the discrete distribution
    y_i denotes the discrete set, usually {0, 1, 2, ..., reg_max}

    Args:
        reg_max (int): The maximal value of the discrete set. Default: 16. You
            may want to reset it according to your new dataset or related
            settings.
    """

    def __init__(self, reg_max=16):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.register_buffer('project',
                             torch.linspace(0, self.reg_max, self.reg_max + 1))

    def forward(self, x):
        """Forward feature from the regression head to get integral result of
        bounding box location.

        Args:
            x (Tensor): Features of the regression head, shape (N, 4*(n+1)),
                n is self.reg_max.

        Returns:
            x (Tensor): Integral result of box locations, i.e., distance
                offsets from the box center in four directions, shape (N, 4).
        """
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, self.project.type_as(x)).reshape(-1, 4)
        return x


@HEADS.register_module()
class SCRFDHead(AnchorHead):
    """Generalized Focal Loss: Learning Qualified and Distributed Bounding
    Boxes for Dense Object Detection.

    GFL head structure is similar with ATSS, however GFL uses
    1) joint representation for classification and localization quality, and
    2) flexible General distribution for bounding box locations,
    which are supervised by
    Quality Focal Loss (QFL) and Distribution Focal Loss (DFL), respectively

    https://arxiv.org/abs/2006.04388

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        stacked_convs (int): Number of conv layers in cls and reg tower.
            Default: 4.
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='GN', num_groups=32, requires_grad=True).
        loss_qfl (dict): Config of Quality Focal Loss (QFL).
        reg_max (int): Max value of integral set :math: `{0, ..., reg_max}`
            in QFL setting. Default: 16.
    Example:
        >>> self = GFLHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_quality_score, bbox_pred = self.forward(feats)
        >>> assert len(cls_quality_score) == len(self.scales)
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 feat_mults=None,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 loss_dfl=None,
                 reg_max=8,
                 cls_reg_share=False,
                 strides_share=True,
                 scale_mode = 1,
                 dw_conv = False,
                 use_kps = False,
                 loss_kps=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=0.1),
                 #loss_kps=dict(type='SmoothL1Loss', beta=1.0, loss_weight=0.3),
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.feat_mults = feat_mults
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.reg_max = reg_max
        self.cls_reg_share = cls_reg_share
        self.strides_share = strides_share
        self.scale_mode = scale_mode
        self.use_dfl = True
        self.dw_conv = dw_conv
        self.NK = 5
        self.extra_flops = 0.0
        if loss_dfl is None or not loss_dfl:
            self.use_dfl = False
        self.use_scale = False
        self.use_kps = use_kps
        if self.scale_mode>0 and (self.strides_share or self.scale_mode==2):
            self.use_scale = True
        #print('USE-SCALE:', self.use_scale)
        super(SCRFDHead, self).__init__(num_classes, in_channels, **kwargs)

        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # SSD sampling=False so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.integral = Integral(self.reg_max)
        if self.use_dfl:
            self.loss_dfl = build_loss(loss_dfl)
        #print('USE_DFL:', self.use_dfl)
        self.loss_kps = build_loss(loss_kps)
        self.loss_kps_std = 1.0
        #print(self.bbox_coder.__class__)
        self.train_step = 0
        self.pos_count = {}
        self.gtgroup_count = {}
        for stride in self.anchor_generator.strides:
            self.pos_count[stride[0]] = 0

    def _get_conv_module(self, in_channel, out_channel):
        if not self.dw_conv:
            conv = ConvModule(
                    in_channel,
                    out_channel,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg)
        else:
            conv = DepthwiseSeparableConvModule(
                    in_channel,
                    out_channel,
                    3,
                    stride=1,
                    padding=1,
                    pw_norm_cfg=self.norm_cfg,
                    dw_norm_cfg=self.norm_cfg)
        return conv

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        conv_strides = [0] if self.strides_share else self.anchor_generator.strides
        self.cls_stride_convs = nn.ModuleDict()
        self.reg_stride_convs = nn.ModuleDict()
        self.stride_cls = nn.ModuleDict()
        self.stride_reg = nn.ModuleDict()
        if self.use_kps:
            self.stride_kps = nn.ModuleDict()
        for stride_idx, conv_stride in enumerate(conv_strides):
            #print('create convs for stride:', conv_stride)
            key = str(conv_stride)
            cls_convs = nn.ModuleList()
            reg_convs = nn.ModuleList()
            stacked_convs = self.stacked_convs[stride_idx] if isinstance(self.stacked_convs, (list, tuple)) else self.stacked_convs
            feat_mult = self.feat_mults[stride_idx] if self.feat_mults is not None else 1
            feat_ch = int(self.feat_channels*feat_mult)
            for i in range(stacked_convs):
                chn = self.in_channels if i == 0 else last_feat_ch
                cls_convs.append( self._get_conv_module(chn, feat_ch) )
                if not self.cls_reg_share:
                    reg_convs.append( self._get_conv_module(chn, feat_ch) )
                last_feat_ch = feat_ch
            self.cls_stride_convs[key] = cls_convs
            self.reg_stride_convs[key] = reg_convs
            self.stride_cls[key] = nn.Conv2d(
                feat_ch, self.cls_out_channels * self.num_anchors, 3, padding=1)
            if not self.use_dfl:
                self.stride_reg[key] = nn.Conv2d(
                    feat_ch, 4 * self.num_anchors, 3, padding=1)
            else:
                self.stride_reg[key] = nn.Conv2d(
                    feat_ch, 4 * (self.reg_max + 1) * self.num_anchors, 3, padding=1)
            if self.use_kps:
                self.stride_kps[key] = nn.Conv2d(
                    feat_ch, self.NK*2*self.num_anchors, 3, padding=1)
        #assert self.num_anchors == 1, 'anchor free version'
        #extra_gflops /= 1e9
        #print('extra_gflops: %.6fG'%extra_gflops)
        if self.use_scale:
            self.scales = nn.ModuleList(
                [Scale(1.0) for _ in self.anchor_generator.strides])
        else:
            self.scales = [None for _ in self.anchor_generator.strides]

    def init_weights(self):
        """Initialize weights of the head."""
        for stride, cls_convs in self.cls_stride_convs.items():
            #print('init cls for stride:', stride)
            for m in cls_convs:
                if not self.dw_conv:
                    try:
                        normal_init(m.conv, std=0.01)
                    except:
                        pass
                else:
                    normal_init(m.depthwise_conv.conv, std=0.01)
                    normal_init(m.pointwise_conv.conv, std=0.01)
        for stride, reg_convs in self.reg_stride_convs.items():
            for m in reg_convs:
                if not self.dw_conv:
                    normal_init(m.conv, std=0.01)
                else:
                    normal_init(m.depthwise_conv.conv, std=0.01)
                    normal_init(m.pointwise_conv.conv, std=0.01)
        #bias_cls = bias_init_with_prob(0.01)
        bias_cls = -4.595
        #bias_cls = -1.595
        for stride, conv in self.stride_cls.items():
            normal_init(conv, std=0.01, bias=bias_cls)
        for stride, conv in self.stride_reg.items():
            normal_init(conv, std=0.01)
        if self.use_kps:
            for stride, conv in self.stride_kps.items():
                normal_init(conv, std=0.01)

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
                cls_scores (list[Tensor]): Classification and quality (IoU)
                    joint scores for all scale levels, each is a 4D-tensor,
                    the channel number is num_classes.
                bbox_preds (list[Tensor]): Box distribution logits for all
                    scale levels, each is a 4D-tensor, the channel number is
                    4*(n+1), n is max value of integral set.
        """
        return multi_apply(self.forward_single, feats, self.scales, self.anchor_generator.strides)

    def forward_single(self, x, scale, stride):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.

        Returns:
            tuple:
                cls_score (Tensor): Cls and quality joint scores for a single
                    scale level the channel number is num_classes.
                bbox_pred (Tensor): Box distribution logits for a single scale
                    level, the channel number is 4*(n+1), n is max value of
                    integral set.
        """
        cls_feat = x
        reg_feat = x
        #print('forward_single in stride:', stride)
        cls_convs = self.cls_stride_convs['0'] if self.strides_share else self.cls_stride_convs[str(stride)]
        for cls_conv in cls_convs:
            cls_feat = cls_conv(cls_feat)
        if not self.cls_reg_share:
            reg_convs = self.reg_stride_convs['0'] if self.strides_share else self.reg_stride_convs[str(stride)]
            for reg_conv in reg_convs:
                reg_feat = reg_conv(reg_feat)
        else:
            reg_feat = cls_feat
        cls_pred_module = self.stride_cls['0'] if self.strides_share else self.stride_cls[str(stride)]
        cls_score = cls_pred_module(cls_feat)
        reg_pred_module = self.stride_reg['0'] if self.strides_share else self.stride_reg[str(stride)]
        _bbox_pred = reg_pred_module(reg_feat)
        if self.use_scale:
            bbox_pred = scale(_bbox_pred)
        else:
            bbox_pred = _bbox_pred
        if self.use_kps:
            kps_pred_module = self.stride_kps['0'] if self.strides_share else self.stride_kps[str(stride)]
            kps_pred = kps_pred_module(reg_feat)
        else:
            kps_pred = bbox_pred.new_zeros( (bbox_pred.shape[0], self.NK*2, bbox_pred.shape[2], bbox_pred.shape[3]) )
        if torch.onnx.is_in_onnx_export():
            assert not self.use_dfl
            print('in-onnx-export', cls_score.shape, bbox_pred.shape)
            #print(scale.parameters())
            #for p in scale.parameters():
                #print(p.name, p.data)
                #scale_val = p.data.item()
                #print(scale_val)
            #print('EEE1', cls_score.shape)
            #cls_score = torch.sigmoid(cls_score).reshape(1, self.cls_out_channels, -1).permute(0, 2, 1)
            #print('EEE2', cls_score.shape)
            #if self.use_dfl:
            #    bbox_pred = self.integral(bbox_pred) * stride[0]
            #else:
            #    bbox_pred = bbox_pred.reshape( (-1,4) ) * stride[0]
            #if self.use_dfl:
            #    bbox_pred = bbox_pred.reshape(1, (self.reg_max+1)*4, -1).permute(0, 2, 1)
            #    bbox_pred = bbox_pred.reshape( (1, -1, 4, self.reg_max+1) )
            #    bbox_pred = F.softmax(bbox_pred, dim=3)
            #else:
            #    bbox_pred = bbox_pred.reshape(1, 4, -1).permute(0, 2, 1)
            #kps_pred = kps_pred.reshape(1, 10, -1).permute(0, 2, 1)

            # Add output batch dim, based on pull request #1593
            batch_size = cls_score.shape[0]
            cls_score = cls_score.permute(0, 2, 3, 1).reshape(batch_size, -1, self.cls_out_channels).sigmoid()
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
            kps_pred = kps_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 10)

        return cls_score, bbox_pred, kps_pred

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_keypointss=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, gt_keypointss, img_metas)
        #print('AAA', gt_bboxes[0].shape, gt_keypointss[0].shape)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            return losses, proposal_list

    def get_anchors(self, featmap_sizes, img_metas, device='cuda'):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.
            device (torch.device | str): Device for returned tensors

        Returns:
            tuple:
                anchor_list (list[Tensor]): Anchors of each image.
                valid_flag_list (list[Tensor]): Valid flags of each image.
        """
        num_imgs = len(img_metas)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = self.anchor_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'], device)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list

    def anchor_center(self, anchors):
        """Get anchor centers from anchors.

        Args:
            anchors (Tensor): Anchor list with shape (N, 4), "xyxy" format.

        Returns:
            Tensor: Anchor centers with shape (N, 2), "xy" format.
        """
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        return torch.stack([anchors_cx, anchors_cy], dim=-1)

    def loss_single(self, anchors, cls_score, bbox_pred, kps_pred, labels, label_weights,
                    bbox_targets, kps_targets, kps_weights, stride, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            cls_score (Tensor): Cls and quality joint scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_pred (Tensor): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor wight
                shape (N, num_total_anchors, 4).
            stride (tuple): Stride in this scale level.
            num_total_samples (int): Number of positive samples that is
                reduced over all GPUs.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert stride[0] == stride[1], 'h stride is not equal to w stride!'
        use_qscore = True
        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        if not self.use_dfl:
            bbox_pred = bbox_pred.permute(0, 2, 3,
                                          1).reshape(-1, 4)
        else:
            bbox_pred = bbox_pred.permute(0, 2, 3,
                                          1).reshape(-1, 4 * (self.reg_max + 1))
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        if self.use_kps:
            kps_pred = kps_pred.permute(0, 2, 3,
                                          1).reshape(-1, self.NK*2)
            kps_targets = kps_targets.reshape( (-1, self.NK*2) )
            kps_weights = kps_weights.reshape( (-1, self.NK*2) )
            #print('AAA000', kps_targets.shape, kps_weights.shape)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)
        score = label_weights.new_zeros(labels.shape)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_anchor_centers = self.anchor_center(pos_anchors) / stride[0]

            weight_targets = cls_score.detach().sigmoid()
            weight_targets = weight_targets.max(dim=1)[0][pos_inds]
            pos_decode_bbox_targets = pos_bbox_targets / stride[0]


            if self.use_dfl:
                pos_bbox_pred_corners = self.integral(pos_bbox_pred)
                pos_decode_bbox_pred = distance2bbox(pos_anchor_centers,
                                                     pos_bbox_pred_corners)
            else:
                pos_decode_bbox_pred = distance2bbox(pos_anchor_centers,
                                                     pos_bbox_pred)
            if self.use_kps:
                pos_kps_targets = kps_targets[pos_inds]
                pos_kps_pred = kps_pred[pos_inds]
                #print('CCC000', kps_weights.shape)
                pos_kps_weights = kps_weights.max(dim=1)[0][pos_inds] * weight_targets
                #pos_kps_weights = kps_weights.max(dim=1)[0][pos_inds]
                pos_kps_weights = pos_kps_weights.reshape( (-1, 1) )
                #pos_kps_weights = kps_weights.max(dim=1, keepdims=True)[0][pos_inds]
                #print('SSS', pos_kps_weights.sum())

                #pos_decode_kps_targets = pos_kps_targets / stride[0]
                #pos_decode_kps_pred = distance2kps(pos_anchor_centers, pos_kps_pred)

                pos_decode_kps_targets = kps2distance(pos_anchor_centers, pos_kps_targets / stride[0])
                pos_decode_kps_pred = pos_kps_pred
                #print('ZZZ', pos_decode_kps_targets.shape, pos_decode_kps_pred.shape)
                #print(pos_kps_weights[0,:].detach().cpu().numpy())
                #print(pos_decode_kps_targets[0,:].detach().cpu().numpy())
                #print(pos_decode_kps_pred[0,:].detach().cpu().numpy())


                #print('CCC111', weight_targets.shape, pos_bbox_pred.shape, pos_decode_bbox_pred.shape, pos_kps_pred.shape, pos_decode_kps_pred.shape, pos_kps_weights.shape)

            if use_qscore:
                score[pos_inds] = bbox_overlaps(
                    pos_decode_bbox_pred.detach(),
                    pos_decode_bbox_targets,
                    is_aligned=True)
            else:
                score[pos_inds] = 1.0

            # regression loss
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=weight_targets,
                avg_factor=1.0)

            if self.use_kps:
                loss_kps = self.loss_kps(
                    pos_decode_kps_pred * self.loss_kps_std,
                    pos_decode_kps_targets * self.loss_kps_std,
                    weight=pos_kps_weights,
                    avg_factor=1.0)
            else:
                loss_kps = kps_pred.sum() * 0

            # dfl loss
            if self.use_dfl:
                pred_corners = pos_bbox_pred.reshape(-1, self.reg_max + 1)
                target_corners = bbox2distance(pos_anchor_centers,
                                               pos_decode_bbox_targets,
                                               self.reg_max).reshape(-1)
                loss_dfl = self.loss_dfl(
                    pred_corners,
                    target_corners,
                    weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                    avg_factor=4.0)
            else:
                loss_dfl = bbox_pred.sum() * 0
        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_dfl = bbox_pred.sum() * 0
            loss_kps = kps_pred.sum() * 0
            weight_targets = torch.tensor(0).cuda()

        loss_cls = self.loss_cls(
            cls_score, (labels, score),
            weight=label_weights,
            avg_factor=num_total_samples)


        return loss_cls, loss_bbox, loss_dfl, loss_kps, weight_targets.sum()

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             kps_preds,
             gt_bboxes,
             gt_labels,
             gt_keypointss,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Cls and quality scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_preds (list[Tensor]): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            gt_keypointss,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, keypoints_targets_list, keypoints_weights_list, num_total_pos, num_total_neg) = cls_reg_targets

        num_total_samples = reduce_mean(
            torch.tensor(num_total_pos, dtype=torch.float,
                         device=device)).item()
        num_total_samples = max(num_total_samples, 1.0)

        losses_cls, losses_bbox, losses_dfl, losses_kps,\
            avg_factor = multi_apply(
                self.loss_single,
                anchor_list,
                cls_scores,
                bbox_preds,
                kps_preds,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                keypoints_targets_list,
                keypoints_weights_list,
                self.anchor_generator.strides,
                num_total_samples=num_total_samples)

        #if self.train_step%100==0:
        #    print('loss_cls:', losses_cls)
        #    print('avg_factor:', avg_factor)


        avg_factor = sum(avg_factor)
        avg_factor = reduce_mean(avg_factor).item()
        losses_bbox = list(map(lambda x: x / avg_factor, losses_bbox))
        losses = dict(loss_cls=losses_cls, loss_bbox=losses_bbox)
        if self.use_kps:
            losses_kps = list(map(lambda x: x / avg_factor, losses_kps))
            losses['loss_kps'] = losses_kps
        if self.use_dfl:
            losses_dfl = list(map(lambda x: x / avg_factor, losses_dfl))
            losses['loss_dfl'] = losses_dfl
        return losses

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'kps_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   kps_preds,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class labelof the
                corresponding box.

        Example:
            >>> import mmcv
            >>> self = AnchorHead(
            >>>     num_classes=9,
            >>>     in_channels=1,
            >>>     anchor_generator=dict(
            >>>         type='AnchorGenerator',
            >>>         scales=[8],
            >>>         ratios=[0.5, 1.0, 2.0],
            >>>         strides=[4,]))
            >>> img_metas = [{'img_shape': (32, 32, 3), 'scale_factor': 1}]
            >>> cfg = mmcv.Config(dict(
            >>>     score_thr=0.00,
            >>>     nms=dict(type='nms', iou_thr=1.0),
            >>>     max_per_img=10))
            >>> feat = torch.rand(1, 1, 3, 3)
            >>> cls_score, bbox_pred = self.forward_single(feat)
            >>> # note the input lists are over different levels, not images
            >>> cls_scores, bbox_preds = [cls_score], [bbox_pred]
            >>> result_list = self.get_bboxes(cls_scores, bbox_preds,
            >>>                               img_metas, cfg)
            >>> det_bboxes, det_labels = result_list[0]
            >>> assert len(result_list) == 1
            >>> assert det_bboxes.shape[1] == 5
            >>> assert len(det_bboxes) == len(det_labels) == cfg.max_per_img
        """
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device=device)

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            if with_nms:
                # some heads don't support with_nms argument
                proposals = self._get_bboxes_single(cls_score_list,
                                                    bbox_pred_list,
                                                    mlvl_anchors, img_shape,
                                                    scale_factor, cfg, rescale)
            else:
                proposals = self._get_bboxes_single(cls_score_list,
                                                    bbox_pred_list,
                                                    mlvl_anchors, img_shape,
                                                    scale_factor, cfg, rescale,
                                                    with_nms)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True):
        """Transform outputs for a single batch item into labeled boxes.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                has shape (num_classes, H, W).
            bbox_preds (list[Tensor]): Box distribution logits for a single
                scale level with shape (4*(n+1), H, W), n is max value of
                integral set.
            mlvl_anchors (list[Tensor]): Box reference for a single scale level
                with shape (num_total_anchors, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple(Tensor):
                det_bboxes (Tensor): Bbox predictions in shape (N, 5), where
                    the first 4 columns are bounding box positions
                    (tl_x, tl_y, br_x, br_y) and the 5-th column is a score
                    between 0 and 1.
                det_labels (Tensor): A (N,) tensor where each item is the
                    predicted class label of the corresponding box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, stride, anchors in zip(
                cls_scores, bbox_preds, self.anchor_generator.strides,
                mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            assert stride[0] == stride[1]

            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0)
            if self.use_dfl:
                bbox_pred = self.integral(bbox_pred) * stride[0]
            else:
                bbox_pred = bbox_pred.reshape( (-1,4) ) * stride[0]

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]

            bboxes = distance2bbox(
                self.anchor_center(anchors), bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)

        mlvl_scores = torch.cat(mlvl_scores)
        # Add a dummy background class to the backend when using sigmoid
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        if with_nms:
            det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)
            return det_bboxes, det_labels
        else:
            return mlvl_bboxes, mlvl_scores

    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    gt_keypointss_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True):
        """Get targets for GFL head.

        This method is almost the same as `AnchorHead.get_targets()`. Besides
        returning the targets as the parent method does, it also returns the
        anchors as the first element of the returned tuple.
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        if gt_keypointss_list is None:
            gt_keypointss_list = [None for _ in range(num_imgs)]
        #print('QQQ:', num_imgs, gt_bboxes_list[0].shape)
        (all_anchors, all_labels, all_label_weights, all_bbox_targets,
         all_bbox_weights, all_keypoints_targets, all_keypoints_weights, 
         pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single,
             anchor_list,
             valid_flag_list,
             num_level_anchors_list,
             gt_bboxes_list,
             gt_bboxes_ignore_list,
             gt_labels_list,
             gt_keypointss_list,
             img_metas,
             label_channels=label_channels,
             unmap_outputs=unmap_outputs)
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        keypoints_targets_list = images_to_levels(all_keypoints_targets,
                                             num_level_anchors)
        keypoints_weights_list = images_to_levels(all_keypoints_weights,
                                             num_level_anchors)
        return (anchors_list, labels_list, label_weights_list,
                bbox_targets_list, bbox_weights_list, keypoints_targets_list, keypoints_weights_list,
                num_total_pos,
                num_total_neg)

    def _get_target_single(self,
                           flat_anchors,
                           valid_flags,
                           num_level_anchors,
                           gt_bboxes,
                           gt_bboxes_ignore,
                           gt_labels,
                           gt_keypointss,
                           img_meta,
                           label_channels=1,
                           unmap_outputs=True):
        """Compute regression, classification targets for anchors in a single
        image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors, 4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            num_level_anchors Tensor): Number of anchors of each scale level.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            img_meta (dict): Meta info of the image.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: N is the number of total anchors in the image.
                anchors (Tensor): All anchors in the image with shape (N, 4).
                labels (Tensor): Labels of all anchors in the image with shape
                    (N,).
                label_weights (Tensor): Label weights of all anchor in the
                    image with shape (N,).
                bbox_targets (Tensor): BBox targets of all anchors in the
                    image with shape (N, 4).
                bbox_weights (Tensor): BBox weights of all anchors in the
                    image with shape (N, 4).
                pos_inds (Tensor): Indices of postive anchor with shape
                    (num_pos,).
                neg_inds (Tensor): Indices of negative anchor with shape
                    (num_neg,).
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        num_level_anchors_inside = self.get_num_level_anchors_inside(
            num_level_anchors, inside_flags)
        if self.assigner.__class__.__name__=='ATSSAssigner':
            assign_result = self.assigner.assign(anchors, num_level_anchors_inside,
                                                 gt_bboxes, gt_bboxes_ignore,
                                                 gt_labels)
        else:
            assign_result = self.assigner.assign(anchors, 
                                                 gt_bboxes, gt_bboxes_ignore,
                                                 gt_labels)

        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        kps_targets = anchors.new_zeros(size=(anchors.shape[0], self.NK*2))
        kps_weights = anchors.new_zeros(size=(anchors.shape[0], self.NK*2))
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if self.use_kps:
                pos_assigned_gt_inds = sampling_result.pos_assigned_gt_inds
                #print('BBB', anchors.shape, gt_bboxes.shape, gt_keypointss.shape, pos_inds.shape, bbox_targets.shape, pos_bbox_targets.shape)
                kps_targets[pos_inds, :] = gt_keypointss[pos_assigned_gt_inds,:,:2].reshape( (-1, self.NK*2) )
                kps_weights[pos_inds, :] = torch.mean(gt_keypointss[pos_assigned_gt_inds,:,2], dim=1, keepdims=True)
            #kps_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(
                labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
            if self.use_kps:
                kps_targets = unmap(kps_targets, num_total_anchors, inside_flags)
                kps_weights = unmap(kps_weights, num_total_anchors, inside_flags)

        return (anchors, labels, label_weights, bbox_targets, bbox_weights,
                kps_targets, kps_weights,
                pos_inds, neg_inds)

    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_anchors_inside

    def aug_test(self, feats, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            feats (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains features for all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[ndarray]: bbox results of each class
        """
        return self.aug_test_bboxes(feats, img_metas, rescale=rescale)
