from collections import deque
import torch.nn.functional as F
import cv2

from utils import *
from transfuser import TransfuserBackbone, SegDecoder, DepthDecoder
from geometric_fusion import GeometricFusionBackbone
from late_fusion import LateFusionBackbone
from latentTF import latentTFBackbone
from copy import deepcopy
from point_pillar import PointPillarNet


from PIL import Image, ImageFont, ImageDraw
from torchvision import models

# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import bias_init_with_prob, normal_init
from mmcv.ops import batched_nms
from mmcv.runner import force_fp32

from mmdet.core import multi_apply
from mmdet.models import HEADS, build_loss
from mmdet.models.utils import gaussian_radius, gen_gaussian_target
from mmdet.models.utils.gaussian_target import (get_local_maximum, get_topk_from_heatmap,
                                     transpose_and_gather_feat)
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmdet.models.dense_heads.dense_test_mixins import BBoxTestMixin


@HEADS.register_module()
class LidarCenterNetHead(BaseDenseHead, BBoxTestMixin):
    """Objects as Points Head. CenterHead use center_point to indicate object's
    position. Paper link <https://arxiv.org/abs/1904.07850>

    Args:
        in_channel (int): Number of channel in the input feature map.
        feat_channel (int): Number of channel in the intermediate feature map.
        num_classes (int): Number of categories excluding the background
            category.
        loss_center_heatmap (dict | None): Config of center heatmap loss.
            Default: GaussianFocalLoss.
        loss_wh (dict | None): Config of wh loss. Default: L1Loss.
        loss_offset (dict | None): Config of offset loss. Default: L1Loss.
        train_cfg (dict | None): Training config. Useless in CenterNet,
            but we keep this variable for SingleStageDetector. Default: None.
        test_cfg (dict | None): Testing config of CenterNet. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channel,
                 feat_channel,
                 num_classes,
                 loss_center_heatmap=dict(type='GaussianFocalLoss', loss_weight=1.0),
                 loss_wh=dict(type='L1Loss', loss_weight=0.1),
                 loss_offset=dict(type='L1Loss', loss_weight=1.0),
                 loss_dir_class=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 loss_dir_res=dict(type='SmoothL1Loss', loss_weight=1.0),
                 loss_velocity=dict(type='L1Loss', loss_weight=1.0),
                 loss_brake=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(LidarCenterNetHead, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.heatmap_head = self._build_head(in_channel, feat_channel,
                                             num_classes)
        self.wh_head = self._build_head(in_channel, feat_channel, 2)
        self.offset_head = self._build_head(in_channel, feat_channel, 2)
        self.num_dir_bins = train_cfg.num_dir_bins
        self.yaw_class_head = self._build_head(in_channel, feat_channel, self.num_dir_bins)
        self.yaw_res_head = self._build_head(in_channel, feat_channel, 1)
        self.velocity_head = self._build_head(in_channel, feat_channel, 1)
        self.brake_head = self._build_head(in_channel, feat_channel, 2)

        self.loss_center_heatmap = build_loss(loss_center_heatmap)
        self.loss_wh = build_loss(loss_wh)
        self.loss_offset = build_loss(loss_offset)
        self.loss_dir_class = build_loss(loss_dir_class)
        self.loss_dir_res = build_loss(loss_dir_res)
        self.loss_velocity = build_loss(loss_velocity)
        self.loss_brake = build_loss(loss_brake)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = train_cfg.fp16_enabled
        self.i = 0

    def _build_head(self, in_channel, feat_channel, out_channel):
        """Build head for each branch."""
        layer = nn.Sequential(
            nn.Conv2d(in_channel, feat_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channel, out_channel, kernel_size=1))
        return layer

    def init_weights(self):
        """Initialize weights of the head."""
        bias_init = bias_init_with_prob(self.train_cfg.center_net_bias_init_with_prob)
        self.heatmap_head[-1].bias.data.fill_(bias_init)
        for head in [self.wh_head, self.offset_head]:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=self.train_cfg.center_net_normal_init_std)

    def forward(self, feats):
        """Forward features. Notice CenterNet head does not use FPN.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            center_heatmap_preds (List[Tensor]): center predict heatmaps for
                all levels, the channels number is num_classes.
            wh_preds (List[Tensor]): wh predicts for all levels, the channels
                number is 2.
            offset_preds (List[Tensor]): offset predicts for all levels, the
               channels number is 2.
        """
        return multi_apply(self.forward_single, feats)

    def forward_single(self, feat):
        """Forward feature of a single level.

        Args:
            feat (Tensor): Feature of a single level.

        Returns:
            center_heatmap_pred (Tensor): center predict heatmaps, the
               channels number is num_classes.
            wh_pred (Tensor): wh predicts, the channels number is 2.
            offset_pred (Tensor): offset predicts, the channels number is 2.
        """
        center_heatmap_pred = self.heatmap_head(feat).sigmoid()
        wh_pred = self.wh_head(feat)
        offset_pred = self.offset_head(feat)
        yaw_class_pred = self.yaw_class_head(feat)
        yaw_res_pred = self.yaw_res_head(feat)
        velocity_pred = self.velocity_head(feat)
        brake_pred = self.brake_head(feat)

        return center_heatmap_pred, wh_pred, offset_pred, yaw_class_pred, yaw_res_pred, velocity_pred, brake_pred

    @force_fp32(apply_to=('center_heatmap_preds', 'wh_preds', 'offset_preds', 'yaw_class_preds', 'yaw_res_preds', 'velocity_pred', 'brake_pred'))
    def loss(self,
             center_heatmap_preds,
             wh_preds,
             offset_preds,
             yaw_class_preds,
             yaw_res_preds,
             velocity_preds,
             brake_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            center_heatmap_preds (list[Tensor]): center predict heatmaps for
               all levels with shape (B, num_classes, H, W).
            wh_preds (list[Tensor]): wh predicts for all levels with
               shape (B, 2, H, W).
            offset_preds (list[Tensor]): offset predicts for all levels
               with shape (B, 2, H, W).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: which has components below:
                - loss_center_heatmap (Tensor): loss of center heatmap.
                - loss_wh (Tensor): loss of hw heatmap
                - loss_offset (Tensor): loss of offset heatmap.
        """
        assert len(center_heatmap_preds) == len(wh_preds) == len(offset_preds) == 1
        center_heatmap_pred = center_heatmap_preds[0]
        wh_pred = wh_preds[0]
        offset_pred = offset_preds[0]
        yaw_class_pred = yaw_class_preds[0]
        yaw_res_pred = yaw_res_preds[0]
        velocity_pred = velocity_preds[0]
        brake_pred = brake_preds[0]

        target_result, avg_factor = self.get_targets(gt_bboxes, gt_labels, gt_bboxes_ignore,
                                                     center_heatmap_pred.shape)
        
        center_heatmap_target = target_result['center_heatmap_target']
        wh_target = target_result['wh_target']
        yaw_class_target = target_result['yaw_class_target']
        yaw_res_target = target_result['yaw_res_target']
        offset_target = target_result['offset_target']
        velocity_target = target_result['velocity_target']
        brake_target = target_result['brake_target']
        wh_offset_target_weight = target_result['wh_offset_target_weight']

        # Since the channel of wh_target and offset_target is 2, the avg_factor
        # of loss_center_heatmap is always 1/2 of loss_wh and loss_offset.
        loss_center_heatmap = self.loss_center_heatmap(
            center_heatmap_pred, center_heatmap_target, avg_factor=avg_factor)
        loss_wh = self.loss_wh(
            wh_pred,
            wh_target,
            wh_offset_target_weight,
            avg_factor=avg_factor * 2)
        loss_offset = self.loss_offset(
            offset_pred,
            offset_target,
            wh_offset_target_weight,
            avg_factor=avg_factor * 2)
        loss_yaw_class = self.loss_dir_class(
            yaw_class_pred,
            yaw_class_target,
            wh_offset_target_weight[:, :1, ...],
            avg_factor=avg_factor)
        loss_yaw_res = self.loss_dir_res(
            yaw_res_pred,
            yaw_res_target,
            wh_offset_target_weight[:, :1, ...],
            avg_factor=avg_factor)
        loss_velocity = self.loss_velocity(
            velocity_pred,
            velocity_target,
            wh_offset_target_weight[:, :1, ...],
            avg_factor=avg_factor)
        loss_brake = self.loss_brake(
            brake_pred,
            brake_target,
            wh_offset_target_weight[:, :1, ...],
            avg_factor=avg_factor)

        return dict(
            loss_center_heatmap=loss_center_heatmap,
            loss_wh=loss_wh,
            loss_offset=loss_offset,
            loss_yaw_class=loss_yaw_class,
            loss_yaw_res=loss_yaw_res,
            loss_velocity=loss_velocity,
            loss_brake=loss_brake)

    def angle2class(self, angle):
        """Convert continuous angle to a discrete class and a residual.
        Convert continuous angle to a discrete class and a small
        regression number from class center angle to current angle.
        Args:
            angle (torch.Tensor): Angle is from 0-2pi (or -pi~pi),
                class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N).
        Returns:
            tuple: Encoded discrete class and residual.
        """
        angle = angle % (2 * np.pi)
        angle_per_class = 2 * np.pi / float(self.num_dir_bins)
        shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
        #NOTE changed this to not trigger a warning anymore. Rounding trunc should be the same as floor as long as angle is positive.
        # I kept it trunc to not change the behavior and keep backwards compatibility. When training a new model "floor" might be the better option.
        angle_cls = torch.div(shifted_angle, angle_per_class, rounding_mode="trunc")
        angle_res = shifted_angle - (angle_cls * angle_per_class + angle_per_class / 2)
        return angle_cls.long(), angle_res

    def class2angle(self, angle_cls, angle_res, limit_period=True):
        """Inverse function to angle2class.
        Args:
            angle_cls (torch.Tensor): Angle class to decode.
            angle_res (torch.Tensor): Angle residual to decode.
            limit_period (bool): Whether to limit angle to [-pi, pi].
        Returns:
            torch.Tensor: Angle decoded from angle_cls and angle_res.
        """
        angle_per_class = 2 * np.pi / float(self.num_dir_bins)
        angle_center = angle_cls.float() * angle_per_class
        angle = angle_center + angle_res
        if limit_period:
            angle[angle > np.pi] -= 2 * np.pi
        return angle

    def get_targets(self, gt_bboxes, gt_labels, gt_ignores, feat_shape):
        """Compute regression and classification targets in multiple images.

        Args:
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            feat_shape (list[int]): feature map shape with value [B, _, H, W]
            img_shape (list[int]): image shape in [h, w] format.

        Returns:
            tuple[dict,float]: The float value is mean avg_factor, the dict has
               components below:
               - center_heatmap_target (Tensor): targets of center heatmap, \
                   shape (B, num_classes, H, W).
               - wh_target (Tensor): targets of wh predict, shape \
                   (B, 2, H, W).
               - offset_target (Tensor): targets of offset predict, shape \
                   (B, 2, H, W).
               - wh_offset_target_weight (Tensor): weights of wh and offset \
                   predict, shape (B, 2, H, W).
        """
        img_h, img_w = self.train_cfg.lidar_resolution_height, self.train_cfg.lidar_resolution_width
        bs, _, feat_h, feat_w = feat_shape

        width_ratio = float(feat_w / img_w)
        height_ratio = float(feat_h / img_h)

        center_heatmap_target = gt_bboxes[-1].new_zeros(
            [bs, self.num_classes, feat_h, feat_w])
        wh_target = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])
        offset_target = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])
        yaw_class_target = gt_bboxes[-1].new_zeros([bs, 1, feat_h, feat_w]).long()
        yaw_res_target = gt_bboxes[-1].new_zeros([bs, 1, feat_h, feat_w])
        velocity_target = gt_bboxes[-1].new_zeros([bs, 1, feat_h, feat_w])
        brake_target = gt_bboxes[-1].new_zeros([bs, 1, feat_h, feat_w]).long()
 
        wh_offset_target_weight = gt_bboxes[-1].new_zeros(
            [bs, 2, feat_h, feat_w])

        for batch_id in range(bs):
            gt_bbox = gt_bboxes[0][batch_id]
            gt_label = gt_labels[0][batch_id]
            gt_ignore = gt_ignores[0][batch_id]

            center_x = gt_bbox[:, [0]] * width_ratio
            center_y = gt_bbox[:, [1]] * width_ratio
            gt_centers = torch.cat((center_x, center_y), dim=1)

            for j, ct in enumerate(gt_centers):
                if gt_ignore[j]:
                    continue

                ctx_int, cty_int = ct.int()
                ctx, cty = ct
                scale_box_h = gt_bbox[j, 3] * height_ratio
                scale_box_w = gt_bbox[j, 2] * width_ratio
                
                radius = gaussian_radius([scale_box_h, scale_box_w], min_overlap=0.1)
                radius = max(2, int(radius))
                ind = gt_label[j].long()
                
                gen_gaussian_target(center_heatmap_target[batch_id, ind], [ctx_int, cty_int], radius)

                wh_target[batch_id, 0, cty_int, ctx_int] = scale_box_w
                wh_target[batch_id, 1, cty_int, ctx_int] = scale_box_h
                
                yaw_class, yaw_res = self.angle2class(gt_bbox[j, 4])

                yaw_class_target[batch_id, 0, cty_int, ctx_int] = yaw_class
                yaw_res_target[batch_id, 0, cty_int, ctx_int] = yaw_res

                velocity_target[batch_id, 0, cty_int, ctx_int] = gt_bbox[j, 5]
                brake_target[batch_id, 0, cty_int, ctx_int] = gt_bbox[j, 6].long()
                 
                offset_target[batch_id, 0, cty_int, ctx_int] = ctx - ctx_int
                offset_target[batch_id, 1, cty_int, ctx_int] = cty - cty_int
                wh_offset_target_weight[batch_id, :, cty_int, ctx_int] = 1

        avg_factor = max(1, center_heatmap_target.eq(1).sum())
        target_result = dict(
            center_heatmap_target=center_heatmap_target,
            wh_target=wh_target,
            yaw_class_target=yaw_class_target.squeeze(1),
            yaw_res_target=yaw_res_target,
            offset_target=offset_target,
            velocity_target=velocity_target,
            brake_target=brake_target.squeeze(1),
            wh_offset_target_weight=wh_offset_target_weight)
        return target_result, avg_factor

    def get_bboxes(self,
                   center_heatmap_preds,
                   wh_preds,
                   offset_preds,
                   yaw_class_preds,
                   yaw_res_preds,
                   velocity_preds, 
                   brake_preds,
                   rescale=True,
                   with_nms=False):
        """Transform network output for a batch into bbox predictions.

        Args:
            center_heatmap_preds (list[Tensor]): center predict heatmaps for
                all levels with shape (B, num_classes, H, W).
            wh_preds (list[Tensor]): wh predicts for all levels with
                shape (B, 2, H, W).
            offset_preds (list[Tensor]): offset predicts for all levels
                with shape (B, 2, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: True.
            with_nms (bool): If True, do nms before return boxes.
                Default: False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        assert len(center_heatmap_preds) == len(wh_preds) == len(offset_preds) == 1

        batch_det_bboxes, batch_labels = self.decode_heatmap(
            center_heatmap_preds[0],
            wh_preds[0],
            offset_preds[0],
            yaw_class_preds[0],
            yaw_res_preds[0],
            velocity_preds[0], 
            brake_preds[0],
            k=self.train_cfg.top_k_center_keypoints,
            kernel=self.train_cfg.center_net_max_pooling_kernel)

        if with_nms:
            det_results = []
            for (det_bboxes, det_labels) in zip(batch_det_bboxes,
                                                batch_labels):
                det_bbox, det_label = self._bboxes_nms(det_bboxes, det_labels,
                                                       self.test_cfg)
                det_results.append(tuple([det_bbox, det_label]))
        else:
            det_results = [
                tuple(bs) for bs in zip(batch_det_bboxes, batch_labels)
            ]
        return det_results

    def decode_heatmap(self,
                       center_heatmap_pred,
                       wh_pred,
                       offset_pred,
                       yaw_class_pred,
                       yaw_res_pred,
                       velocity_pred,
                       brake_pred,
                       k=100,
                       kernel=3):
        """Transform outputs into detections raw bbox prediction.

        Args:
            center_heatmap_pred (Tensor): center predict heatmap,
               shape (B, num_classes, H, W).
            wh_pred (Tensor): wh predict, shape (B, 2, H, W).
            offset_pred (Tensor): offset predict, shape (B, 2, H, W).
            img_shape (list[int]): image shape in [h, w] format.
            k (int): Get top k center keypoints from heatmap. Default 100.
            kernel (int): Max pooling kernel for extract local maximum pixels.
               Default 3.

        Returns:
            tuple[torch.Tensor]: Decoded output of CenterNetHead, containing
               the following Tensors:

              - batch_bboxes (Tensor): Coords of each box with shape (B, k, 5)
              - batch_topk_labels (Tensor): Categories of each box with \
                  shape (B, k)
        """
        center_heatmap_pred = get_local_maximum(
            center_heatmap_pred, kernel=kernel)

        *batch_dets, topk_ys, topk_xs = get_topk_from_heatmap(
            center_heatmap_pred, k=k)
        batch_scores, batch_index, batch_topk_labels = batch_dets

        wh = transpose_and_gather_feat(wh_pred, batch_index)
        offset = transpose_and_gather_feat(offset_pred, batch_index)
        yaw_class = transpose_and_gather_feat(yaw_class_pred, batch_index)
        yaw_res = transpose_and_gather_feat(yaw_res_pred, batch_index)
        velocity = transpose_and_gather_feat(velocity_pred, batch_index)
        brake = transpose_and_gather_feat(brake_pred, batch_index)
        brake = torch.argmax(brake, -1)
        velocity = velocity[..., 0]

        # convert class + res to yaw
        yaw_class = torch.argmax(yaw_class, -1)
        yaw = self.class2angle(yaw_class, yaw_res.squeeze(2))
        # speed
        
        topk_xs = topk_xs + offset[..., 0]
        topk_ys = topk_ys + offset[..., 1]

        ratio = 4.

        batch_bboxes = torch.stack([topk_xs, topk_ys, wh[..., 0], wh[..., 1], yaw, velocity, brake], dim=2)
        batch_bboxes = torch.cat((batch_bboxes, batch_scores[..., None]),
                                 dim=-1)
        batch_bboxes[:, :, :4] *= ratio

        return batch_bboxes, batch_topk_labels

    def _bboxes_nms(self, bboxes, labels, cfg):
        if labels.numel() == 0:
            return bboxes, labels

        out_bboxes, keep = batched_nms(bboxes[:, :4].contiguous(),
                                       bboxes[:, -1].contiguous(), labels,
                                       cfg.nms_cfg)
        out_labels = labels[keep]

        if len(out_bboxes) > 0:
            idx = torch.argsort(out_bboxes[:, -1], descending=True)
            idx = idx[:cfg.max_per_img]
            out_bboxes = out_bboxes[idx]
            out_labels = out_labels[idx]

        return out_bboxes, out_labels


class PIDController(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D

        self._window = deque([0 for _ in range(n)], maxlen=n)

    def step(self, error):
        self._window.append(error)

        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = (self._window[-1] - self._window[-2])
        else:
            integral = 0.0
            derivative = 0.0

        return self._K_P * error + self._K_I * integral + self._K_D * derivative


class LidarCenterNet(nn.Module):
    """
    Encoder network for LiDAR input list
    Args:
        in_channels: input channels
    """

    def __init__(self, config, device, backbone, image_architecture='resnet34', lidar_architecture='resnet18', use_velocity=True):
        super().__init__()
        self.device = device
        self.config = config
        self.pred_len = config.pred_len
        self.use_target_point_image = config.use_target_point_image
        self.gru_concat_target_point = config.gru_concat_target_point
        self.use_point_pillars = config.use_point_pillars

        if(self.use_point_pillars == True):
            self.point_pillar_net = PointPillarNet(config.num_input, config.num_features,
                                                   min_x = config.min_x, max_x = config.max_x,
                                                   min_y = config.min_y, max_y = config.max_y,
                                                   pixels_per_meter = int(config.pixels_per_meter),
                                                  )

        self.backbone = backbone


        if(backbone == 'transFuser'):
            self._model = TransfuserBackbone(config, image_architecture, lidar_architecture, use_velocity=use_velocity).to(self.device)
        elif(backbone == 'late_fusion'):
            self._model = LateFusionBackbone(config, image_architecture, lidar_architecture, use_velocity=use_velocity).to(self.device)
        elif(backbone == 'geometric_fusion'):
            self._model = GeometricFusionBackbone(config, image_architecture, lidar_architecture, use_velocity=use_velocity).to(self.device)
        elif (backbone == 'latentTF'):
            self._model = latentTFBackbone(config, image_architecture, lidar_architecture, use_velocity=use_velocity).to(self.device)
        else:
            raise("The chosen vision backbone does not exist. The options are: transFuser, late_fusion, geometric_fusion, latentTF")

        if config.multitask:
            self.seg_decoder   = SegDecoder(self.config,   self.config.perception_output_features).to(self.device)
            self.depth_decoder = DepthDecoder(self.config, self.config.perception_output_features).to(self.device)

        channel = config.channel

        self.pred_bev = nn.Sequential(
                            nn.Conv2d(channel, channel, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=True),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(channel, 3, kernel_size=(1, 1), stride=1, padding=0, bias=True)
        ).to(self.device)

        # prediction heads
        self.head = LidarCenterNetHead(channel, channel, 1, train_cfg=config).to(self.device)
        self.i = 0

        # waypoints prediction
        self.join = nn.Sequential(
                            nn.Linear(512, 256),
                            nn.ReLU(inplace=True),
                            nn.Linear(256, 128),
                            nn.ReLU(inplace=True),
                            nn.Linear(128, 64),
                            nn.ReLU(inplace=True),
                        ).to(self.device)

        self.decoder = nn.GRUCell(input_size=4 if self.gru_concat_target_point else 2, # 2 represents x,y coordinate
                                  hidden_size=self.config.gru_hidden_size).to(self.device)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.output = nn.Linear(self.config.gru_hidden_size, 3).to(self.device)

        # pid controller
        self.turn_controller = PIDController(K_P=config.turn_KP, K_I=config.turn_KI, K_D=config.turn_KD, n=config.turn_n)
        self.speed_controller = PIDController(K_P=config.speed_KP, K_I=config.speed_KI, K_D=config.speed_KD, n=config.speed_n)

    def forward_gru(self, z, target_point):
        z = self.join(z)
    
        output_wp = list()
        
        # initial input variable to GRU
        x = torch.zeros(size=(z.shape[0], 2), dtype=z.dtype).to(z.device)

        target_point = target_point.clone()
        target_point[:, 1] *= -1
        
        # autoregressive generation of output waypoints
        for _ in range(self.pred_len):
            if self.gru_concat_target_point:
                x_in = torch.cat([x, target_point], dim=1)
            else:
                x_in = x
            
            z = self.decoder(x_in, z)
            dx = self.output(z)
            
            x = dx[:,:2] + x
            
            output_wp.append(x[:,:2])
            
        pred_wp = torch.stack(output_wp, dim=1)

        # pred the wapoints in the vehicle coordinate and we convert it to lidar coordinate here because the GT waypoints is in lidar coordinate
        pred_wp[:, :, 0] = pred_wp[:, :, 0] - self.config.lidar_pos[0]
            
        pred_brake = None
        steer = None
        throttle = None
        brake = None

        return pred_wp, pred_brake, steer, throttle, brake

    def control_pid(self, waypoints, velocity, is_stuck):
        ''' Predicts vehicle control with a PID controller.
        Args:
            waypoints (tensor): output of self.plan()
            velocity (tensor): speedometer input
        '''
        assert(waypoints.size(0)==1)
        waypoints = waypoints[0].data.cpu().numpy()
        # when training we transform the waypoints to lidar coordinate, so we need to change is back when control
        waypoints[:, 0] += self.config.lidar_pos[0]

        speed = velocity[0].data.cpu().numpy()

        desired_speed = np.linalg.norm(waypoints[0] - waypoints[1]) * 2.0

        if is_stuck:
            desired_speed = np.array(self.config.default_speed) # default speed of 14.4 km/h

        brake = ((desired_speed < self.config.brake_speed) or ((speed / desired_speed) > self.config.brake_ratio))

        delta = np.clip(desired_speed - speed, 0.0, self.config.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.config.clip_throttle)
        throttle = throttle if not brake else 0.0
        aim = (waypoints[1] + waypoints[0]) / 2.0
        angle = np.degrees(np.arctan2(aim[1], aim[0])) / 90.0
        if (speed < 0.01):
            angle = 0.0  # When we don't move we don't want the angle error to accumulate in the integral
        if brake:
            angle = 0.0
        
        steer = self.turn_controller.step(angle)

        steer = np.clip(steer, -1.0, 1.0) #Valid steering values are in [-1,1]

        return steer, throttle, brake
    
    def forward_ego(self, rgb, lidar_bev, target_point, target_point_image, ego_vel, bev_points=None, cam_points=None, save_path=None, expert_waypoints=None,
                    stuck_detector=0, forced_move=False, num_points=None, rgb_back=None, debug=False):
        
        if(self.use_point_pillars == True):
            lidar_bev = self.point_pillar_net(lidar_bev, num_points)
            lidar_bev = torch.rot90(lidar_bev, -1, dims=(2, 3)) #For consitency this is also done in voxelization

        if self.use_target_point_image:
            lidar_bev = torch.cat((lidar_bev, target_point_image), dim=1)

        if (self.backbone == 'transFuser'):
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel)
        elif (self.backbone == 'late_fusion'):
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel)
        elif (self.backbone == 'geometric_fusion'):
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel, bev_points, cam_points)
        elif (self.backbone == 'latentTF'):
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel)
        else:
            raise ("The chosen vision backbone does not exist. The options are: transFuser, late_fusion, geometric_fusion, latentTF")

        pred_wp, _, _, _, _ = self.forward_gru(fused_features, target_point)

        preds = self.head([features[0]])
        results = self.head.get_bboxes(preds[0], preds[1], preds[2], preds[3], preds[4], preds[5], preds[6])
        bboxes, _ = results[0]

        # filter bbox based on the confidence of the prediction
        bboxes = bboxes[bboxes[:, -1] > self.config.bb_confidence_threshold]
        rotated_bboxes = []
        for bbox in bboxes.detach().cpu().numpy():
            bbox = self.get_bbox_local_metric(bbox)
            rotated_bboxes.append(bbox)

        self.i += 1
        if debug and self.i % 2 == 0 and not (save_path is None):
            pred_bev = self.pred_bev(features[0])
            pred_bev = F.interpolate(pred_bev, (self.config.bev_resolution_height, self.config.bev_resolution_width), mode='bilinear', align_corners=True)
            pred_semantic = self.seg_decoder(image_features_grid)
            pred_depth = self.depth_decoder(image_features_grid)

            self.visualize_model_io(save_path, self.i, self.config, rgb, lidar_bev, target_point,
                            pred_wp, pred_bev, pred_semantic, pred_depth, bboxes, self.device,
                            gt_bboxes=None, expert_waypoints=expert_waypoints, stuck_detector=stuck_detector, forced_move=forced_move)


        return pred_wp, rotated_bboxes

    def forward(self, rgb, lidar_bev, ego_waypoint, target_point, target_point_image, ego_vel, bev, label, depth, semantic, num_points=None, save_path=None, bev_points=None, cam_points=None):
        loss = {}

        if(self.use_point_pillars == True):
            lidar_bev = self.point_pillar_net(lidar_bev, num_points)
            lidar_bev = torch.rot90(lidar_bev, -1, dims=(2, 3)) #For consitency this is also done in voxelization


        if self.use_target_point_image:
            lidar_bev = torch.cat((lidar_bev, target_point_image), dim=1)

        if (self.backbone == 'transFuser'):
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel)
        elif (self.backbone == 'late_fusion'):
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel)
        elif (self.backbone == 'geometric_fusion'):
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel, bev_points, cam_points)
        elif (self.backbone == 'latentTF'):
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel)
        else:
            raise ("The chosen vision backbone does not exist. The options are: transFuser, late_fusion, geometric_fusion, latentTF")


        pred_wp, _, _, _, _ = self.forward_gru(fused_features, target_point)

        # pred topdown view
        pred_bev = self.pred_bev(features[0])
        pred_bev = F.interpolate(pred_bev, (self.config.bev_resolution_height, self.config.bev_resolution_width), mode='bilinear', align_corners=True)

        weight = torch.from_numpy(np.array([1., 1., 3.])).to(dtype=torch.float32, device=pred_bev.device)
        loss_bev = F.cross_entropy(pred_bev, bev, weight=weight).mean()

        loss_wp = torch.mean(torch.abs(pred_wp - ego_waypoint))
        loss.update({
            "loss_wp": loss_wp,
            "loss_bev": loss_bev
        })

        preds = self.head([features[0]])

        gt_labels = torch.zeros_like(label[:, :, 0])
        gt_bboxes_ignore = label.sum(dim=-1) == 0.
        loss_bbox = self.head.loss(preds[0], preds[1], preds[2], preds[3], preds[4], preds[5], preds[6],
                                [label], gt_labels=[gt_labels], gt_bboxes_ignore=[gt_bboxes_ignore], img_metas=None)
        
        loss.update(loss_bbox)

        if self.config.multitask:
            pred_semantic = self.seg_decoder(image_features_grid)
            pred_depth = self.depth_decoder(image_features_grid)
            loss_semantic = self.config.ls_seg * F.cross_entropy(pred_semantic, semantic).mean()
            loss_depth = self.config.ls_depth * F.l1_loss(pred_depth, depth).mean()
            loss.update({
                "loss_depth": loss_depth,
                "loss_semantic": loss_semantic
            })
        else:
            loss.update({
                "loss_depth": torch.zeros_like(loss_wp),
                "loss_semantic": torch.zeros_like(loss_wp)
            })

        self.i += 1
        if ((self.config.debug == True) and (self.i % self.config.train_debug_save_freq == 0) and (save_path != None)):
            with torch.no_grad():
                results = self.head.get_bboxes(preds[0], preds[1], preds[2], preds[3], preds[4], preds[5], preds[6])
                bboxes, _ = results[0]
                bboxes = bboxes[bboxes[:, -1] > self.config.bb_confidence_threshold]
                self.visualize_model_io(save_path, self.i, self.config, rgb, lidar_bev, target_point,
                                   pred_wp, pred_bev, pred_semantic, pred_depth, bboxes, self.device,
                                   gt_bboxes=label, expert_waypoints=ego_waypoint, stuck_detector=0, forced_move=False)

        return loss


    # Converts the coordinate system to x front y right, vehicle center at the origin.
    # Units are converted from pixels to meters
    def get_bbox_local_metric(self, bbox):
        x, y, w, h, yaw, speed, brake, confidence = bbox

        w = w / self.config.bounding_box_divisor / self.config.pixels_per_meter # We multiplied by 2 when collecting the data, and multiplied by 8 when loading the labels.
        h = h / self.config.bounding_box_divisor / self.config.pixels_per_meter # We multiplied by 2 when collecting the data, and multiplied by 8 when loading the labels.

        T = get_lidar_to_bevimage_transform()
        T_inv = np.linalg.inv(T)

        center = np.array([x,y,1.0])

        center_old_coordinate_sys = T_inv @ center

        center_old_coordinate_sys = center_old_coordinate_sys + np.array(self.config.lidar_pos)

        #Convert to standard CARLA right hand coordinate system
        center_old_coordinate_sys[1] =  -center_old_coordinate_sys[1]

        bbox = np.array([[-h, -w, 1],
                         [-h,  w, 1],
                         [ h,  w, 1],
                         [ h, -w, 1],
                         [ 0,  0, 1],
                         [ 0, h * speed * 0.5, 1]])

        R = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                      [np.sin(yaw),  np.cos(yaw), 0],
                      [0,                      0, 1]])

        for point_index in range(bbox.shape[0]):
            bbox[point_index] = R @ bbox[point_index]
            bbox[point_index] = bbox[point_index] + np.array([center_old_coordinate_sys[0], center_old_coordinate_sys[1],0])

        return bbox, brake, confidence

    # this is different
    def get_rotated_bbox(self, bbox):
        x, y, w, h, yaw, speed, brake =  bbox

        bbox = np.array([[h,   w, 1],
                         [h,  -w, 1],
                         [-h, -w, 1],
                         [-h,  w, 1],
                         [0, 0, 1],
                         [-h * speed * 0.5, 0, 1]])
        bbox[:, :2] /= self.config.bounding_box_divisor
        bbox[:, :2] = bbox[:, [1, 0]]

        c, s = np.cos(yaw), np.sin(yaw)
        # use y x because coordinate is changed
        r1_to_world = np.array([[c, -s, x], [s, c, y], [0, 0, 1]])

        bbox = r1_to_world @ bbox.T
        bbox = bbox.T

        return bbox, brake

    def draw_bboxes(self, bboxes, image, color=(255, 255, 255), brake_color=(0, 0, 255)):
        idx = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5]]
        for bbox, brake in bboxes:
            bbox = bbox.astype(np.int32)[:, :2]
            for s, e in idx:
                if brake >= self.config.draw_brake_threshhold:
                    color = brake_color
                else:
                    color = color
                # brake is true while still have high velocity
                cv2.line(image, tuple(bbox[s]), tuple(bbox[e]), color=color, thickness=1)
        return image


    def draw_waypoints(self, label, waypoints, image, color = (255, 255, 255)):
        waypoints = waypoints.detach().cpu().numpy()
        label = label.detach().cpu().numpy()

        for bbox, points in zip(label, waypoints):
            x, y, w, h, yaw, speed, brake =  bbox
            c, s = np.cos(yaw), np.sin(yaw)
            # use y x because coordinate is changed
            r1_to_world = np.array([[c, -s, x], [s, c, y], [0, 0, 1]])

            # convert to image space
            # need to negate y componet as we do for lidar points
            # we directly construct points in the image coordiante
            # for lidar, forward +x, right +y
            #            x
            #            +
            #            |
            #            |
            #            |---------+y
            #
            # for image, ---------> x
            #            |
            #            |
            #            +
            #            y

            points[:, 0] *= -1
            points = points * self.config.pixels_per_meter
            points = points[:, [1, 0]]
            points = np.concatenate((points, np.ones_like(points[:, :1])), axis=-1)

            points = r1_to_world @ points.T
            points = points.T

            points_to_draw = []
            for point in points[:, :2]:
                points_to_draw.append(point.copy())
                point = point.astype(np.int32)
                cv2.circle(image, tuple(point), radius=3, color=color, thickness=3)
        return image


    def draw_target_point(self, target_point, image, color = (255, 255, 255)):
        target_point = target_point.copy()

        target_point[1] += self.config.lidar_pos[0]
        point = target_point * self.config.pixels_per_meter
        point[1] *= -1
        point[1] = self.config.lidar_resolution_width - point[1] #Might be LiDAR height
        point[0] += int(self.config.lidar_resolution_height / 2.0) #Might be LiDAR width
        point = point.astype(np.int32)
        point = np.clip(point, 0, 512)
        cv2.circle(image, tuple(point), radius=5, color=color, thickness=3)
        return image

    def visualize_model_io(self, save_path, step, config, rgb, lidar_bev, target_point,
                        pred_wp, pred_bev, pred_semantic, pred_depth, bboxes, device,
                        gt_bboxes=None, expert_waypoints=None, stuck_detector=0, forced_move=False):
        font = ImageFont.load_default()
        i = 0 # We only visualize the first image if there is a batch of them.
        if config.multitask:
            classes_list = config.classes_list
            converter = np.array(classes_list)

            depth_image = pred_depth[i].detach().cpu().numpy()

            indices = np.argmax(pred_semantic.detach().cpu().numpy(), axis=1)
            semantic_image = converter[indices[i, ...], ...].astype('uint8')

            ds_image = np.stack((depth_image, depth_image, depth_image), axis=2)
            ds_image = (ds_image * 255).astype(np.uint8)
            ds_image = np.concatenate((ds_image, semantic_image), axis=0)
            ds_image = cv2.resize(ds_image, (640, 256))
            ds_image = np.concatenate([ds_image, np.zeros_like(ds_image[:50])], axis=0)

        images = np.concatenate(list(lidar_bev.detach().cpu().numpy()[i][:2]), axis=1)
        images = (images * 255).astype(np.uint8)
        images = np.stack([images, images, images], axis=-1)
        images = np.concatenate([images, np.zeros_like(images[:50])], axis=0)

        # draw bbox GT
        if (not (gt_bboxes is None)):
            rotated_bboxes_gt = []
            for bbox in gt_bboxes.detach().cpu().numpy()[i]:
                bbox = self.get_rotated_bbox(bbox)
                rotated_bboxes_gt.append(bbox)
            images = self.draw_bboxes(rotated_bboxes_gt, images, color=(0, 255, 0), brake_color=(0, 255, 128))

        rotated_bboxes = []
        for bbox in bboxes.detach().cpu().numpy():
            bbox = self.get_rotated_bbox(bbox[:7])
            rotated_bboxes.append(bbox)
        images = self.draw_bboxes(rotated_bboxes, images, color=(255, 0, 0), brake_color=(0, 255, 255))

        label = torch.zeros((1, 1, 7)).to(device)
        label[:, -1, 0] = 128.
        label[:, -1, 1] = 256.

        if not expert_waypoints is None:
            images = self.draw_waypoints(label[0], expert_waypoints[i:i+1], images, color=(0, 0, 255))

        images = self.draw_waypoints(label[0], deepcopy(pred_wp[i:i + 1, 2:]), images, color=(255, 255, 255)) # Auxliary waypoints in white
        images = self.draw_waypoints(label[0], deepcopy(pred_wp[i:i + 1, :2]), images, color=(255, 0, 0))     # First two, relevant waypoints in blue

        # draw target points
        images = self.draw_target_point(target_point[i].detach().cpu().numpy(), images)

        # stuck text
        images = Image.fromarray(images)
        draw = ImageDraw.Draw(images)
        draw.text((10, 0), "stuck detector:   %04d" % (stuck_detector), font=font)
        draw.text((10, 30), "forced move:      %s" % (" True" if forced_move else "False"), font=font,
                  fill=(255, 0, 0, 255) if forced_move else (255, 255, 255, 255))
        images = np.array(images)

        bev = pred_bev[i].detach().cpu().numpy().argmax(axis=0) / 2.
        bev = np.stack([bev, bev, bev], axis=2) * 255.
        bev_image = bev.astype(np.uint8)
        bev_image = cv2.resize(bev_image, (256, 256))
        bev_image = np.concatenate([bev_image, np.zeros_like(bev_image[:50])], axis=0)

        if not expert_waypoints is None:
            bev_image = self.draw_waypoints(label[0], expert_waypoints[i:i+1], bev_image, color=(0, 0, 255))

        bev_image = self.draw_waypoints(label[0], deepcopy(pred_wp[i:i + 1, 2:]), bev_image, color=(255, 255, 255))
        bev_image = self.draw_waypoints(label[0], deepcopy(pred_wp[i:i + 1, :2]), bev_image, color=(255, 0, 0))

        bev_image = self.draw_target_point(target_point[i].detach().cpu().numpy(), bev_image)

        if (not (expert_waypoints is None)):
            aim = expert_waypoints[i:i + 1, :2].detach().cpu().numpy()[0].mean(axis=0)
            expert_angle = np.degrees(np.arctan2(aim[1], aim[0] + self.config.lidar_pos[0]))

            aim = pred_wp[i:i + 1, :2].detach().cpu().numpy()[0].mean(axis=0)
            ego_angle = np.degrees(np.arctan2(aim[1], aim[0] + self.config.lidar_pos[0]))
            angle_error = normalize_angle_degree(expert_angle - ego_angle)

            bev_image = Image.fromarray(bev_image)
            draw = ImageDraw.Draw(bev_image)
            draw.text((0, 0), "Angle error:        %.2f°" % (angle_error), font=font)

        bev_image = np.array(bev_image)

        rgb_image = rgb[i].permute(1, 2, 0).detach().cpu().numpy()[:, :, [2, 1, 0]]
        rgb_image = cv2.resize(rgb_image, (1280 + 128, 320 + 32))
        assert (config.multitask)
        images = np.concatenate((bev_image, images, ds_image), axis=1)

        images = np.concatenate((rgb_image, images), axis=0)

        cv2.imwrite(str(save_path + ("/%d.png" % (step // 2))), images)