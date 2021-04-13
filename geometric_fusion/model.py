from collections import deque

import numpy as np
import torch 
from torch import nn
import torch.nn.functional as F
from torchvision import models


class ImageCNN(nn.Module):
    """ 
    Encoder network for image input list.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
    """

    def __init__(self, c_dim, normalize=True):
        super().__init__()
        self.normalize = normalize
        self.features = models.resnet34(pretrained=True)
        self.features.fc = nn.Sequential()

    def forward(self, inputs):
        c = 0
        for x in inputs:
            if self.normalize:
                x = normalize_imagenet(x)
            c += self.features(x)
        return c

def normalize_imagenet(x):
    """ Normalize input images according to ImageNet standards.
    Args:
        x (tensor): input images
    """
    x = x.clone()
    x[:, 0] = (x[:, 0] - 0.485) / 0.229
    x[:, 1] = (x[:, 1] - 0.456) / 0.224
    x[:, 2] = (x[:, 2] - 0.406) / 0.225
    return x


class LidarEncoder(nn.Module):
    """
    Encoder network for LiDAR input list
    Args:
        num_classes: output feature dimension
        in_channels: input channels
    """

    def __init__(self, num_classes=512, in_channels=2):
        super().__init__()

        self._model = models.resnet18()
        self._model.fc = nn.Sequential()
        _tmp = self._model.conv1
        self._model.conv1 = nn.Conv2d(in_channels, out_channels=_tmp.out_channels, 
            kernel_size=_tmp.kernel_size, stride=_tmp.stride, padding=_tmp.padding, bias=_tmp.bias)

    def forward(self, inputs):
        features = 0
        for lidar_data in inputs:
            lidar_feature = self._model(lidar_data)
            features += lidar_feature

        return features


class Encoder(nn.Module):
    """
    Multi-scale image + LiDAR fusion encoder using geometric feature projections
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.avgpool = nn.AdaptiveAvgPool2d((self.config.vert_anchors, self.config.horz_anchors))
        
        self.image_encoder = ImageCNN(512, normalize=True)
        self.lidar_encoder = LidarEncoder(num_classes=512, in_channels=2)

        self.image_conv1 = nn.Conv2d(64, config.n_embd, 1)
        self.image_conv2 = nn.Conv2d(128, config.n_embd, 1)
        self.image_conv3 = nn.Conv2d(256, config.n_embd, 1)
        self.image_conv4 = nn.Conv2d(512, config.n_embd, 1)
        self.image_deconv1 = nn.Conv2d(config.n_embd, 64, 1)
        self.image_deconv2 = nn.Conv2d(config.n_embd, 128, 1)
        self.image_deconv3 = nn.Conv2d(config.n_embd, 256, 1)
        self.image_deconv4 = nn.Conv2d(config.n_embd, 512, 1)

        self.lidar_conv1 = nn.Conv2d(64, config.n_embd, 1)
        self.lidar_conv2 = nn.Conv2d(128, config.n_embd, 1)
        self.lidar_conv3 = nn.Conv2d(256, config.n_embd, 1)
        self.lidar_conv4 = nn.Conv2d(512, config.n_embd, 1)
        self.lidar_deconv1 = nn.Conv2d(config.n_embd, 64, 1)
        self.lidar_deconv2 = nn.Conv2d(config.n_embd, 128, 1)
        self.lidar_deconv3 = nn.Conv2d(config.n_embd, 256, 1)
        self.lidar_deconv4 = nn.Conv2d(config.n_embd, 512, 1)

        self.image_projection1 = nn.Sequential(nn.Linear(512, 512), nn.ReLU(True), nn.Linear(512, 512), nn.ReLU(True), nn.Linear(512, 512), nn.ReLU(True))
        self.image_projection2 = nn.Sequential(nn.Linear(512, 512), nn.ReLU(True), nn.Linear(512, 512), nn.ReLU(True), nn.Linear(512, 512), nn.ReLU(True))
        self.image_projection3 = nn.Sequential(nn.Linear(512, 512), nn.ReLU(True), nn.Linear(512, 512), nn.ReLU(True), nn.Linear(512, 512), nn.ReLU(True))
        self.image_projection4 = nn.Sequential(nn.Linear(512, 512), nn.ReLU(True), nn.Linear(512, 512), nn.ReLU(True), nn.Linear(512, 512), nn.ReLU(True))
        self.lidar_projection1 = nn.Sequential(nn.Linear(512, 512), nn.ReLU(True), nn.Linear(512, 512), nn.ReLU(True), nn.Linear(512, 512), nn.ReLU(True))
        self.lidar_projection2 = nn.Sequential(nn.Linear(512, 512), nn.ReLU(True), nn.Linear(512, 512), nn.ReLU(True), nn.Linear(512, 512), nn.ReLU(True))
        self.lidar_projection3 = nn.Sequential(nn.Linear(512, 512), nn.ReLU(True), nn.Linear(512, 512), nn.ReLU(True), nn.Linear(512, 512), nn.ReLU(True))
        self.lidar_projection4 = nn.Sequential(nn.Linear(512, 512), nn.ReLU(True), nn.Linear(512, 512), nn.ReLU(True), nn.Linear(512, 512), nn.ReLU(True))
        
    def forward(self, image_list, lidar_list, velocity, bev_points, img_points):
        '''
        Image + LiDAR feature fusion using geometric projections
        Args:
            image_list (list): list of input images
            lidar_list (list): list of input LiDAR BEV
            target_point (tensor): goal location registered to ego-frame
            velocity (tensor): input velocity from speedometer
            bev_points (tensor): projected image pixels onto the BEV grid
            cam_points (tensor): projected LiDAR point cloud onto the image space
        '''
        if self.image_encoder.normalize:
            image_list = [normalize_imagenet(image_input) for image_input in image_list]

        bz, _, h, w = lidar_list[0].shape
        img_channel = image_list[0].shape[1]
        lidar_channel = lidar_list[0].shape[1]
        self.config.n_views = len(image_list) // self.config.seq_len

        image_tensor = torch.stack(image_list, dim=1).view(bz * self.config.n_views * self.config.seq_len, img_channel, h, w)
        lidar_tensor = torch.stack(lidar_list, dim=1).view(bz * self.config.seq_len, lidar_channel, h, w)

        image_features = self.image_encoder.features.conv1(image_tensor)
        image_features = self.image_encoder.features.bn1(image_features)
        image_features = self.image_encoder.features.relu(image_features)
        image_features = self.image_encoder.features.maxpool(image_features)
        lidar_features = self.lidar_encoder._model.conv1(lidar_tensor)
        lidar_features = self.lidar_encoder._model.bn1(lidar_features)
        lidar_features = self.lidar_encoder._model.relu(lidar_features)
        lidar_features = self.lidar_encoder._model.maxpool(lidar_features)

        image_features = self.image_encoder.features.layer1(image_features.contiguous())
        lidar_features = self.lidar_encoder._model.layer1(lidar_features.contiguous())
        if self.config.n_scale >= 4:
            # fusion at (B, 64, 64, 64)
            image_embd_layer1 = self.image_conv1(image_features)
            image_embd_layer1 = self.avgpool(image_embd_layer1)
            lidar_embd_layer1 = self.lidar_conv1(lidar_features)
            lidar_embd_layer1 = self.avgpool(lidar_embd_layer1)
            
            curr_h, curr_w = image_embd_layer1.shape[-2:]
            
            # project image features to bev
            bev_points_layer1 = bev_points.view(bz*curr_h*curr_w*5, 2)
            bev_encoding_layer1 = image_embd_layer1.permute(0,2,3,1).contiguous()[:,bev_points_layer1[:,0],bev_points_layer1[:,1]].view(bz, bz, curr_h, curr_w, 5, -1)
            bev_encoding_layer1 = torch.diagonal(bev_encoding_layer1, 0).permute(4,3,0,1,2).contiguous()
            bev_encoding_layer1 = torch.sum(bev_encoding_layer1, -1)
            bev_encoding_layer1 = self.image_projection1(bev_encoding_layer1.permute(0,2,3,1)).permute(0,3,1,2).contiguous()
            lidar_features_layer1 = F.interpolate(bev_encoding_layer1, scale_factor=8, mode='bilinear')
            lidar_features_layer1 = self.lidar_deconv1(lidar_features_layer1)
            lidar_features = lidar_features + lidar_features_layer1

            # project bev features to image
            img_points_layer1 = img_points.view(bz*curr_h*curr_w*5, 2)
            img_encoding_layer1 = lidar_embd_layer1.permute(0,2,3,1).contiguous()[:,img_points_layer1[:,0],img_points_layer1[:,1]].view(bz, bz, curr_h, curr_w, 5, -1)
            img_encoding_layer1 = torch.diagonal(img_encoding_layer1, 0).permute(4,3,0,1,2).contiguous()
            img_encoding_layer1 = torch.sum(img_encoding_layer1, -1)
            img_encoding_layer1 = self.lidar_projection1(img_encoding_layer1.permute(0,2,3,1)).permute(0,3,1,2).contiguous()
            image_features_layer1 = F.interpolate(img_encoding_layer1, scale_factor=8, mode='bilinear')
            image_features_layer1 = self.image_deconv1(image_features_layer1)
            image_features = image_features + image_features_layer1

        image_features = self.image_encoder.features.layer2(image_features.contiguous())
        lidar_features = self.lidar_encoder._model.layer2(lidar_features.contiguous())
        if self.config.n_scale >= 3:
            # fusion at (B, 128, 32, 32)
            image_embd_layer2 = self.image_conv2(image_features)
            image_embd_layer2 = self.avgpool(image_embd_layer2)
            lidar_embd_layer2 = self.lidar_conv2(lidar_features)
            lidar_embd_layer2 = self.avgpool(lidar_embd_layer2)

            curr_h, curr_w = image_embd_layer2.shape[-2:]
            
            # project image features to bev
            bev_points_layer2 = bev_points.view(bz*curr_h*curr_w*5, 2)
            bev_encoding_layer2 = image_embd_layer2.permute(0,2,3,1).contiguous()[:,bev_points_layer2[:,0],bev_points_layer2[:,1]].view(bz, bz, curr_h, curr_w, 5, -1)
            bev_encoding_layer2 = torch.diagonal(bev_encoding_layer2, 0).permute(4,3,0,1,2).contiguous()
            bev_encoding_layer2 = torch.sum(bev_encoding_layer2, -1)
            bev_encoding_layer2 = self.image_projection1(bev_encoding_layer2.permute(0,2,3,1)).permute(0,3,1,2).contiguous()
            lidar_features_layer2 = F.interpolate(bev_encoding_layer2, scale_factor=4, mode='bilinear')
            lidar_features_layer2 = self.lidar_deconv2(lidar_features_layer2)
            lidar_features = lidar_features + lidar_features_layer2

            # project bev features to image
            img_points_layer2 = img_points.view(bz*curr_h*curr_w*5, 2)
            img_encoding_layer2 = lidar_embd_layer2.permute(0,2,3,1).contiguous()[:,img_points_layer2[:,0],img_points_layer2[:,1]].view(bz, bz, curr_h, curr_w, 5, -1)
            img_encoding_layer2 = torch.diagonal(img_encoding_layer2, 0).permute(4,3,0,1,2).contiguous()
            img_encoding_layer2 = torch.sum(img_encoding_layer2, -1)
            img_encoding_layer2 = self.lidar_projection2(img_encoding_layer2.permute(0,2,3,1)).permute(0,3,1,2).contiguous()
            image_features_layer2 = F.interpolate(img_encoding_layer2, scale_factor=4, mode='bilinear')
            image_features_layer2 = self.image_deconv2(image_features_layer2)
            image_features = image_features + image_features_layer2

        image_features = self.image_encoder.features.layer3(image_features.contiguous())
        lidar_features = self.lidar_encoder._model.layer3(lidar_features.contiguous())
        if self.config.n_scale >= 2:
            # fusion at (B, 256, 16, 16)
            image_embd_layer3 = self.image_conv3(image_features)
            image_embd_layer3 = self.avgpool(image_embd_layer3)
            lidar_embd_layer3 = self.lidar_conv3(lidar_features)
            lidar_embd_layer3 = self.avgpool(lidar_embd_layer3)

            curr_h, curr_w = image_embd_layer3.shape[-2:]
            
            # project image features to bev
            bev_points_layer3 = bev_points.view(bz*curr_h*curr_w*5, 2)
            bev_encoding_layer3 = image_embd_layer3.permute(0,2,3,1).contiguous()[:,bev_points_layer3[:,0],bev_points_layer3[:,1]].view(bz, bz, curr_h, curr_w, 5, -1)
            bev_encoding_layer3 = torch.diagonal(bev_encoding_layer3, 0).permute(4,3,0,1,2).contiguous()
            bev_encoding_layer3 = torch.sum(bev_encoding_layer3, -1)
            bev_encoding_layer3 = self.image_projection3(bev_encoding_layer3.permute(0,2,3,1)).permute(0,3,1,2).contiguous()
            lidar_features_layer3 = F.interpolate(bev_encoding_layer3, scale_factor=2, mode='bilinear')
            lidar_features_layer3 = self.lidar_deconv3(lidar_features_layer3)
            lidar_features = lidar_features + lidar_features_layer3

            # project bev features to image
            img_points_layer3 = img_points.view(bz*curr_h*curr_w*5, 2)
            img_encoding_layer3 = lidar_embd_layer3.permute(0,2,3,1).contiguous()[:,img_points_layer3[:,0],img_points_layer3[:,1]].view(bz, bz, curr_h, curr_w, 5, -1)
            img_encoding_layer3 = torch.diagonal(img_encoding_layer3, 0).permute(4,3,0,1,2).contiguous()
            img_encoding_layer3 = torch.sum(img_encoding_layer3, -1)
            img_encoding_layer3 = self.lidar_projection3(img_encoding_layer3.permute(0,2,3,1)).permute(0,3,1,2).contiguous()
            image_features_layer3 = F.interpolate(img_encoding_layer3, scale_factor=2, mode='bilinear')
            image_features_layer3 = self.image_deconv3(image_features_layer3)
            image_features = image_features + image_features_layer3

        image_features = self.image_encoder.features.layer4(image_features.contiguous())
        lidar_features = self.lidar_encoder._model.layer4(lidar_features.contiguous())
        if self.config.n_scale >= 1:
            # fusion at (B, 512, 8, 8)
            image_embd_layer4 = self.image_conv4(image_features)
            image_embd_layer4 = self.avgpool(image_embd_layer4)
            lidar_embd_layer4 = self.lidar_conv4(lidar_features)
            lidar_embd_layer4 = self.avgpool(lidar_embd_layer4)

            curr_h, curr_w = image_embd_layer4.shape[-2:]
            
            # project image features to bev
            bev_points_layer4 = bev_points.view(bz*curr_h*curr_w*5, 2)
            bev_encoding_layer4 = image_embd_layer4.permute(0,2,3,1).contiguous()[:,bev_points_layer4[:,0],bev_points_layer4[:,1]].view(bz, bz, curr_h, curr_w, 5, -1)
            bev_encoding_layer4 = torch.diagonal(bev_encoding_layer4, 0).permute(4,3,0,1,2).contiguous()
            bev_encoding_layer4 = torch.sum(bev_encoding_layer4, -1)
            bev_encoding_layer4 = self.image_projection4(bev_encoding_layer4.permute(0,2,3,1)).permute(0,3,1,2).contiguous()
            lidar_features_layer4 = self.lidar_deconv4(bev_encoding_layer4)
            lidar_features = lidar_features + lidar_features_layer4

            # project bev features to image
            img_points_layer4 = img_points.view(bz*curr_h*curr_w*5, 2)
            img_encoding_layer4 = lidar_embd_layer3.permute(0,2,3,1).contiguous()[:,img_points_layer4[:,0],img_points_layer4[:,1]].view(bz, bz, curr_h, curr_w, 5, -1)
            img_encoding_layer4 = torch.diagonal(img_encoding_layer4, 0).permute(4,3,0,1,2).contiguous()
            img_encoding_layer4 = torch.sum(img_encoding_layer4, -1)
            img_encoding_layer4 = self.lidar_projection4(img_encoding_layer4.permute(0,2,3,1)).permute(0,3,1,2).contiguous()
            image_features_layer4 = self.image_deconv4(img_encoding_layer4)
            image_features = image_features + image_features_layer4

        image_features = self.image_encoder.features.avgpool(image_features)
        image_features = torch.flatten(image_features, 1)
        image_features = image_features.view(bz, self.config.n_views * self.config.seq_len, -1)
        lidar_features = self.lidar_encoder._model.avgpool(lidar_features)
        lidar_features = torch.flatten(lidar_features, 1)
        lidar_features = lidar_features.view(bz, self.config.seq_len, -1)

        fused_features = torch.cat([image_features, lidar_features], dim=1)
        fused_features = torch.sum(fused_features, dim=1)

        return fused_features


class PIDController(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D

        self._window = deque([0 for _ in range(n)], maxlen=n)
        self._max = 0.0
        self._min = 0.0

    def step(self, error):
        self._window.append(error)
        self._max = max(self._max, abs(error))
        self._min = -abs(self._max)

        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = (self._window[-1] - self._window[-2])
        else:
            integral = 0.0
            derivative = 0.0

        return self._K_P * error + self._K_I * integral + self._K_D * derivative


class GeometricFusion(nn.Module):
    '''
    Image + LiDAR feature fusion using geometric projections followed by
    GRU-based waypoint prediction network and PID controller
    '''

    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.config = config
        self.pred_len = config.pred_len

        self.turn_controller = PIDController(K_P=config.turn_KP, K_I=config.turn_KI, K_D=config.turn_KD, n=config.turn_n)
        self.speed_controller = PIDController(K_P=config.speed_KP, K_I=config.speed_KI, K_D=config.speed_KD, n=config.speed_n)

        self.encoder = Encoder(config).to(self.device)

        self.join = nn.Sequential(
                            nn.Linear(512, 256),
                            nn.ReLU(inplace=True),
                            nn.Linear(256, 128),
                            nn.ReLU(inplace=True),
                            nn.Linear(128, 64),
                            nn.ReLU(inplace=True),
                        ).to(self.device)
        self.decoder = nn.GRUCell(input_size=2, hidden_size=64).to(self.device)
        self.output = nn.Linear(64, 2).to(self.device)
        
    def forward(self, image_list, lidar_list, target_point, velocity, bev_points, cam_points):
        '''
        Predicts waypoint from geometric feature projections of image + LiDAR input
        Args:
            image_list (list): list of input images
            lidar_list (list): list of input LiDAR BEV
            target_point (tensor): goal location registered to ego-frame
            velocity (tensor): input velocity from speedometer
            bev_points (tensor): projected image pixels onto the BEV grid
            cam_points (tensor): projected LiDAR point cloud onto the image space
        '''
        fused_features = self.encoder(image_list, lidar_list, velocity, bev_points, cam_points)
        z = self.join(fused_features)

        output_wp = list()

        # initial input variable to GRU
        x = torch.zeros(size=(z.shape[0], 2), dtype=z.dtype).to(self.device)

        # autoregressive generation of output waypoints
        for _ in range(self.pred_len):
            x_in = x + target_point
            z = self.decoder(x_in, z)
            dx = self.output(z)
            x = dx + x
            output_wp.append(x)

        pred_wp = torch.stack(output_wp, dim=1)

        return pred_wp

    def control_pid(self, waypoints, velocity):
        ''' 
        Predicts vehicle control with a PID controller.
        Args:
            waypoints (tensor): predicted waypoints
            velocity (tensor): speedometer input
        '''
        assert(waypoints.size(0)==1)
        waypoints = waypoints[0].data.cpu().numpy()

        # flip y is (forward is negative in our waypoints)
        waypoints[:,1] *= -1
        speed = velocity[0].data.cpu().numpy()

        aim = (waypoints[1] + waypoints[0]) / 2.0
        angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
        steer = self.turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)

        desired_speed = np.linalg.norm(waypoints[0] - waypoints[1]) * 2.0
        brake = desired_speed < self.config.brake_speed or (speed / desired_speed) > self.config.brake_ratio

        delta = np.clip(desired_speed - speed, 0.0, self.config.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.config.max_throttle)
        throttle = throttle if not brake else 0.0

        metadata = {
            'speed': float(speed.astype(np.float64)),
            'steer': float(steer),
            'throttle': float(throttle),
            'brake': float(brake),
            'wp_2': tuple(waypoints[1].astype(np.float64)),
            'wp_1': tuple(waypoints[0].astype(np.float64)),
            'desired_speed': float(desired_speed.astype(np.float64)),
            'angle': float(angle.astype(np.float64)),
            'aim': tuple(aim.astype(np.float64)),
            'delta': float(delta.astype(np.float64)),
        }

        return steer, throttle, brake, metadata