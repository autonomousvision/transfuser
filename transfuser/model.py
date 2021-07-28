import math
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


class SelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, block_exp, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),
            nn.ReLU(True), # changed from GELU
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        B, T, C = x.size()

        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, n_embd, n_head, block_exp, n_layer, 
                    vert_anchors, horz_anchors, seq_len, 
                    embd_pdrop, attn_pdrop, resid_pdrop, config):
        super().__init__()
        self.n_embd = n_embd
        self.seq_len = seq_len
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors
        self.config = config

        # positional embedding parameter (learnable), image + lidar
        self.pos_emb = nn.Parameter(torch.zeros(1, (self.config.n_views + 1) * seq_len * vert_anchors * horz_anchors, n_embd))
        
        # velocity embedding
        self.vel_emb = nn.Linear(1, n_embd)
        self.drop = nn.Dropout(embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, 
                        block_exp, attn_pdrop, resid_pdrop)
                        for layer in range(n_layer)])
        
        # decoder head
        self.ln_f = nn.LayerNorm(n_embd)

        self.block_size = seq_len
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.BatchNorm2d)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # create the pytorch optimizer object
        param_dict = {pn: p for pn, p in self.named_parameters()}
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        return optim_groups

    def forward(self, image_tensor, lidar_tensor, velocity):
        """
        Args:
            image_tensor (tensor): B*4*seq_len, C, H, W
            lidar_tensor (tensor): B*seq_len, C, H, W
            velocity (tensor): ego-velocity
        """
        
        bz = lidar_tensor.shape[0] // self.seq_len
        h, w = lidar_tensor.shape[2:4]
        
        # forward the image model for token embeddings
        image_tensor = image_tensor.view(bz, self.config.n_views * self.seq_len, -1, h, w)
        lidar_tensor = lidar_tensor.view(bz, self.seq_len, -1, h, w)

        # pad token embeddings along number of tokens dimension
        token_embeddings = torch.cat([image_tensor, lidar_tensor], dim=1).permute(0,1,3,4,2).contiguous()
        token_embeddings = token_embeddings.view(bz, -1, self.n_embd) # (B, an * T, C)

        # project velocity to n_embed
        velocity_embeddings = self.vel_emb(velocity.unsqueeze(1)) # (B, C)

        # add (learnable) positional embedding and velocity embedding for all tokens
        x = self.drop(self.pos_emb + token_embeddings + velocity_embeddings.unsqueeze(1)) # (B, an * T, C)
        # x = self.drop(token_embeddings + velocity_embeddings.unsqueeze(1)) # (B, an * T, C)
        x = self.blocks(x) # (B, an * T, C)
        x = self.ln_f(x) # (B, an * T, C)
        x = x.view(bz, (self.config.n_views + 1) * self.seq_len, self.vert_anchors, self.horz_anchors, self.n_embd)
        x = x.permute(0,1,4,2,3).contiguous() # same as token_embeddings

        image_tensor_out = x[:, :self.config.n_views*self.seq_len, :, :, :].contiguous().view(bz * self.config.n_views * self.seq_len, -1, h, w)
        lidar_tensor_out = x[:, self.config.n_views*self.seq_len:, :, :, :].contiguous().view(bz * self.seq_len, -1, h, w)
        
        return image_tensor_out, lidar_tensor_out


class Encoder(nn.Module):
    """
    Multi-scale Fusion Transformer for image + LiDAR feature fusion
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.avgpool = nn.AdaptiveAvgPool2d((self.config.vert_anchors, self.config.horz_anchors))
        
        self.image_encoder = ImageCNN(512, normalize=True)
        self.lidar_encoder = LidarEncoder(num_classes=512, in_channels=2)

        self.transformer1 = GPT(n_embd=64,
                            n_head=config.n_head, 
                            block_exp=config.block_exp, 
                            n_layer=config.n_layer, 
                            vert_anchors=config.vert_anchors, 
                            horz_anchors=config.horz_anchors, 
                            seq_len=config.seq_len, 
                            embd_pdrop=config.embd_pdrop, 
                            attn_pdrop=config.attn_pdrop, 
                            resid_pdrop=config.resid_pdrop,
                            config=config)
        self.transformer2 = GPT(n_embd=128,
                            n_head=config.n_head, 
                            block_exp=config.block_exp, 
                            n_layer=config.n_layer, 
                            vert_anchors=config.vert_anchors, 
                            horz_anchors=config.horz_anchors, 
                            seq_len=config.seq_len, 
                            embd_pdrop=config.embd_pdrop, 
                            attn_pdrop=config.attn_pdrop, 
                            resid_pdrop=config.resid_pdrop,
                            config=config)
        self.transformer3 = GPT(n_embd=256,
                            n_head=config.n_head, 
                            block_exp=config.block_exp, 
                            n_layer=config.n_layer, 
                            vert_anchors=config.vert_anchors, 
                            horz_anchors=config.horz_anchors, 
                            seq_len=config.seq_len, 
                            embd_pdrop=config.embd_pdrop, 
                            attn_pdrop=config.attn_pdrop, 
                            resid_pdrop=config.resid_pdrop,
                            config=config)
        self.transformer4 = GPT(n_embd=512,
                            n_head=config.n_head, 
                            block_exp=config.block_exp, 
                            n_layer=config.n_layer, 
                            vert_anchors=config.vert_anchors, 
                            horz_anchors=config.horz_anchors, 
                            seq_len=config.seq_len, 
                            embd_pdrop=config.embd_pdrop, 
                            attn_pdrop=config.attn_pdrop, 
                            resid_pdrop=config.resid_pdrop,
                            config=config)

        
    def forward(self, image_list, lidar_list, velocity):
        '''
        Image + LiDAR feature fusion using transformers
        Args:
            image_list (list): list of input images
            lidar_list (list): list of input LiDAR BEV
            velocity (tensor): input velocity from speedometer
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

        image_features = self.image_encoder.features.layer1(image_features)
        lidar_features = self.lidar_encoder._model.layer1(lidar_features)
        # fusion at (B, 64, 64, 64)
        image_embd_layer1 = self.avgpool(image_features)
        lidar_embd_layer1 = self.avgpool(lidar_features)
        image_features_layer1, lidar_features_layer1 = self.transformer1(image_embd_layer1, lidar_embd_layer1, velocity)
        image_features_layer1 = F.interpolate(image_features_layer1, scale_factor=8, mode='bilinear')
        lidar_features_layer1 = F.interpolate(lidar_features_layer1, scale_factor=8, mode='bilinear')
        image_features = image_features + image_features_layer1
        lidar_features = lidar_features + lidar_features_layer1

        image_features = self.image_encoder.features.layer2(image_features)
        lidar_features = self.lidar_encoder._model.layer2(lidar_features)
        # fusion at (B, 128, 32, 32)
        image_embd_layer2 = self.avgpool(image_features)
        lidar_embd_layer2 = self.avgpool(lidar_features)
        image_features_layer2, lidar_features_layer2 = self.transformer2(image_embd_layer2, lidar_embd_layer2, velocity)
        image_features_layer2 = F.interpolate(image_features_layer2, scale_factor=4, mode='bilinear')
        lidar_features_layer2 = F.interpolate(lidar_features_layer2, scale_factor=4, mode='bilinear')
        image_features = image_features + image_features_layer2
        lidar_features = lidar_features + lidar_features_layer2

        image_features = self.image_encoder.features.layer3(image_features)
        lidar_features = self.lidar_encoder._model.layer3(lidar_features)
        # fusion at (B, 256, 16, 16)
        image_embd_layer3 = self.avgpool(image_features)
        lidar_embd_layer3 = self.avgpool(lidar_features)
        image_features_layer3, lidar_features_layer3 = self.transformer3(image_embd_layer3, lidar_embd_layer3, velocity)
        image_features_layer3 = F.interpolate(image_features_layer3, scale_factor=2, mode='bilinear')
        lidar_features_layer3 = F.interpolate(lidar_features_layer3, scale_factor=2, mode='bilinear')
        image_features = image_features + image_features_layer3
        lidar_features = lidar_features + lidar_features_layer3

        image_features = self.image_encoder.features.layer4(image_features)
        lidar_features = self.lidar_encoder._model.layer4(lidar_features)
        # fusion at (B, 512, 8, 8)
        image_embd_layer4 = self.avgpool(image_features)
        lidar_embd_layer4 = self.avgpool(lidar_features)
        image_features_layer4, lidar_features_layer4 = self.transformer4(image_embd_layer4, lidar_embd_layer4, velocity)
        image_features = image_features + image_features_layer4
        lidar_features = lidar_features + lidar_features_layer4

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


class TransFuser(nn.Module):
    '''
    Transformer-based feature fusion followed by GRU-based waypoint prediction network and PID controller
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
        
    def forward(self, image_list, lidar_list, target_point, velocity):
        '''
        Predicts waypoint from geometric feature projections of image + LiDAR input
        Args:
            image_list (list): list of input images
            lidar_list (list): list of input LiDAR BEV
            target_point (tensor): goal location registered to ego-frame
            velocity (tensor): input velocity from speedometer
        '''
        fused_features = self.encoder(image_list, lidar_list, velocity)
        z = self.join(fused_features)

        output_wp = list()

        # initial input variable to GRU
        x = torch.zeros(size=(z.shape[0], 2), dtype=z.dtype).to(self.device)

        # autoregressive generation of output waypoints
        for _ in range(self.pred_len):
            # x_in = torch.cat([x, target_point], dim=1)
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

        desired_speed = np.linalg.norm(waypoints[0] - waypoints[1]) * 2.0
        brake = desired_speed < self.config.brake_speed or (speed / desired_speed) > self.config.brake_ratio

        aim = (waypoints[1] + waypoints[0]) / 2.0
        angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
        if(speed < 0.01):
            angle = np.array(0.0) # When we don't move we don't want the angle error to accumulate in the integral
        steer = self.turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)

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