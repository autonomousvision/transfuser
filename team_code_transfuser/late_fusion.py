import torch
from torch import nn
import timm

class LateFusionBackbone(nn.Module):
    """
    image_architecture: Architecture used in the image branch. ResNet, RegNet and ConvNext are supported
    lidar_architecture: Architecture used in the lidar branch. ResNet, RegNet and ConvNext are supported
    use_velocity: Whether to use the velocity input in the transformer.
    """
    def __init__(self, config, image_architecture='resnet34', lidar_architecture='resnet18', use_velocity=0):
        super().__init__()
        self.config = config

        if (config.use_point_pillars == True):
            in_channels = config.num_features[-1]
        else:
            in_channels = 2 * config.lidar_seq_len

        if (self.config.use_target_point_image == True):
            in_channels += 1

        self.image_encoder = ImageCNN(architecture=image_architecture, normalize=True)
        self.lidar_encoder = LidarEncoder(architecture=lidar_architecture, in_channels=in_channels)

        if (image_architecture.startswith('convnext')):
            self.norm_after_pool_img = nn.LayerNorm((self.config.perception_output_features,), eps=1e-06)
        else:
            self.norm_after_pool_img = nn.Sequential()

        if (lidar_architecture.startswith('convnext')):
            self.norm_after_pool_lidar = nn.LayerNorm((self.config.perception_output_features,), eps=1e-06)
        else:
            self.norm_after_pool_lidar = nn.Sequential()

        # velocity embedding
        self.use_velocity = use_velocity
        if(use_velocity):
            self.vel_emb = nn.Linear(1, self.config.perception_output_features)

        # FPN fusion
        channel = self.config.bev_features_chanels
        self.relu = nn.ReLU(inplace=True)

        if(self.image_encoder.features.num_features != self.config.perception_output_features):
            self.reduce_channels_conv_image = nn.Conv2d(self.image_encoder.features.num_features, self.config.perception_output_features, (1, 1))
        else:
            self.reduce_channels_conv_image = nn.Sequential() # For backwards compatibility. ResNet models were trained without this.
        if(self.image_encoder.features.num_features != self.config.perception_output_features):
            self.reduce_channels_conv_lidar = nn.Conv2d(self.lidar_encoder._model.num_features, self.config.perception_output_features, (1, 1))
        else:
            self.reduce_channels_conv_lidar = nn.Sequential()

        # top down
        self.upsample = nn.Upsample(scale_factor=self.config.bev_upsample_factor, mode='bilinear', align_corners=False)
        self.up_conv5 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv4 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv3 = nn.Conv2d(channel, channel, (1, 1))
        
        # lateral
        self.c5_conv = nn.Conv2d(self.config.perception_output_features, channel, (1, 1))
        
    def top_down(self, c5):

        p5 = self.relu(self.c5_conv(c5))
        p4 = self.relu(self.up_conv5(self.upsample(p5)))
        p3 = self.relu(self.up_conv4(self.upsample(p4)))
        p2 = self.relu(self.up_conv3(self.upsample(p3)))
        
        return p2, p3, p4, p5

    def forward(self, image, lidar, velocity):
        '''
        Image + LiDAR feature fusion
        Args:
            image_list (list): list of input images
            lidar_list (list): list of input LiDAR BEV
            velocity (tensor): input velocity from speedometer
        '''
        if self.image_encoder.normalize:
            image_tensor = normalize_imagenet(image)
        else:
            image_tensor = image

        #Image branch
        output_features_image = self.image_encoder.features(image_tensor)
        output_features_image = self.reduce_channels_conv_image(output_features_image)
        image_features_grid = output_features_image

        image_features = torch.nn.AdaptiveAvgPool2d((1,1))(output_features_image)
        image_features = torch.flatten(image_features, 1)
        image_features = self.norm_after_pool_img(image_features)

        # LiDAR branch
        output_features_lidar = self.lidar_encoder._model(lidar)
        output_features_lidar = self.reduce_channels_conv_lidar(output_features_lidar)
        lidar_features_grid = output_features_lidar
        features = self.top_down(lidar_features_grid)

        lidar_features = torch.nn.AdaptiveAvgPool2d((1,1))(output_features_lidar)
        lidar_features = torch.flatten(lidar_features, 1)
        lidar_features = self.norm_after_pool_lidar(lidar_features)

        # Fusion
        fused_features = torch.add(image_features, lidar_features)

        if(self.use_velocity):
            velocity_embeddings = self.vel_emb(velocity) # (B, C) .unsqueeze(1)
            fused_features = torch.add(fused_features, velocity_embeddings)

        return features, image_features_grid, fused_features

        
class ImageCNN(nn.Module):
    """ 
    Encoder network for image input list.
    Args:
        architecture (string): Vision architecture to be used from the TIMM model library.
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
    """

    def __init__(self, architecture, normalize=True):
        super().__init__()
        self.normalize = normalize
        self.features = timm.create_model(architecture, pretrained=True)
        self.features.fc = nn.Sequential()
        self.features.classifier = nn.Sequential()
        self.features.global_pool = nn.Sequential()
        self.features.head = nn.Sequential()



def normalize_imagenet(x):
    """ Normalize input images according to ImageNet standards.
    Args:
        x (tensor): input images
    """
    x = x.clone()
    x[:, 0] = ((x[:, 0] / 255.0) - 0.485) / 0.229
    x[:, 1] = ((x[:, 1] / 255.0) - 0.456) / 0.224
    x[:, 2] = ((x[:, 2] / 255.0) - 0.406) / 0.225
    return x


class LidarEncoder(nn.Module):
    """
    Encoder network for LiDAR input list
    Args:
        architecture (string): Vision architecture to be used from the TIMM model library.
        num_classes: output feature dimension
        in_channels: input channels
    """

    def __init__(self, architecture, in_channels=2):
        super().__init__()

        self._model = timm.create_model(architecture, pretrained=False, in_chans=in_channels)
        self._model.fc = nn.Sequential()
        self._model.global_pool = nn.Sequential()
        self._model.classifier = nn.Sequential()
        self._model.head = nn.Sequential()

