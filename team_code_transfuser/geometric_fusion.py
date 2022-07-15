import torch
from torch import nn
import torch.nn.functional as F
import timm

class GeometricFusionBackbone(nn.Module):
    """
    image_architecture: Architecture used in the image branch. ResNet, RegNet and ConvNext are supported
    lidar_architecture: Architecture used in the lidar branch. ResNet, RegNet and ConvNext are supported
    use_velocity: Whether to use the velocity input in the transformer.
    """

    def __init__(self, config, image_architecture='resnet34', lidar_architecture='resnet18', use_velocity=0):
        super().__init__()
        self.config = config
        self.use_velocity = use_velocity

        self.avgpool_img = nn.AdaptiveAvgPool2d((self.config.img_vert_anchors, self.config.img_horz_anchors))
        self.avgpool_lidar = nn.AdaptiveAvgPool2d((self.config.lidar_vert_anchors, self.config.lidar_horz_anchors))

        if(config.use_point_pillars == True):
            in_channels = config.num_features[-1]
        else:
            in_channels = 2 * config.lidar_seq_len

        if(self.config.use_target_point_image == True):
            in_channels += 1

        self.image_encoder = ImageCNN(architecture=image_architecture, normalize=True)
        self.lidar_encoder = LidarEncoder(architecture=lidar_architecture, in_channels=in_channels)

        self.image_conv1 = nn.Conv2d(self.image_encoder.features.feature_info[1]['num_chs'], config.n_embd, 1)
        self.image_conv2 = nn.Conv2d(self.image_encoder.features.feature_info[2]['num_chs'], config.n_embd, 1)
        self.image_conv3 = nn.Conv2d(self.image_encoder.features.feature_info[3]['num_chs'], config.n_embd, 1)
        self.image_conv4 = nn.Conv2d(self.image_encoder.features.feature_info[4]['num_chs'], config.n_embd, 1)
        self.image_deconv1 = nn.Conv2d(config.n_embd, self.image_encoder.features.feature_info[1]['num_chs'], 1)
        self.image_deconv2 = nn.Conv2d(config.n_embd, self.image_encoder.features.feature_info[2]['num_chs'], 1)
        self.image_deconv3 = nn.Conv2d(config.n_embd, self.image_encoder.features.feature_info[3]['num_chs'], 1)
        self.image_deconv4 = nn.Conv2d(config.n_embd, self.image_encoder.features.feature_info[4]['num_chs'], 1)
        if(use_velocity):
            self.vel_emb1 = nn.Linear(1, self.image_encoder.features.feature_info[1]['num_chs'])
            self.vel_emb2 = nn.Linear(1, self.image_encoder.features.feature_info[2]['num_chs'])
            self.vel_emb3 = nn.Linear(1, self.image_encoder.features.feature_info[3]['num_chs'])
            self.vel_emb4 = nn.Linear(1, self.image_encoder.features.feature_info[4]['num_chs'])

        self.lidar_conv1 = nn.Conv2d(self.image_encoder.features.feature_info[1]['num_chs'], config.n_embd, 1)
        self.lidar_conv2 = nn.Conv2d(self.image_encoder.features.feature_info[2]['num_chs'], config.n_embd, 1)
        self.lidar_conv3 = nn.Conv2d(self.image_encoder.features.feature_info[3]['num_chs'], config.n_embd, 1)
        self.lidar_conv4 = nn.Conv2d(self.image_encoder.features.feature_info[4]['num_chs'], config.n_embd, 1)
        self.lidar_deconv1 = nn.Conv2d(config.n_embd, self.image_encoder.features.feature_info[1]['num_chs'], 1)
        self.lidar_deconv2 = nn.Conv2d(config.n_embd, self.image_encoder.features.feature_info[2]['num_chs'], 1)
        self.lidar_deconv3 = nn.Conv2d(config.n_embd, self.image_encoder.features.feature_info[3]['num_chs'], 1)
        self.lidar_deconv4 = nn.Conv2d(config.n_embd, self.image_encoder.features.feature_info[4]['num_chs'], 1)

        hid_dim = config.n_embd
        self.image_projection1 = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(True), nn.Linear(hid_dim, hid_dim), nn.ReLU(True), nn.Linear(hid_dim, hid_dim), nn.ReLU(True))
        self.image_projection2 = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(True), nn.Linear(hid_dim, hid_dim), nn.ReLU(True), nn.Linear(hid_dim, hid_dim), nn.ReLU(True))
        self.image_projection3 = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(True), nn.Linear(hid_dim, hid_dim), nn.ReLU(True), nn.Linear(hid_dim, hid_dim), nn.ReLU(True))
        self.image_projection4 = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(True), nn.Linear(hid_dim, hid_dim), nn.ReLU(True), nn.Linear(hid_dim, hid_dim), nn.ReLU(True))
        self.lidar_projection1 = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(True), nn.Linear(hid_dim, hid_dim), nn.ReLU(True), nn.Linear(hid_dim, hid_dim), nn.ReLU(True))
        self.lidar_projection2 = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(True), nn.Linear(hid_dim, hid_dim), nn.ReLU(True), nn.Linear(hid_dim, hid_dim), nn.ReLU(True))
        self.lidar_projection3 = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(True), nn.Linear(hid_dim, hid_dim), nn.ReLU(True), nn.Linear(hid_dim, hid_dim), nn.ReLU(True))
        self.lidar_projection4 = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(True), nn.Linear(hid_dim, hid_dim), nn.ReLU(True), nn.Linear(hid_dim, hid_dim), nn.ReLU(True))

        if(self.image_encoder.features.feature_info[4]['num_chs'] != self.config.perception_output_features):
            self.change_channel_conv_image = nn.Conv2d(self.image_encoder.features.feature_info[4]['num_chs'], self.config.perception_output_features, (1, 1))
            self.change_channel_conv_lidar = nn.Conv2d(self.image_encoder.features.feature_info[4]['num_chs'], self.config.perception_output_features, (1, 1))
        else:
            self.change_channel_conv_image = nn.Sequential()
            self.change_channel_conv_lidar = nn.Sequential()

        # FPN fusion
        channel = self.config.bev_features_chanels
        self.relu = nn.ReLU(inplace=True)
        # top down
        self.upsample = nn.Upsample(scale_factor=self.config.bev_upsample_factor, mode='bilinear', align_corners=False)
        self.up_conv5 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv4 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv3 = nn.Conv2d(channel, channel, (1, 1))
        
        # lateral
        self.c5_conv = nn.Conv2d(self.config.perception_output_features, channel, (1, 1))
        
    def top_down(self, x):

        p5 = self.relu(self.c5_conv(x))
        p4 = self.relu(self.up_conv5(self.upsample(p5)))
        p3 = self.relu(self.up_conv4(self.upsample(p4)))
        p2 = self.relu(self.up_conv3(self.upsample(p3)))
        
        return p2, p3, p4, p5

    def forward(self, image, lidar, velocity, bev_points, img_points):
        '''
        Image + LiDAR feature fusion using transformers
        Args:
            image_list (list): list of input images
            lidar_list (list): list of input LiDAR BEV
            velocity (tensor): input velocity from speedometer
            bev_points (tensor): projected image pixels onto the BEV grid
            cam_points (tensor): projected LiDAR point cloud onto the image space
        '''

        if self.image_encoder.normalize:
            image_tensor = normalize_imagenet(image)
        else:
            image_tensor = image
        lidar_tensor = lidar

        bz = lidar_tensor.shape[0]

        image_features = self.image_encoder.features.conv1(image_tensor)
        image_features = self.image_encoder.features.bn1(image_features)
        image_features = self.image_encoder.features.act1(image_features)
        image_features = self.image_encoder.features.maxpool(image_features)
        lidar_features = self.lidar_encoder._model.conv1(lidar_tensor)
        lidar_features = self.lidar_encoder._model.bn1(lidar_features)
        lidar_features = self.lidar_encoder._model.act1(lidar_features)
        lidar_features = self.lidar_encoder._model.maxpool(lidar_features)

        image_features = self.image_encoder.features.layer1(image_features)
        lidar_features = self.lidar_encoder._model.layer1(lidar_features)
        if self.config.n_scale >= 4:
            # fusion at (B, 64, 64, 64)
            image_embd_layer1 = self.image_conv1(image_features)
            image_embd_layer1 = self.avgpool_img(image_embd_layer1)
            lidar_embd_layer1 = self.lidar_conv1(lidar_features)
            lidar_embd_layer1 = self.avgpool_lidar(lidar_embd_layer1)

            curr_h_image, curr_w_image = image_embd_layer1.shape[-2:]
            curr_h_lidar, curr_w_lidar = lidar_embd_layer1.shape[-2:]

            # project image features to bev
            bev_points_layer1 = bev_points.view(bz*curr_h_lidar*curr_w_lidar*5, 2)
            bev_encoding_layer1 = image_embd_layer1.permute(0,2,3,1).contiguous()[:,bev_points_layer1[:,1],bev_points_layer1[:,0]].view(bz, bz, curr_h_lidar, curr_w_lidar, 5, -1)
            bev_encoding_layer1 = torch.diagonal(bev_encoding_layer1, 0).permute(4,3,0,1,2).contiguous()
            bev_encoding_layer1 = torch.sum(bev_encoding_layer1, -1)
            bev_encoding_layer1 = self.image_projection1(bev_encoding_layer1.permute(0,2,3,1)).permute(0,3,1,2).contiguous()
            lidar_features_layer1 = F.interpolate(bev_encoding_layer1, scale_factor=8, mode='bilinear', align_corners=False)
            lidar_features_layer1 = self.lidar_deconv1(lidar_features_layer1)
            lidar_features = lidar_features + lidar_features_layer1
            if self.use_velocity:
                vel_embedding1 = self.vel_emb1(velocity).unsqueeze(-1).unsqueeze(-1)
                lidar_features = lidar_features + vel_embedding1

            # project bev features to image
            img_points_layer1 = img_points.view(bz*curr_h_image*curr_w_image*5, 2)
            img_encoding_layer1 = lidar_embd_layer1.permute(0,2,3,1).contiguous()[:,img_points_layer1[:,1],img_points_layer1[:,0]].view(bz, bz, curr_h_image, curr_w_image, 5, -1)
            img_encoding_layer1 = torch.diagonal(img_encoding_layer1, 0).permute(4,3,0,1,2).contiguous()
            img_encoding_layer1 = torch.sum(img_encoding_layer1, -1)
            img_encoding_layer1 = self.lidar_projection1(img_encoding_layer1.permute(0,2,3,1)).permute(0,3,1,2).contiguous()
            image_features_layer1 = F.interpolate(img_encoding_layer1, scale_factor=8, mode='bilinear', align_corners=False)
            image_features_layer1 = self.image_deconv1(image_features_layer1)
            image_features = image_features + image_features_layer1

            if self.use_velocity:
                image_features = image_features + vel_embedding1

        image_features = self.image_encoder.features.layer2(image_features)
        lidar_features = self.lidar_encoder._model.layer2(lidar_features)
        if self.config.n_scale >= 3:
            # fusion at (B, 128, 32, 32)
            image_embd_layer2 = self.image_conv2(image_features)
            image_embd_layer2 = self.avgpool_img(image_embd_layer2)
            lidar_embd_layer2 = self.lidar_conv2(lidar_features)
            lidar_embd_layer2 = self.avgpool_lidar(lidar_embd_layer2)

            curr_h_image, curr_w_image = image_embd_layer2.shape[-2:]
            curr_h_lidar, curr_w_lidar = lidar_embd_layer2.shape[-2:]
            
            # project image features to bev
            bev_points_layer2 = bev_points.view(bz*curr_h_lidar*curr_w_lidar*5, 2)
            bev_encoding_layer2 = image_embd_layer2.permute(0,2,3,1).contiguous()[:,bev_points_layer2[:,1],bev_points_layer2[:,0]].view(bz, bz, curr_h_lidar, curr_w_lidar, 5, -1)
            bev_encoding_layer2 = torch.diagonal(bev_encoding_layer2, 0).permute(4,3,0,1,2).contiguous()
            bev_encoding_layer2 = torch.sum(bev_encoding_layer2, -1)
            bev_encoding_layer2 = self.image_projection2(bev_encoding_layer2.permute(0,2,3,1)).permute(0,3,1,2).contiguous()
            lidar_features_layer2 = F.interpolate(bev_encoding_layer2, scale_factor=4, mode='bilinear', align_corners=False)
            lidar_features_layer2 = self.lidar_deconv2(lidar_features_layer2)
            lidar_features = lidar_features + lidar_features_layer2

            if self.use_velocity:
                vel_embedding2 = self.vel_emb2(velocity).unsqueeze(-1).unsqueeze(-1)
                lidar_features = lidar_features + vel_embedding2

            # project bev features to image
            img_points_layer2 = img_points.view(bz*curr_h_image*curr_w_image*5, 2)
            img_encoding_layer2 = lidar_embd_layer2.permute(0,2,3,1).contiguous()[:,img_points_layer2[:,1],img_points_layer2[:,0]].view(bz, bz, curr_h_image, curr_w_image, 5, -1)
            img_encoding_layer2 = torch.diagonal(img_encoding_layer2, 0).permute(4,3,0,1,2).contiguous()
            img_encoding_layer2 = torch.sum(img_encoding_layer2, -1)
            img_encoding_layer2 = self.lidar_projection2(img_encoding_layer2.permute(0,2,3,1)).permute(0,3,1,2).contiguous()
            image_features_layer2 = F.interpolate(img_encoding_layer2, scale_factor=4, mode='bilinear', align_corners=False)
            image_features_layer2 = self.image_deconv2(image_features_layer2)
            image_features = image_features + image_features_layer2

            if self.use_velocity:
                image_features = image_features + vel_embedding2

        image_features = self.image_encoder.features.layer3(image_features)
        lidar_features = self.lidar_encoder._model.layer3(lidar_features)
        if self.config.n_scale >= 2:
            # fusion at (B, 256, 16, 16)
            image_embd_layer3 = self.image_conv3(image_features)
            image_embd_layer3 = self.avgpool_img(image_embd_layer3)
            lidar_embd_layer3 = self.lidar_conv3(lidar_features)
            lidar_embd_layer3 = self.avgpool_lidar(lidar_embd_layer3)

            curr_h_image, curr_w_image = image_embd_layer3.shape[-2:]
            curr_h_lidar, curr_w_lidar = lidar_embd_layer3.shape[-2:]
            
            # project image features to bev
            bev_points_layer3 = bev_points.view(bz*curr_h_lidar*curr_w_lidar*5, 2)
            bev_encoding_layer3 = image_embd_layer3.permute(0,2,3,1).contiguous()[:,bev_points_layer3[:,1],bev_points_layer3[:,0]].view(bz, bz, curr_h_lidar, curr_w_lidar, 5, -1)
            bev_encoding_layer3 = torch.diagonal(bev_encoding_layer3, 0).permute(4,3,0,1,2).contiguous()
            bev_encoding_layer3 = torch.sum(bev_encoding_layer3, -1)
            bev_encoding_layer3 = self.image_projection3(bev_encoding_layer3.permute(0,2,3,1)).permute(0,3,1,2).contiguous()
            lidar_features_layer3 = F.interpolate(bev_encoding_layer3, scale_factor=2, mode='bilinear', align_corners=False)
            lidar_features_layer3 = self.lidar_deconv3(lidar_features_layer3)
            lidar_features = lidar_features + lidar_features_layer3

            if self.use_velocity:
                vel_embedding3 = self.vel_emb3(velocity).unsqueeze(-1).unsqueeze(-1)
                lidar_features = lidar_features + vel_embedding3

            # project bev features to image
            img_points_layer3 = img_points.view(bz*curr_h_image*curr_w_image*5, 2)
            img_encoding_layer3 = lidar_embd_layer3.permute(0,2,3,1).contiguous()[:,img_points_layer3[:,1],img_points_layer3[:,0]].view(bz, bz, curr_h_image, curr_w_image, 5, -1)
            img_encoding_layer3 = torch.diagonal(img_encoding_layer3, 0).permute(4,3,0,1,2).contiguous()
            img_encoding_layer3 = torch.sum(img_encoding_layer3, -1)
            img_encoding_layer3 = self.lidar_projection3(img_encoding_layer3.permute(0,2,3,1)).permute(0,3,1,2).contiguous()
            image_features_layer3 = F.interpolate(img_encoding_layer3, scale_factor=2, mode='bilinear', align_corners=False)
            image_features_layer3 = self.image_deconv3(image_features_layer3)
            image_features = image_features + image_features_layer3
            if self.use_velocity:
                image_features = image_features + vel_embedding3

        image_features = self.image_encoder.features.layer4(image_features)
        lidar_features = self.lidar_encoder._model.layer4(lidar_features)
        # fusion at (B, 512, 8, 8)
        if self.config.n_scale >= 1:
            # fusion at (B, 512, 8, 8)
            image_embd_layer4 = self.image_conv4(image_features)
            image_embd_layer4 = self.avgpool_img(image_embd_layer4)
            lidar_embd_layer4 = self.lidar_conv4(lidar_features)
            lidar_embd_layer4 = self.avgpool_lidar(lidar_embd_layer4)

            curr_h_image, curr_w_image = image_embd_layer4.shape[-2:]
            curr_h_lidar, curr_w_lidar = lidar_embd_layer4.shape[-2:]
            
            # project image features to bev
            bev_points_layer4 = bev_points.view(bz*curr_h_lidar*curr_w_lidar*5, 2)
            bev_encoding_layer4 = image_embd_layer4.permute(0,2,3,1).contiguous()[:,bev_points_layer4[:,1],bev_points_layer4[:,0]].view(bz, bz, curr_h_lidar, curr_w_lidar, 5, -1)
            bev_encoding_layer4 = torch.diagonal(bev_encoding_layer4, 0).permute(4,3,0,1,2).contiguous()
            bev_encoding_layer4 = torch.sum(bev_encoding_layer4, -1)
            bev_encoding_layer4 = self.image_projection4(bev_encoding_layer4.permute(0,2,3,1)).permute(0,3,1,2).contiguous()
            lidar_features_layer4 = self.lidar_deconv4(bev_encoding_layer4)
            lidar_features = lidar_features + lidar_features_layer4

            if self.use_velocity:
                vel_embedding4 = self.vel_emb4(velocity).unsqueeze(-1).unsqueeze(-1)
                lidar_features = lidar_features + vel_embedding4

            # project bev features to image
            img_points_layer4 = img_points.view(bz*curr_h_image*curr_w_image*5, 2)
            img_encoding_layer4 = lidar_embd_layer3.permute(0,2,3,1).contiguous()[:,img_points_layer4[:,1],img_points_layer4[:,0]].view(bz, bz, curr_h_image, curr_w_image, 5, -1)
            img_encoding_layer4 = torch.diagonal(img_encoding_layer4, 0).permute(4,3,0,1,2).contiguous()
            img_encoding_layer4 = torch.sum(img_encoding_layer4, -1)
            img_encoding_layer4 = self.lidar_projection4(img_encoding_layer4.permute(0,2,3,1)).permute(0,3,1,2).contiguous()
            image_features_layer4 = self.image_deconv4(img_encoding_layer4)
            image_features = image_features + image_features_layer4
            if self.use_velocity:
                image_features = image_features + vel_embedding4

        # Downsamples channels to 512
        image_features = self.change_channel_conv_image(image_features)
        lidar_features = self.change_channel_conv_lidar(lidar_features)

        x4 = lidar_features

        image_features_grid = image_features  # For auxilliary information
        image_features = self.image_encoder.features.global_pool(image_features)
        image_features = torch.flatten(image_features, 1)
        lidar_features = self.lidar_encoder._model.global_pool(lidar_features)
        lidar_features = torch.flatten(lidar_features, 1)
        
        fused_features = image_features + lidar_features

        features = self.top_down(x4)
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
        self.features.fc = None
        # Delete parts of the networks we don't want
        if (architecture.startswith('regnet')): # Rename modules so we can use the same code
            self.features.conv1 = self.features.stem.conv
            self.features.bn1  = self.features.stem.bn
            self.features.act1 = nn.Sequential() #The Relu is part of the batch norm here.
            self.features.maxpool =  nn.Sequential()
            self.features.layer1 =self.features.s1
            self.features.layer2 =self.features.s2
            self.features.layer3 =self.features.s3
            self.features.layer4 =self.features.s4
            self.features.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
            self.features.head = nn.Sequential()

        elif (architecture.startswith('convnext')):
            self.features.conv1 = self.features.stem._modules['0']
            self.features.bn1 = self.features.stem._modules['1']
            self.features.act1 = nn.Sequential()  # Don't see any activatin function after the stem. Need to verify
            self.features.maxpool = nn.Sequential()
            self.features.layer1 = self.features.stages._modules['0']
            self.features.layer2 = self.features.stages._modules['1']
            self.features.layer3 = self.features.stages._modules['2']
            self.features.layer4 = self.features.stages._modules['3']
            self.features.global_pool = self.features.head
            self.features.global_pool.flatten = nn.Sequential()
            self.features.global_pool.fc = nn.Sequential()
            self.features.head = nn.Sequential()
            # ConvNext don't have the 0th entry that res nets use.
            self.features.feature_info.append(self.features.feature_info[3])
            self.features.feature_info[3] = self.features.feature_info[2]
            self.features.feature_info[2] = self.features.feature_info[1]
            self.features.feature_info[1] = self.features.feature_info[0]

            #This layer norm is not pretrained anymore but that shouldn't matter since it is the last layer in the network.
            _tmp = self.features.global_pool.norm
            self.features.global_pool.norm = nn.LayerNorm((512,1,1), _tmp.eps, _tmp.elementwise_affine)


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

        self._model = timm.create_model(architecture, pretrained=False)
        self._model.fc = None

        if (architecture.startswith('regnet')): # Rename modules so we can use the same code
            self._model.conv1 = self._model.stem.conv
            self._model.bn1  = self._model.stem.bn
            self._model.act1 = nn.Sequential()#The Relu is part of the batch norm here
            self._model.maxpool =  nn.Sequential() #This is used in ResNets
            self._model.layer1 = self._model.s1
            self._model.layer2 = self._model.s2
            self._model.layer3 = self._model.s3
            self._model.layer4 = self._model.s4
            self._model.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
            self._model.head = nn.Sequential()


        elif (architecture.startswith('convnext')):
            self._model.conv1 = self._model.stem._modules['0']
            self._model.bn1 = self._model.stem._modules['1']
            self._model.act1 = nn.Sequential()  # ConvNext does not use an activation function after the stem.
            self._model.maxpool = nn.Sequential()
            self._model.layer1 = self._model.stages._modules['0']
            self._model.layer2 = self._model.stages._modules['1']
            self._model.layer3 = self._model.stages._modules['2']
            self._model.layer4 = self._model.stages._modules['3']
            self._model.global_pool = self._model.head
            self._model.global_pool.flatten = nn.Sequential()
            self._model.global_pool.fc = nn.Sequential()
            self._model.head = nn.Sequential()
            _tmp = self._model.global_pool.norm
            self._model.global_pool.norm = nn.LayerNorm((self.config.perception_output_features,1,1), _tmp.eps, _tmp.elementwise_affine)

        _tmp = self._model.conv1
        use_bias = (_tmp.bias != None)
        self._model.conv1 = nn.Conv2d(in_channels, out_channels=_tmp.out_channels, 
            kernel_size=_tmp.kernel_size, stride=_tmp.stride, padding=_tmp.padding, bias=use_bias)
        # Need to delete the old conv_layer to avoid unused parameters
        del _tmp
        del self._model.stem
        torch.cuda.empty_cache()
        if(use_bias):
            self._model.conv1.bias = _tmp.bias
