import math
import torch
from torch import nn
import torch.nn.functional as F
import timm


class latentTFBackbone(nn.Module):
    """
    Multi-scale Fusion Transformer for image + pos_embedding feature fusion
    image_architecture: Architecture used in the image branch. ResNet, RegNet and ConvNext are supported
    lidar_architecture: Architecture used in the lidar branch. ResNet, RegNet and ConvNext are supported
    use_velocity: Whether to use the velocity input in the transformer.
    """

    def __init__(self, config, image_architecture='resnet34', lidar_architecture='resnet18', use_velocity=True):
        super().__init__()
        self.config = config

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

        self.transformer1 = GPT(n_embd=self.image_encoder.features.feature_info[1]['num_chs'],
                            n_head=config.n_head,
                            block_exp=config.block_exp,
                            n_layer=config.n_layer,
                            img_vert_anchors=config.img_vert_anchors,
                            img_horz_anchors=config.img_horz_anchors,
                            lidar_vert_anchors=config.lidar_vert_anchors,
                            lidar_horz_anchors=config.lidar_horz_anchors,
                            seq_len=config.seq_len,
                            embd_pdrop=config.embd_pdrop,
                            attn_pdrop=config.attn_pdrop,
                            resid_pdrop=config.resid_pdrop,
                            config=config, use_velocity=use_velocity)

        self.transformer2 = GPT(n_embd=self.image_encoder.features.feature_info[2]['num_chs'],
                            n_head=config.n_head,
                            block_exp=config.block_exp,
                            n_layer=config.n_layer,
                            img_vert_anchors=config.img_vert_anchors,
                            img_horz_anchors=config.img_horz_anchors,
                            lidar_vert_anchors=config.lidar_vert_anchors,
                            lidar_horz_anchors=config.lidar_horz_anchors,
                            seq_len=config.seq_len,
                            embd_pdrop=config.embd_pdrop,
                            attn_pdrop=config.attn_pdrop,
                            resid_pdrop=config.resid_pdrop,
                            config=config, use_velocity=use_velocity)

        self.transformer3 = GPT(n_embd=self.image_encoder.features.feature_info[3]['num_chs'],
                            n_head=config.n_head,
                            block_exp=config.block_exp,
                            n_layer=config.n_layer,
                            img_vert_anchors=config.img_vert_anchors,
                            img_horz_anchors=config.img_horz_anchors,
                            lidar_vert_anchors=config.lidar_vert_anchors,
                            lidar_horz_anchors=config.lidar_horz_anchors,
                            seq_len=config.seq_len,
                            embd_pdrop=config.embd_pdrop,
                            attn_pdrop=config.attn_pdrop,
                            resid_pdrop=config.resid_pdrop,
                            config=config, use_velocity=use_velocity)

        self.transformer4 = GPT(n_embd=self.image_encoder.features.feature_info[4]['num_chs'],
                            n_head=config.n_head,
                            block_exp=config.block_exp,
                            n_layer=config.n_layer,
                            img_vert_anchors=config.img_vert_anchors,
                            img_horz_anchors=config.img_horz_anchors,
                            lidar_vert_anchors=config.lidar_vert_anchors,
                            lidar_horz_anchors=config.lidar_horz_anchors,
                            seq_len=config.seq_len,
                            embd_pdrop=config.embd_pdrop,
                            attn_pdrop=config.attn_pdrop,
                            resid_pdrop=config.resid_pdrop,
                            config=config, use_velocity=use_velocity)

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

    def forward(self, image, lidar, velocity):
        '''
        Image + LiDAR feature fusion using transformers
        Args:
            image: input rgb image
            lidar: LiDAR input will be replaced by positional encoding. Third channel may contain target point.
            velocity (tensor): input velocity from speedometer
        '''

        if self.image_encoder.normalize:
            image_tensor = normalize_imagenet(image)
        else:
            image_tensor = image

        x = torch.linspace(-1, 1, self.config.lidar_resolution_width)
        y = torch.linspace(-1, 1, self.config.lidar_resolution_height)
        y_grid, x_grid = torch.meshgrid(x, y, indexing='ij')

        lidar[:,0] = y_grid.unsqueeze(0) #Top down positional encoding
        lidar[:,1] = x_grid.unsqueeze(0) #Left right positional encoding
        lidar_tensor = lidar

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

        # fusion at (B, 64, 64, 64)
        image_embd_layer1 = self.avgpool_img(image_features)
        lidar_embd_layer1 = self.avgpool_lidar(lidar_features)

        image_features_layer1, lidar_features_layer1 = self.transformer1(image_embd_layer1, lidar_embd_layer1, velocity)

        image_features_layer1 = F.interpolate(image_features_layer1, size=(image_features.shape[2],image_features.shape[3]), mode='bilinear', align_corners=False)
        lidar_features_layer1 = F.interpolate(lidar_features_layer1, size=(lidar_features.shape[2],lidar_features.shape[3]), mode='bilinear', align_corners=False)
        image_features = image_features + image_features_layer1
        lidar_features = lidar_features + lidar_features_layer1

        image_features = self.image_encoder.features.layer2(image_features)
        lidar_features = self.lidar_encoder._model.layer2(lidar_features)
        # fusion at (B, 128, 32, 32)
        image_embd_layer2 = self.avgpool_img(image_features)
        lidar_embd_layer2 = self.avgpool_lidar(lidar_features)

        image_features_layer2, lidar_features_layer2 = self.transformer2(image_embd_layer2, lidar_embd_layer2, velocity)

        image_features_layer2 = F.interpolate(image_features_layer2, size=(image_features.shape[2],image_features.shape[3]), mode='bilinear', align_corners=False)
        lidar_features_layer2 = F.interpolate(lidar_features_layer2, size=(lidar_features.shape[2],lidar_features.shape[3]), mode='bilinear', align_corners=False)
        image_features = image_features + image_features_layer2
        lidar_features = lidar_features + lidar_features_layer2

        image_features = self.image_encoder.features.layer3(image_features)
        lidar_features = self.lidar_encoder._model.layer3(lidar_features)
        # fusion at (B, 256, 16, 16)
        image_embd_layer3 = self.avgpool_img(image_features)
        lidar_embd_layer3 = self.avgpool_lidar(lidar_features)

        image_features_layer3, lidar_features_layer3 = self.transformer3(image_embd_layer3, lidar_embd_layer3, velocity)

        image_features_layer3 = F.interpolate(image_features_layer3, size=(image_features.shape[2],image_features.shape[3]), mode='bilinear', align_corners=False)
        lidar_features_layer3 = F.interpolate(lidar_features_layer3, size=(lidar_features.shape[2],lidar_features.shape[3]), mode='bilinear', align_corners=False)
        image_features = image_features + image_features_layer3
        lidar_features = lidar_features + lidar_features_layer3

        image_features = self.image_encoder.features.layer4(image_features)
        lidar_features = self.lidar_encoder._model.layer4(lidar_features)
        # fusion at (B, 512, 8, 8)
        image_embd_layer4 = self.avgpool_img(image_features)
        lidar_embd_layer4 = self.avgpool_lidar(lidar_features)

        image_features_layer4, lidar_features_layer4 = self.transformer4(image_embd_layer4, lidar_embd_layer4, velocity)

        image_features_layer4 = F.interpolate(image_features_layer4, size=(image_features.shape[2],image_features.shape[3]), mode='bilinear', align_corners=False)
        lidar_features_layer4 = F.interpolate(lidar_features_layer4, size=(lidar_features.shape[2],lidar_features.shape[3]), mode='bilinear', align_corners=False)
        image_features = image_features + image_features_layer4
        lidar_features = lidar_features + lidar_features_layer4

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



class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, n_embd, n_head, block_exp, n_layer, 
                    img_vert_anchors, img_horz_anchors, 
                    lidar_vert_anchors, lidar_horz_anchors,
                    seq_len, 
                    embd_pdrop, attn_pdrop, resid_pdrop, config, use_velocity=True):
        super().__init__()
        self.n_embd = n_embd
        #We currently only support seq len 1
        self.seq_len = 1
        
        self.img_vert_anchors = img_vert_anchors
        self.img_horz_anchors = img_horz_anchors
        self.lidar_vert_anchors = lidar_vert_anchors
        self.lidar_horz_anchors = lidar_horz_anchors
        self.config = config

        # positional embedding parameter (learnable), image + lidar
        self.pos_emb = nn.Parameter(torch.zeros(1, self.seq_len * img_vert_anchors * img_horz_anchors + self.seq_len * lidar_vert_anchors * lidar_horz_anchors, n_embd))
        
        # velocity embedding
        self.use_velocity = use_velocity
        if(use_velocity == True):
            self.vel_emb = nn.Linear(self.seq_len, n_embd)

        self.drop = nn.Dropout(embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, 
                        block_exp, attn_pdrop, resid_pdrop)
                        for layer in range(n_layer)])
        
        # decoder head
        self.ln_f = nn.LayerNorm(n_embd)

        self.block_size = self.seq_len
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=self.config.gpt_linear_layer_init_mean, std=self.config.gpt_linear_layer_init_std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(self.config.gpt_layer_norm_init_weight)

    def forward(self, image_tensor, lidar_tensor, velocity):
        """
        Args:
            image_tensor (tensor): B*4*seq_len, C, H, W
            lidar_tensor (tensor): B*seq_len, C, H, W
            velocity (tensor): ego-velocity
        """
        
        bz = lidar_tensor.shape[0]
        lidar_h, lidar_w = lidar_tensor.shape[2:4]
        img_h, img_w = image_tensor.shape[2:4]
        
        assert self.seq_len == 1
        image_tensor = image_tensor.view(bz, self.seq_len, -1, img_h, img_w).permute(0,1,3,4,2).contiguous().view(bz, -1, self.n_embd)
        lidar_tensor = lidar_tensor.view(bz, self.seq_len, -1, lidar_h, lidar_w).permute(0,1,3,4,2).contiguous().view(bz, -1, self.n_embd)

        
        token_embeddings = torch.cat((image_tensor, lidar_tensor), dim=1)
        # project velocity to n_embed
        if(self.use_velocity==True):
            velocity_embeddings = self.vel_emb(velocity) # (B, C)
            # add (learnable) positional embedding and velocity embedding for all tokens
            x = self.drop(self.pos_emb + token_embeddings + velocity_embeddings.unsqueeze(1)) #(B, an * T, C)
        else:
            x = self.drop(self.pos_emb + token_embeddings)
        x = self.blocks(x) # (B, an * T, C)
        x = self.ln_f(x) # (B, an * T, C)
        x = x.view(bz, self.seq_len*self.img_vert_anchors*self.img_horz_anchors + self.seq_len*self.lidar_vert_anchors*self.lidar_horz_anchors, self.n_embd)

        image_tensor_out = x[:, :self.seq_len*self.img_vert_anchors*self.img_horz_anchors, :].contiguous().view(bz * self.seq_len, -1, img_h, img_w)
        lidar_tensor_out = x[:, self.seq_len*self.img_vert_anchors*self.img_horz_anchors:, :].contiguous().view(bz * self.seq_len, -1, lidar_h, lidar_w)
        
        return image_tensor_out, lidar_tensor_out

        
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
            self._model.act1 = nn.Sequential()
            self._model.maxpool =  nn.Sequential()
            self._model.layer1 = self._model.s1
            self._model.layer2 = self._model.s2
            self._model.layer3 = self._model.s3
            self._model.layer4 = self._model.s4
            self._model.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
            self._model.head = nn.Sequential()
        elif (architecture.startswith('convnext')):
            self._model.conv1 = self._model.stem._modules['0']
            self._model.bn1 = self._model.stem._modules['1']
            self._model.act1 = nn.Sequential()  # Conv NeXt does not use an activation function after the stem
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
            self._model.global_pool.norm = nn.LayerNorm((512,1,1), _tmp.eps, _tmp.elementwise_affine)

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
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x
