'''
Instead of predicting actions directly form consecutive frames,
follow the paper "Learning to Act without Actions" to create latent action extractor
'''

import torch.nn as nn
import numpy as np

from models.direct_cnn_mlp import FramesConvolution as IDM

class FiLM(nn.Module):
    def __init__(self, latent_length, unet_latent_dim=32, unet_latent_length=256):
        super(FiLM, self).__init__()
        
        # 1x1 convolution to align the feature map channels with the target channels
        self.unet_latent_dim = unet_latent_dim
        self.channel_mapper = nn.Conv2d(latent_length, unet_latent_length, kernel_size=1)
        
        # Additional layers to generate gamma and beta for conditioning
        self.gamma_layer = nn.Conv2d(unet_latent_length, unet_latent_length, kernel_size=1)
        self.beta_layer = nn.Conv2d(unet_latent_length, unet_latent_length, kernel_size=1)
    
    def forward(self, feature_map, x):
        # Align the feature map channels with the target channels
        aligned_feature_map = self.channel_mapper(feature_map)

        if aligned_feature_map.size(2) != self.unet_latent_dim or aligned_feature_map.size(3) != self.unet_latent_dim:
            aligned_feature_map = nn.functional.interpolate(
                aligned_feature_map, size=(int(self.unet_latent_dim), int(self.unet_latent_dim)), mode='bilinear', align_corners=False
            )
        
        # Generate FiLM parameters (gamma and beta) from the aligned feature map
        gamma = self.gamma_layer(aligned_feature_map)
        beta = self.beta_layer(aligned_feature_map)
        
        # Apply FiLM: gamma * x + beta
        return gamma * x + beta

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class FDM(nn.Module):
    def __init__(self, latent_length=1, latent_dim=4, video_length=2, unet_latent_length=256):
        super(FDM, self).__init__()
        
        self.video_length = video_length
        self.latent_dim = latent_dim
        self.latent_length = latent_length
        unet_latent_dim = 2**(np.log2(128) - 2)

        in_channels = video_length * 3
        out_channels = video_length * 3
        
        # FiLM conditioning
        self.film = FiLM(latent_length, unet_latent_dim=unet_latent_dim, unet_latent_length=unet_latent_length)
        
        # U-Net Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, unet_latent_length, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(unet_latent_length),
            nn.ReLU()
        )
        
        # Residual Blocks
        self.res_block = ResidualBlock(unet_latent_length, unet_latent_length)
        
        # U-Net Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(unet_latent_length, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        
    def forward(self, feature_map, image_sequence):

        x1 = self.enc1(image_sequence)  # Output shape: (batch_size, 64, 128, 128)
        x2 = self.enc2(x1)  # Output shape: (batch_size, 128, 64, 64)
        x3 = self.enc3(x2)  # Output shape: (batch_size, unet_latent_length, 32, 32)
        
        # FiLM conditioning applied to the deepest layer
        x3 = self.film(feature_map, x3)
        
        # Residual block
        x3 = self.res_block(x3)
        
        # Decoder with skip connections
        x = self.dec1(x3)  # Output shape: (batch_size, 128, 64, 64)
        x = x + x2  # Skip connection from encoder
        x = self.dec2(x)  # Output shape: (batch_size, 64, 128, 128)
        x = x + x1  # Skip connection from encoder
        x = self.final_conv(x)
        
        # Output the transformed image sequence
        return x
    
class ActionExtractionCNNUNet(nn.Module):
    def __init__(self, latent_dim=16, video_length=2):
        super(ActionExtractionCNNUNet, self).__init__()
        self.idm = IDM(latent_dim=latent_dim, video_length=video_length, latent_length=video_length-1)
        
        self.fdm = FDM(latent_dim=latent_dim, video_length=video_length-1, latent_length=video_length-1)
        
    def forward(self, image_sequence):
        
        # Pass the image sequence through the IDM to get the feature map
        feature_map = self.idm(image_sequence)
        
        # Exclude the last image from the sequence
        past_observations = image_sequence[:, :-3, :, :]
        
        # Pass the reduced sequence and feature map to the FDM
        prediction = self.fdm(feature_map, past_observations)

        return prediction
        