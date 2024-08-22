'''
Instead of predicting actions directly form consecutive frames,
follow the paper "Learning to Act without Actions" to create latent action extractor
'''

import torch
import torch.nn as nn
import numpy as np

from models.direct_cnn_mlp import FramesConvolution as IDM

class FDM(nn.Module):
    def __init__(self, latent_dim=16, video_length=2, latent_length=1):
        super(FDM, self).__init__()
        self.latent_size = latent_dim**2
        output_dim = latent_dim
        initial_dim = 128  # Assuming input is 128x128

        # Calculate number of downsampling layers needed to reach desired output size
        num_downsamples = int(np.log2(initial_dim / output_dim))

        layers = []
        in_channels = video_length * 3
        out_channels = 16
        
        for _ in range(num_downsamples):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels
            out_channels *= 2
        
        # Add extra convolutional layers to make it 10 layers in total
        for _ in range(10 - num_downsamples):
            layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())

        # Adjust final output to be (2, sqrt(latent_size), sqrt(latent_size))
        layers.append(nn.Conv2d(in_channels, out_channels=latent_length, kernel_size=3, padding=1))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)