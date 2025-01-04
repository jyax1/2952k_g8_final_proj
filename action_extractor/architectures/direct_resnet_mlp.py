import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetMLP(nn.Module):
    def __init__(self, input_size, hidden_size=512, final_size=32, output_size=7, num_layers=30, dropout_prob=0.5):
        super(ResNetMLP, self).__init__()
        
        layers = [
            nn.Linear(in_features=input_size, out_features=hidden_size),
            nn.BatchNorm1d(hidden_size),  # Add batch normalization
            nn.LeakyReLU(),
            # nn.Dropout(p=dropout_prob)    # Add dropout
        ]
        
        # Add hidden layers
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            # layers.append(nn.BatchNorm1d(hidden_size))  # Add batch normalization
            layers.append(nn.LeakyReLU())
            # layers.append(nn.Dropout(p=dropout_prob))  # Add dropout

        # Add final layer that condenses from hidden_size (512) to final_size (32)
        layers.append(nn.Linear(hidden_size, final_size))
        # layers.append(nn.BatchNorm1d(final_size))  # Add batch normalization
        layers.append(nn.LeakyReLU())

        # Final output layer
        layers.append(nn.Linear(final_size, output_size))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, video_length=2, in_channels=3, use_spatial_softmax=True):
        super(ResNet, self).__init__()
        self.use_spatial_softmax = use_spatial_softmax  # <--- NEW parameter

        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels * video_length, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # We'll keep AdaptiveAvgPool2d for the case of global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x shape typically [B, 6, H, W] if video_length=2 and in_channels=3

        x = self.conv1(x)  # [B, 64, H/2, W/2]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # [B, 64, H/4, W/4]

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # e.g. [B, 512, H/32, W/32] if ResNet18

        if self.use_spatial_softmax:
            # ----------- Spatial Softmax Approach -----------
            # 1) Flatten the spatial dims to do a softmax across them
            B, C, H, W = x.shape
            # flatten to [B, C, H*W]
            x_2d = x.view(B, C, -1)            
            # softmax across the last dimension (the spatial dimension)
            softmax_map = F.softmax(x_2d, dim=2)  # [B, C, H*W]
            # reshape back to [B, C, H, W]
            softmax_map = softmax_map.view(B, C, H, W)
            # Weighted sum of the original x by the softmax => shape [B, C]
            x = (softmax_map * x).sum(dim=[2, 3])  # [B, C]
        else:
            # ----------- Global Average Pooling Approach -----------
            x = self.avgpool(x)  # [B, 512, 1, 1]
            x = x.view(x.size(0), -1)  # flatten to [B, C]

        return x

from .utils import resnet_builder

class ActionExtractionResNet(nn.Module):
    def __init__(self, resnet_version='resnet18', video_length=2, in_channels=3, action_length=1, num_classes=7, num_mlp_layers=3):
        super(ActionExtractionResNet, self).__init__()

        # Define the ResNet version to use
        self.conv, resnet_out_dim = resnet_builder(resnet_version=resnet_version, video_length=video_length, in_channels=in_channels)

        # Use ResNetMLP for the MLP head
        self.mlp = ResNetMLP(
            input_size=resnet_out_dim,
            hidden_size=512,
            final_size=32,
            output_size=num_classes * action_length,
            num_layers=num_mlp_layers
        )
        
    def forward(self, x):
        # Pass through ResNet backbone
        x = self.conv(x)
        # Flatten the output to fit into the MLP
        x = torch.flatten(x, 1)
        # Pass through MLP head
        x = self.mlp(x)
        return x