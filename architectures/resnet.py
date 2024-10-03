import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18(nn.Module):
    def __init__(self, output_size=7, mlp_hidden_size=512, num_mlp_layers=5):
        super(ResNet18, self).__init__()
        
        self.resnet = models.resnet18(pretrained=False)
        
        self.conv_part = nn.Sequential(*list(self.resnet.children())[:-1])
        
        mlp_layers = []
        input_size = self.resnet.fc.in_features  # Input size for the MLP, which is 512 in ResNet18

        for _ in range(num_mlp_layers - 1):
            mlp_layers.append(nn.Linear(input_size, mlp_hidden_size))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.BatchNorm1d(mlp_hidden_size))
            mlp_layers.append(nn.Dropout(0.5))
            input_size = mlp_hidden_size
            
        # Constrict to 49 perceptrons
        mlp_layers.append(nn.Linear(mlp_hidden_size, 49))
        
        # Final layer that outputs the action vector
        mlp_layers.append(nn.Linear(49, output_size))
        
        # Define MLP as a Sequential model
        self.mlp_part = nn.Sequential(*mlp_layers)
    
    def forward(self, x):
        conv_features = self.conv_part(x)
        # Flatten the convolutional output to feed into the MLP
        conv_features = conv_features.view(conv_features.size(0), -1)
        output = self.mlp_part(conv_features)
        
        return output

    def extract_conv_features(self, x):
        with torch.no_grad():
            conv_features = self.conv_part(x)
            conv_features = conv_features.view(conv_features.size(0), -1)
        return conv_features

    def forward_mlp_only(self, conv_features):
        return self.mlp_part(conv_features)


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
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


class Bottleneck3D(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.conv3 = nn.Conv3d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels * self.expansion)
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


class ResNet3D(nn.Module):
    def __init__(self, block, layers, input_channels=4, num_classes=7):
        super(ResNet3D, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv3d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet18_3d(input_channels=4, num_classes=7):
    """Constructs a ResNet-18 3D model with 4 input channels."""
    return ResNet3D(BasicBlock3D, [2, 2, 2, 2], input_channels=input_channels, num_classes=num_classes)


def resnet50_3d(input_channels=4, num_classes=7):
    """Constructs a ResNet-50 3D model with 4 input channels."""
    return ResNet3D(Bottleneck3D, [3, 4, 6, 3], input_channels=input_channels, num_classes=num_classes)