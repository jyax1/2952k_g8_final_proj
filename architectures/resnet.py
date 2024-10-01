import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18(nn.Module):
    def __init__(self, output_size=7):
        super(ResNet18, self).__init__()
        
        self.resnet = models.resnet18(pretrained=False)
        
        self.conv_part = nn.Sequential(*list(self.resnet.children())[:-1])  # All layers except the last FC
        
        self.mlp_part = nn.Linear(in_features=self.resnet.fc.in_features, out_features=output_size)
    
    def forward(self, x):
        conv_features = self.conv_part(x)
        
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