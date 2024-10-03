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