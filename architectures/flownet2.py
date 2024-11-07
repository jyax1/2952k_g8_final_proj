import torch
import torch.nn as nn
from flownet2_pytorch.models import FlowNet2  # Importing FlowNet2 from models.py

class FlowNet2MLP(nn.Module):
    def __init__(self, input_size, hidden_size=512, final_size=32, output_size=7, num_layers=30, dropout_prob=0.5):
        super(FlowNet2MLP, self).__init__()
        
        layers = [
            nn.Linear(in_features=input_size, out_features=hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
        ]
        
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.LeakyReLU())

        layers.append(nn.Linear(hidden_size, final_size))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(final_size, output_size))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class FlowNet2ActionExtractor(nn.Module):
    def __init__(self, flownet_version='flownet2', video_length=2, in_channels=3, action_length=1, num_classes=7, num_mlp_layers=3):
        super(FlowNet2ActionExtractor, self).__init__()

        # Define the FlowNet2 model
        self.flownet2 = FlowNet2(args=None)  # Assuming args are not needed for this context

        # Use FlowNet2MLP for the MLP head
        self.mlp = FlowNet2MLP(
            input_size=1024,  # Assuming the output size of FlowNet2 is 1024
            hidden_size=512,
            final_size=32,
            output_size=num_classes * action_length,
            num_layers=num_mlp_layers
        )
        
    def forward(self, x):
        # Pass through FlowNet2 backbone
        x = self.flownet2(x)
        # Flatten the output to fit into the MLP
        x = torch.flatten(x, 1)
        # Pass through MLP head
        x = self.mlp(x)
        return x
