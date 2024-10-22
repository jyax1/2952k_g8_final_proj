import torch
import torch.nn as nn
from flownet2 import FlowNet2, FlowNet2C, FlowNet2CS, FlowNet2CSS, FlowNet2CSS_ftSD, FlowNet2S, FlowNet2SD

class FlowNet2PoseEstimation(nn.Module):
    def __init__(self, video_length=2, in_channels=3, num_classes=9, version='FlowNet2', num_fc_layers=3, fc_hidden_size=512):
        super(FlowNet2PoseEstimation, self).__init__()
        
        # Load the specified FlowNet2 version
        self.flownet2 = self._select_flownet_version(version)
        
        # Modify the first convolutional layer to accept video_length * in_channels input
        original_conv1 = self.flownet2.module.flownets[0].conv1  # access the first conv layer
        
        # New conv1 to accept video_length * in_channels channels
        self.flownet2.module.flownets[0].conv1 = nn.Conv2d(
            in_channels=video_length * in_channels,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding
        )

        # Fully connected layers part
        self.fc = self._build_fc_layers(1024, fc_hidden_size, num_classes, num_fc_layers)

    def forward(self, x):
        # Pass the input through the FlowNet2 backbone
        flownet_output = self.flownet2(x)
        
        # Flatten the output to pass into the fully connected layers
        flattened = flownet_output.view(flownet_output.size(0), -1)
        
        # Pose estimation through fully connected layers
        pose = self.fc(flattened)
        return pose

    def _build_fc_layers(self, input_size, hidden_size, output_size, num_fc_layers):
        """Helper function to create the adjustable FC layers."""
        layers = []
        
        # Add the first layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())

        # Add the adjustable number of hidden layers
        for _ in range(num_fc_layers - 1):  # We subtract 1 because we already added the first layer
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        # Final output layer
        layers.append(nn.Linear(hidden_size, output_size))

        return nn.Sequential(*layers)

    def _select_flownet_version(self, version):
        """Select the appropriate FlowNet2 version."""
        if version == 'FlowNet2':
            return FlowNet2()
        elif version == 'FlowNet2-C':
            return FlowNet2C()
        elif version == 'FlowNet2-CS':
            return FlowNet2CS()
        elif version == 'FlowNet2-CSS':
            return FlowNet2CSS()
        elif version == 'FlowNet2-CSS-ft-sd':
            return FlowNet2CSS_ftSD()
        elif version == 'FlowNet2-S':
            return FlowNet2S()
        elif version == 'FlowNet2-SD':
            return FlowNet2SD()
        else:
            raise ValueError(f"Unsupported FlowNet2 version: {version}")
        
        
model = FlowNet2PoseEstimation(video_length=2, in_channels=4, num_classes=9, version='FlowNet2-CSS', num_fc_layers=4, fc_hidden_size=512)
input_data = torch.randn(1, 2 * 4, 128, 128)  # Example input for a sequence of 2 RGB-D frames
output = model(input_data)
print(output.shape)  # Should output a (batch_size, 9) tensor