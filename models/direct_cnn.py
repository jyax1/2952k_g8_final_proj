import torch
import torch.nn as nn
import numpy as np

class FramesConvolution(nn.Module):
    def __init__(self, latent_size=16):
        super(FramesConvolution, self).__init__()
        self.latent_size = latent_size
        output_dim = int(np.sqrt(latent_size))
        initial_dim = 128  # Assuming input is 128x128

        # Calculate number of downsampling layers needed to reach desired output size
        num_downsamples = int(np.log2(initial_dim / output_dim))

        layers = []
        in_channels = 6
        out_channels = 16
        
        for _ in range(num_downsamples):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels
            out_channels *= 2
        
        # Adjust final output to be (2, sqrt(latent_size), sqrt(latent_size))
        layers.append(nn.Conv2d(in_channels, out_channels=2, kernel_size=3, padding=1))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

class ActionMLP(nn.Module):
    def __init__(self, latent_size=16):
        super(ActionMLP, self).__init__()
        self.latent_size = latent_size
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=2*self.latent_size, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=7)
        )

    def forward(self, x):
        return self.fc(x)

class ActionExtractionCNN(nn.Module):
    def __init__(self, latent_size=16):
        super(ActionExtractionCNN, self).__init__()
        self.latent_size = latent_size
        self.frames_convolution_model = FramesConvolution(latent_size=latent_size)
        self.action_mlp_model = ActionMLP(latent_size=latent_size)

    def forward(self, x):
        x = self.frames_convolution_model(x)
        x = self.action_mlp_model(x)
        return x
    
if __name__ == '__main__':
    model = ActionExtractionCNN(latent_size=16)
    input_tensor = torch.randn(1, 6, 128, 128)
    output = model(input_tensor)
    print(output.shape)  # torch.Size([1, 7])

    # Save the model parts separately
    torch.save(model.frames_convolution_model.state_dict(), 'frames_convolution_cnn.pth')
    torch.save(model.action_mlp_model.state_dict(), 'action_mlp.pth')

    # Load the model parts separately
    frames_encoder_cnn = FramesConvolution()
    action_extraction_mlp = ActionMLP()
    frames_encoder_cnn.load_state_dict(torch.load('frames_convolution_cnn.pth'))
    action_extraction_mlp.load_state_dict(torch.load('action_mlp.pth'))