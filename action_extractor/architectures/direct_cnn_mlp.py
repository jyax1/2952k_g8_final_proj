import torch
import torch.nn as nn
import numpy as np


class FramesConvolution(nn.Module):
    def __init__(self, latent_dim=16, video_length=2, latent_length=2, num_layers=3):
        super(FramesConvolution, self).__init__()
        self.latent_size = latent_dim ** 2
        output_dim = latent_dim
        initial_dim = 128  # Assuming input is 128x128

        # Calculate number of downsampling layers needed to reach desired output size
        num_downsamples = int(np.log2(initial_dim / output_dim))

        layers = []
        in_channels = video_length * 3
        out_channels = 16

        for _ in range(num_downsamples):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))  # Added BatchNorm2d
            layers.append(nn.ReLU())
            in_channels = out_channels
            out_channels *= 2
            
        # Add extra convolutional layers to make it 10 layers in total
        for _ in range(num_layers - num_downsamples):
            layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(in_channels))  # Added BatchNorm2d
            layers.append(nn.ReLU())

        # Adjust final output to be (2, sqrt(latent_size), sqrt(latent_size))
        layers.append(nn.Conv2d(in_channels, out_channels=latent_length, kernel_size=3, padding=1))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

import torch.nn as nn

class ActionMLP(nn.Module):
    def __init__(self, latent_dim=4, latent_length=2, action_length=1, num_layers=9):
        super(ActionMLP, self).__init__()
        self.latent_size = latent_dim ** 2
        
        layers = [
            nn.Flatten(),
            nn.Linear(in_features=latent_length * self.latent_size, out_features=512),
            nn.BatchNorm1d(512),  # Add batch normalization
            nn.ReLU(),
            nn.Dropout(p=0.5)  # Add dropout to prevent overfitting
        ]
        
        # 8 layers in total
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features=512, out_features=512))
            layers.append(nn.BatchNorm1d(512))  # Add batch normalization
            layers.append(nn.ReLU())
            # layers.append(nn.Dropout(p=0.5))  # Add dropout with 50% probability

        # Final layers
        layers.append(nn.Linear(in_features=512, out_features=32))
        layers.append(nn.BatchNorm1d(32))  # Add batch normalization
        layers.append(nn.ReLU())
        
        layers.append(nn.Linear(in_features=32, out_features=7 * action_length))
        layers.append(nn.Tanh())  # Ensure output is in the [-1, 1] range

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)
    
class PoseMLP(nn.Module):
    def __init__(self, latent_dim=4, latent_length=2, pose_length=1, num_layers=9):
        super(PoseMLP, self).__init__()
        self.latent_size = latent_dim ** 2
        
        layers = [
            nn.Flatten(),
            nn.Linear(in_features=latent_length * self.latent_size, out_features=512),
            nn.BatchNorm1d(512),  # Add batch normalization
            nn.ReLU(),
            # nn.Dropout(p=0.5)  # Add dropout to prevent overfitting
        ]
        
        # 8 layers in total
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features=512, out_features=512))
            layers.append(nn.BatchNorm1d(512))  # Add batch normalization
            layers.append(nn.ReLU())
            # layers.append(nn.Dropout(p=0.5))  # Add dropout with 50% probability

        # Final layers
        layers.append(nn.Linear(in_features=512, out_features=32))
        layers.append(nn.BatchNorm1d(32))  # Add batch normalization
        layers.append(nn.ReLU())
        
        layers.append(nn.Linear(in_features=32, out_features=7 * pose_length))

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)

class ActionExtractionCNN(nn.Module):
    def __init__(self, latent_dim=4, video_length=2, motion=False, image_plus_motion=False, num_mlp_layers=6):
        super(ActionExtractionCNN, self).__init__()
        assert not (motion and image_plus_motion), "Choose either only motion or only image_plus_motion"
        if motion:
            self.video_length = video_length - 1
            self.latent_length = self.video_length
        elif image_plus_motion:
            self.video_length = video_length + 1
            self.latent_length = video_length
        else:
            self.video_length = video_length
            self.latent_length = video_length

        self.action_length = video_length - 1
        self.frames_convolution_model = FramesConvolution(latent_dim=latent_dim, video_length=self.video_length, latent_length=self.latent_length)
        self.action_mlp_model = ActionMLP(latent_dim=latent_dim, action_length=self.action_length, latent_length=self.latent_length, num_layers=num_mlp_layers)

    def forward(self, x):
        x = self.frames_convolution_model(x)
        x = self.action_mlp_model(x)
        return x
    

class FramesConvolution3D(nn.Module):
    def __init__(self, latent_dim=16, input_channels=4, latent_length=2, num_layers=3):
        super(FramesConvolution3D, self).__init__()
        self.latent_size = latent_dim ** 3  # Update to latent_dim ** 3 to reflect 3D output
        output_dim = latent_dim
        initial_dim = 64  # Assuming input is 64x64x64

        # Calculate number of downsampling layers needed to reach desired output size
        num_downsamples = int(np.log2(initial_dim / output_dim))

        layers = []
        in_channels = input_channels
        out_channels = 16

        # Downsampling layers
        for _ in range(num_downsamples):
            layers.append(nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm3d(out_channels))  # Added BatchNorm3d
            layers.append(nn.ReLU())
            in_channels = out_channels
            out_channels *= 2
            
        # Add extra convolutional layers to make it 10 layers in total
        for _ in range(num_layers - num_downsamples):
            layers.append(nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm3d(in_channels))  # Added BatchNorm3d
            layers.append(nn.ReLU())

        # Adjust final output to be (latent_length, latent_dim, latent_dim, latent_dim)
        layers.append(nn.Conv3d(in_channels, out_channels=latent_length, kernel_size=3, padding=1))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

class ActionMLP3D(nn.Module):
    def __init__(self, latent_dim=4, latent_length=2, action_length=1, num_layers=9):
        super(ActionMLP3D, self).__init__()
        self.latent_size = latent_dim ** 3  # Update to reflect 3D latent size
        
        layers = [
            nn.Flatten(),
            nn.Linear(in_features=latent_length * self.latent_size, out_features=512),
            nn.BatchNorm1d(512),  # Add batch normalization
            nn.ReLU(),
            nn.Dropout(p=0.5)  # Add dropout to prevent overfitting
        ]
        
        # Add hidden layers
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features=512, out_features=512))
            layers.append(nn.BatchNorm1d(512))  # Add batch normalization
            layers.append(nn.ReLU())

        # Final layers
        layers.append(nn.Linear(in_features=512, out_features=32))
        layers.append(nn.BatchNorm1d(32))  # Add batch normalization
        layers.append(nn.ReLU())
        
        layers.append(nn.Linear(in_features=32, out_features=7 * action_length))
        layers.append(nn.Tanh())  # Ensure output is in the [-1, 1] range

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)
    
class PoseMLP3D(nn.Module):
    def __init__(self, latent_dim=4, latent_length=2, pose_length=1, num_layers=9):
        super(PoseMLP3D, self).__init__()
        self.latent_size = latent_dim ** 3  # Update to reflect 3D latent size
        
        layers = [
            nn.Flatten(),
            nn.Linear(in_features=latent_length * self.latent_size, out_features=512),
            nn.BatchNorm1d(512),  # Add batch normalization
            nn.ReLU(),
        ]
        
        # Add hidden layers
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features=512, out_features=512))
            layers.append(nn.BatchNorm1d(512))  # Add batch normalization
            layers.append(nn.ReLU())

        # Final layers
        layers.append(nn.Linear(in_features=512, out_features=32))
        layers.append(nn.BatchNorm1d(32))  # Add batch normalization
        layers.append(nn.ReLU())
        
        layers.append(nn.Linear(in_features=32, out_features=7 * pose_length))

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)

class ActionExtractionCNN3D(nn.Module):
    def __init__(self, latent_dim=4, input_channels=4, motion=False, image_plus_motion=False, num_mlp_layers=6):
        super(ActionExtractionCNN3D, self).__init__()
        assert not (motion and image_plus_motion), "Choose either only motion or only image_plus_motion"
        if motion:
            self.latent_length = input_channels - 1
        elif image_plus_motion:
            self.latent_length = input_channels + 1
        else:
            self.latent_length = input_channels

        self.action_length = input_channels - 1
        self.frames_convolution_model = FramesConvolution3D(latent_dim=latent_dim, input_channels=input_channels, latent_length=self.latent_length)
        self.action_mlp_model = ActionMLP3D(latent_dim=latent_dim, action_length=self.action_length, latent_length=self.latent_length, num_layers=num_mlp_layers)

    def forward(self, x):
        x = self.frames_convolution_model(x)
        x = x.view(x.size(0), -1)  # Flatten the convolutional output to fit into the MLP
        x = self.action_mlp_model(x)
        return x
    
class PoseExtractionCNN3D(nn.Module):
    def __init__(self, latent_dim=4, input_channels=4, motion=False, image_plus_motion=False, num_mlp_layers=6):
        super(PoseExtractionCNN3D, self).__init__()
        assert not (motion and image_plus_motion), "Choose either only motion or only image_plus_motion"
        if motion:
            self.latent_length = input_channels - 1
        elif image_plus_motion:
            self.latent_length = input_channels + 1
        else:
            self.latent_length = input_channels
            
        self.frames_convolution_model = FramesConvolution3D(latent_dim=latent_dim, input_channels=input_channels, latent_length=self.latent_length)
        self.pose_mlp_model = PoseMLP3D(latent_dim=latent_dim, pose_length=1, latent_length=self.latent_length, num_layers=num_mlp_layers)

    def forward(self, x):
        x = self.frames_convolution_model(x)
        x = self.pose_mlp_model(x)
        return x
    
if __name__ == '__main__':
    model = ActionExtractionCNN(latent_size=16)
    input_tensor = torch.randn(1, 3, 128, 128)
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