import torch
import torch.nn as nn

from models.latent_cnn_unet import IDM
from models.direct_cnn_vit import ActionTransformer

class LatentDecoderMLP(nn.Module):
    def __init__(self, idm_model_path, latent_dim=16, video_length=2, latent_length=1, mlp_layers=3):
        super(LatentDecoderMLP, self).__init__()

        # Load the pre-trained IDM model and freeze its parameters
        self.idm = IDM(latent_dim=latent_dim, video_length=video_length, latent_length=latent_length)
        self.idm.load_state_dict(torch.load(idm_model_path))
        for param in self.idm.parameters():
            param.requires_grad = False

        # Define the MLP layers
        self.latent_size = latent_dim ** 2
        mlp_layers_list = [nn.Flatten(), nn.Linear(latent_length * self.latent_size, 512), nn.ReLU()]

        # Add adjustable number of layers
        for _ in range(mlp_layers):
            mlp_layers_list.append(nn.Linear(512, 512))
            mlp_layers_list.append(nn.ReLU())

        # Final output layer
        mlp_layers_list.append(nn.Linear(512, 7 * (video_length-1)))

        # Build the MLP sequential model
        self.mlp = nn.Sequential(*mlp_layers_list)

    def forward(self, x):
        # Pass input through the frozen IDM model
        idm_output = self.idm(x)

        # Pass the IDM output through the MLP
        mlp_output = self.mlp(idm_output)

        return mlp_output
    

class LatentDecoderTransformer(nn.Module):
    def __init__(self, idm_model_path, latent_dim=16, video_length=2, latent_length=1, num_heads=8, num_layers=6, hidden_dim=512):
        super(LatentDecoderTransformer, self).__init__()

        # Load and freeze the pre-trained IDM model
        self.idm = IDM(latent_dim=latent_dim, video_length=video_length, latent_length=latent_length)
        self.idm.load_state_dict(torch.load(idm_model_path))
        for param in self.idm.parameters():
            param.requires_grad = False

        # Define the transformer for action extraction
        action_length = 1
        self.transformer = ActionTransformer(latent_dim=latent_dim, latent_length=latent_length, num_heads=num_heads, num_layers=num_layers, hidden_dim=hidden_dim, action_length=action_length)

    def forward(self, x):
        # Pass the input through the frozen IDM model
        idm_output = self.idm(x)
        
        # Pass the IDM output through the transformer
        output = self.transformer(idm_output)

        return output