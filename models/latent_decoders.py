import torch
import torch.nn as nn

from models.latent_cnn_unet import IDM, FDM
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
        idm_output = self.idm(x)

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
        self.transformer = ActionTransformer(latent_dim=latent_dim, 
                                             latent_length=latent_length,
                                             num_heads=num_heads, 
                                             num_layers=num_layers, 
                                             hidden_dim=hidden_dim, 
                                             action_length=video_length-1)

    def forward(self, x):
        idm_output = self.idm(x)
        
        output = self.transformer(idm_output)

        return output
    
class LatentDecoderObsConditionedUNetMLP(nn.Module):
    def __init__(self, idm_model_path, latent_dim=16, video_length=2, latent_length=1, unet_latent_length=256, mlp_layers=3):
        super(LatentDecoderObsConditionedUNetMLP, self).__init__()

        self.idm = IDM(latent_dim=latent_dim, video_length=video_length, latent_length=latent_length)
        self.idm.load_state_dict(torch.load(idm_model_path))
        for param in self.idm.parameters():
            param.requires_grad = False

        self.unet = FDM(latent_length=latent_length, latent_dim=latent_dim, video_length=video_length, unet_latent_length=unet_latent_length)
        
        # Define the MLP layers
        self.latent_size = latent_dim ** 2
        mlp_layers_list = [nn.Flatten(), nn.Linear(video_length * 3 * 128**2, 512), nn.ReLU()]

        # Add adjustable number of layers
        for _ in range(mlp_layers):
            mlp_layers_list.append(nn.Linear(512, 512))
            mlp_layers_list.append(nn.ReLU())

        # Final output layer
        mlp_layers_list.append(nn.Linear(512, 7 * (video_length-1)))

        # Build the MLP sequential model
        self.mlp = nn.Sequential(*mlp_layers_list)

    def forward(self, x):
        idm_output = self.idm(x)

        unet_output = self.unet(idm_output, x)
        
        output = self.mlp(unet_output)

        return output
    

class LatentDecoderAuxiliaryTransformer(nn.Module):
    def __init__(self, 
                 idm_model_path, 
                 fdm_model_path, 
                 latent_dim=16, 
                 video_length=2, 
                 num_heads=8, 
                 num_layers=6, 
                 hidden_dim=512, 
                 freeze_idm=False, 
                 freeze_fdm=False):
        super(LatentDecoderAuxiliaryTransformer, self).__init__()

        # Load and optionally freeze the pretrained IDM model
        self.idm = IDM(latent_dim=latent_dim, video_length=video_length, latent_length=video_length-1)
        self.idm.load_state_dict(torch.load(idm_model_path))
        self.freeze_idm = freeze_idm
        if freeze_idm:
            for param in self.idm.parameters():
                param.requires_grad = False

        # Load and optionally freeze the pretrained FDM model
        self.fdm = FDM(latent_dim=latent_dim, video_length=video_length-1, latent_length=video_length-1)
        self.fdm.load_state_dict(torch.load(fdm_model_path))
        self.freeze_fdm = freeze_fdm
        if freeze_fdm:
            for param in self.fdm.parameters():
                param.requires_grad = False

        # Define the transformer for action extraction
        self.transformer = ActionTransformer(latent_dim=latent_dim, 
                                             latent_length=video_length-1,
                                             num_heads=num_heads, 
                                             num_layers=num_layers, 
                                             hidden_dim=hidden_dim, 
                                             action_length=video_length-1)

    def forward(self, image_sequence):
        # Pass the image sequence through the IDM to get the feature map
        feature_map = self.idm(image_sequence)

        # Exclude the last image from the sequence
        past_observations = image_sequence[:, :-3, :, :]

        # Pass the reduced sequence and feature map to the FDM for reconstruction
        reconstructed_image = self.fdm(feature_map, past_observations)

        # Use the transformer to predict the action vector
        action_vector = self.transformer(feature_map)

        action_vector = action_vector.view(action_vector.shape[0], action_vector.shape[1], 1, 1).expand(-1, -1, 128, 128)

        output = torch.cat((reconstructed_image, action_vector), dim=1)

        return output