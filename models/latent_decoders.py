import torch
import torch.nn as nn
import numpy as np

from models.latent_cnn_unet import IDM, FDM
from models.direct_cnn_vit import ActionTransformer, ActionTransformerMLP

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
    

class LatentDecoderAuxiliarySeparateUNetTransformer(nn.Module):
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
        super(LatentDecoderAuxiliarySeparateUNetTransformer, self).__init__()

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
    

class LatentDecoderAuxiliarySeparateUNetMLP(nn.Module):
    def __init__(self, 
                 idm_model_path, 
                 fdm_model_path, 
                 latent_dim=16, 
                 video_length=2, 
                 num_layers=8, 
                 freeze_idm=False, 
                 freeze_fdm=False):
        super(LatentDecoderAuxiliarySeparateUNetMLP, self).__init__()

        # Load and optionally freeze the pretrained IDM model
        self.idm = IDM(latent_dim=latent_dim, video_length=video_length, latent_length=video_length-1)
        self.idm.load_state_dict(torch.load(idm_model_path))
        if freeze_idm:
            for param in self.idm.parameters():
                param.requires_grad = False

        # Load and optionally freeze the pretrained FDM model
        self.fdm = FDM(latent_dim=latent_dim, video_length=video_length-1, latent_length=video_length-1)
        self.fdm.load_state_dict(torch.load(fdm_model_path))
        if freeze_fdm:
            for param in self.fdm.parameters():
                param.requires_grad = False

        # Define the MLP for action extraction
        input_dim = (video_length - 1) * latent_dim ** 2
        mlp_layers = [nn.Flatten(), nn.Linear(input_dim, 512), nn.ReLU()]
        hidden_dim = 512

        for _ in range(num_layers - 1):  # Create hidden layers
            mlp_layers.append(nn.Linear(hidden_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())

        # Final layer outputs the action vector of size 7 * (video_length - 1)
        mlp_layers.append(nn.Linear(hidden_dim, 7 * (video_length - 1)))

        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, image_sequence):
        # Pass the image sequence through the IDM to get the feature map
        feature_map = self.idm(image_sequence)

        # Exclude the last image from the sequence
        past_observations = image_sequence[:, :-3, :, :]

        # Pass the reduced sequence and feature map to the FDM for reconstruction
        reconstructed_image = self.fdm(feature_map, past_observations)

        # Use the MLP to predict the action vector
        action_vector = self.mlp(feature_map)  # Shape: (batch_size, 7 * (video_length - 1))

        # Expand the action vector to match the image dimensions
        action_vector = action_vector.view(action_vector.shape[0], -1, 1, 1).expand(-1, -1, 128, 128)

        # Concatenate the reconstructed image and the action vector
        output = torch.cat((reconstructed_image, action_vector), dim=1)

        return output
    

class ActionVideoReconstructionViT(nn.Module):
    def __init__(self, latent_dim=4, latent_length=2, num_heads=8, num_layers=6, hidden_dim=512, action_length=1):
        super(ActionVideoReconstructionViT, self).__init__()
        self.latent_size = latent_dim**2
        self.latent_length = latent_length
        self.hidden_dim = hidden_dim

        self.patch_size = int(np.sqrt(self.latent_size / 16))  # Assuming latent_size / 16 gives the size of each patch
        self.patch_dim = self.patch_size**2
        self.num_patches = int(latent_length * self.latent_size / self.patch_dim)
        
        self.linear_projection = nn.Linear(self.patch_dim, hidden_dim)

        self.action_token = nn.Parameter(torch.randn(1, 1, hidden_dim))  # Learnable action token

        self.positional_embeddings = nn.Parameter(torch.randn(1, self.num_patches + 1, hidden_dim))  # Positional embeddings

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)  # Transformer layer
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)  # Transformer encoder

        self.action_transformer_mlp = ActionTransformerMLP(hidden_dim=hidden_dim, action_length=action_length)  # MLP head for action labels

        # Decoder layers for video reconstruction
        self.decoder_conv1 = nn.ConvTranspose2d(hidden_dim, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder_conv3 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.output_activation = nn.Sigmoid()  # For generating valid pixel values

    def forward(self, x):
        batch_size = x.size(0)

        # Divide input into patches and flatten them
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(batch_size, -1, self.patch_dim)

        x = self.linear_projection(x)

        # Prepend action token to the patch embeddings
        action_token = self.action_token.expand(batch_size, -1, -1)
        x = torch.cat((action_token, x), dim=1)

        x = x + self.positional_embeddings

        x = self.transformer_encoder(x)

        # Extract class token's output for action prediction
        action_token_output = x[:, 0, :]
        action_output = self.action_transformer_mlp(action_token_output)

        # Reshape remaining tokens into a feature map for reconstruction
        recon_feature_map = x[:, 1:, :].permute(0, 2, 1).view(batch_size, self.hidden_dim, self.patch_size, self.patch_size)

        # Pass through decoder layers for video reconstruction
        recon_image = self.decoder_conv1(recon_feature_map)
        recon_image = self.decoder_conv2(recon_image)
        recon_image = self.decoder_conv3(recon_image)
        recon_image = self.output_activation(recon_image)

        # Expand action output to match the spatial dimensions of the reconstructed image
        action_output_expanded = action_output.view(batch_size, action_output.shape[1], 1, 1).expand(-1, -1, 128, 128)

        # Concatenate reconstructed image and expanded action output
        output = torch.cat((recon_image, action_output_expanded), dim=1)

        return output

class LatentDecoderAuxiliaryCombinedViT(nn.Module):
    def __init__(self, 
                 idm_model_path, 
                 latent_dim=16, 
                 video_length=2, 
                 num_heads=8, 
                 num_layers=6, 
                 freeze_idm=False):
        super(LatentDecoderAuxiliaryCombinedViT, self).__init__()

        # Load and optionally freeze the pretrained IDM model
        self.idm = IDM(latent_dim=latent_dim, video_length=video_length, latent_length=video_length-1)
        self.idm.load_state_dict(torch.load(idm_model_path))
        self.freeze_idm = freeze_idm
        if freeze_idm:
            for param in self.idm.parameters():
                param.requires_grad = False

        # The unified transformer decoder for both video reconstruction and action generation
        self.video_action_decoder = ActionVideoReconstructionViT(
            latent_dim=latent_dim, 
            latent_length=video_length-1, 
            num_heads=num_heads, 
            num_layers=num_layers
        )

    def forward(self, image_sequence):
        # Pass the image sequence through the IDM to get the feature map
        feature_map = self.idm(image_sequence)

        # Use the transformer decoder to generate both the reconstructed video and action labels
        output = self.video_action_decoder(feature_map)

        return output


