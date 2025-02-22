import torch
import torch.nn as nn
import numpy as np

import torch
import torch.nn as nn
import numpy as np

from .direct_cnn_mlp import FramesConvolution

class ActionTransformerMLP(nn.Module):
    def __init__(self, hidden_dim=512, action_length=1):
        super(ActionTransformerMLP, self).__init__()
        layers = [
            nn.Linear(in_features=hidden_dim, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=7*action_length)
        ]
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)
    
class ActionTransformer(nn.Module):
    def __init__(self, latent_dim=4, latent_length=2, num_heads=8, num_layers=6, hidden_dim=512, action_length=1, patch_size = 1):
        super(ActionTransformer, self).__init__()
        self.latent_size = latent_dim**2
        self.latent_length = latent_length

        # self.patch_size = int(np.sqrt(self.latent_size / 16))  # Assuming latent_size / 16 gives the size of each patch
        self.patch_size = patch_size
        self.patch_dim = self.patch_size**2
        self.num_patches = int(latent_length*self.latent_size / self.patch_dim)
        
        self.linear_projection = nn.Linear(self.patch_dim, hidden_dim)

        self.class_token = nn.Parameter(torch.randn(1, 1, hidden_dim)) # Learnable action token

        self.positional_embeddings = nn.Parameter(torch.randn(1, self.num_patches + 1, hidden_dim)) # Positional embeddings

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads) # Transformer layer
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers) # Transformer encoder

        self.action_transformer_mlp = ActionTransformerMLP(hidden_dim=hidden_dim, action_length=action_length) # MLP head

    def forward(self, x):
        batch_size = x.size(0)

        # Divide input into patches and flatten them
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(batch_size, -1, self.patch_dim)

        x = self.linear_projection(x)

        # Prepend action token to the patch embeddings
        class_token = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat((class_token, x), dim=1)

        x = x + self.positional_embeddings

        x = self.transformer_encoder(x)

        # Extract class token's output and use for final prediction
        class_output = x[:, 0, :]
        output = self.action_transformer_mlp(class_output)

        return output
    

class ActionExtractionViT(nn.Module):
    def __init__(self, latent_dim=4, video_length=2, motion=False, image_plus_motion=False, vit_patch_size=1):
        super(ActionExtractionViT, self).__init__()
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
        self.latent_size = latent_dim

        self.frames_convolution_model = FramesConvolution(latent_dim=latent_dim, video_length=self.video_length, latent_length=self.latent_length)
        
        self.action_transformer_model = ActionTransformer(latent_dim=latent_dim, 
                                                          latent_length=self.latent_length, 
                                                          action_length=self.action_length,
                                                          patch_size = vit_patch_size)

    def forward(self, x):
        x = self.frames_convolution_model(x)
        x = self.action_transformer_model(x)
        return x