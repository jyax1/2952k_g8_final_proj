import torch
import torch.nn as nn
from .utils import resnet_builder

class ActionExtractionVariationalResNet(nn.Module):
    def __init__(self, resnet_version='resnet18', video_length=2, in_channels=3, latent_dim=32, num_classes=7, num_mlp_layers=3):
        super(ActionExtractionVariationalResNet, self).__init__()

        # Build the ResNet backbone
        self.conv, resnet_out_dim = resnet_builder(
            resnet_version=resnet_version, 
            video_length=video_length, 
            in_channels=in_channels
        )

        # Encoder outputs for mean and log variance
        self.fc_mu = nn.Linear(resnet_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(resnet_out_dim, latent_dim)

        # Decoder (MLP head)
        mlp_layers = [
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU()
        ]
        for _ in range(num_mlp_layers):
            mlp_layers.extend([
                nn.Linear(512, 512),
                nn.LeakyReLU()
            ])
        mlp_layers.append(nn.Linear(512, num_classes))
        self.mlp = nn.Sequential(*mlp_layers)

    def encode(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # Standard deviation
        eps = torch.randn_like(std)    # Sample epsilon from standard normal
        z = mu + eps * std             # Reparameterization trick
        return z

    def decode(self, z):
        return self.mlp(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        output = self.decode(z)
        return output, mu, logvar