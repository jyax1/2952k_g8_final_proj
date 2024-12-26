import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import resnet_builder
from action_extractor.architectures.direct_resnet_mlp import ResNetMLP

class ActionExtractionVariationalResNet(BaseVAE):
    def __init__(self, resnet_version='resnet18', video_length=2, in_channels=3, 
                 latent_dim=32, action_length=1, num_classes=7, num_mlp_layers=3):
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
        
        self.mlp = ResNetMLP(
            input_size=latent_dim,
            hidden_size=512,
            final_size=32,
            output_size=num_classes * action_length,
            num_layers=num_mlp_layers
        )

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

    def kl_divergence(self, mu, logvar):
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


class ActionExtractionHypersphericalResNet(BaseVAE):
    def __init__(self, resnet_version='resnet18', video_length=2, in_channels=3, 
                 latent_dim=32, action_length=1, num_classes=7, num_mlp_layers=3):
        super(ActionExtractionHypersphericalResNet, self).__init__()

        # Build the ResNet backbone
        self.conv, resnet_out_dim = resnet_builder(
            resnet_version=resnet_version, 
            video_length=video_length, 
            in_channels=in_channels
        )

        # Encoder outputs for mean direction and concentration
        self.fc_mu = nn.Linear(resnet_out_dim, latent_dim)
        self.fc_kappa = nn.Linear(resnet_out_dim, 1)
        
        self.mlp = ResNetMLP(
            input_size=latent_dim,
            hidden_size=512,
            final_size=32,
            output_size=num_classes * action_length,
            num_layers=num_mlp_layers
        )

    def encode(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        # Get mean direction (mu) and concentration (kappa)
        mu = F.normalize(self.fc_mu(h), dim=-1)  # Normalize to unit vector
        kappa = F.softplus(self.fc_kappa(h)) + 1  # Ensure kappa > 0
        return mu, kappa

    def reparameterize(self, mu, kappa):
        batch_size = mu.shape[0]
        latent_dim = mu.shape[1]

        # Sample from vMF distribution
        v = torch.randn(batch_size, latent_dim, device=mu.device)
        v = F.normalize(v, dim=-1)
        
        # Sample from radial component
        w = self._sample_vMF_radial(kappa, latent_dim)
        
        # Combine direction and magnitude
        z = self._householder_rotation(v, mu)
        z = z * w.unsqueeze(-1)
        return z

    def _sample_vMF_radial(self, kappa, dim):
        w = torch.zeros_like(kappa)
        for i in range(kappa.shape[0]):
            k = kappa[i].item()
            done = False
            while not done:
                z = torch.rand(1) * (1 - torch.exp(-2*k)) + torch.exp(-2*k)
                w_temp = (1 - torch.log(z)/k)
                u = torch.rand(1)
                if torch.log(u) <= (dim-3)*torch.log(w_temp) + k*w_temp:
                    w[i] = w_temp
                    done = True
        return w

    def _householder_rotation(self, v, mu):
        u = F.normalize(v - mu, dim=-1)
        return v - 2 * (v * u).sum(-1, keepdim=True) * u

    def forward(self, x):
        mu, kappa = self.encode(x)
        z = self.reparameterize(mu, kappa)
        return self.mlp(z), mu, kappa

    def kl_divergence(self, mu, kappa):
        dim = mu.shape[1]
        return torch.mean(
            kappa * -(torch.besseli(dim/2, kappa) / torch.besseli(dim/2-1, kappa)) + \
            (dim/2-1) * torch.log(kappa) - dim/2 * np.log(2*np.pi) - \
            torch.log(torch.besseli(dim/2-1, kappa))
        )