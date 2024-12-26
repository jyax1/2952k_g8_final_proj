import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import resnet_builder
from action_extractor.architectures.direct_resnet_mlp import ResNetMLP
import numpy as np
import math

class BaseVAE(nn.Module):
    def kl_divergence(self, *args):
        raise NotImplementedError

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
        return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))


class ActionExtractionHypersphericalResNet(BaseVAE):
    def __init__(self, resnet_version='resnet18', video_length=2, in_channels=3, 
                 latent_dim=32, action_length=1, num_classes=7, num_mlp_layers=3):
        super(ActionExtractionHypersphericalResNet, self).__init__()

        self.latent_dim = latent_dim
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
        mu = F.normalize(self.fc_mu(h), dim=-1)
        kappa = F.softplus(self.fc_kappa(h)) + 1
        return mu, kappa

    def _sample_vMF_radial(self, kappa, dim):
        """Sample from vMF radial component
        Args:
            kappa: [batch_size]
            dim: scalar latent dimension
        Returns:
            w: [batch_size] radial components
        """
        w = torch.zeros(kappa.shape[0], device=kappa.device)
        for i in range(kappa.shape[0]):
            k = kappa[i]
            done = False
            while not done:
                # Numerically stable sampling
                z = torch.rand(1, device=k.device)
                z = z * (1 - torch.exp(-2*k)) + torch.exp(-2*k)
                w_temp = 1 - torch.log(z)/k
                u = torch.rand(1, device=k.device)
                if torch.log(u) <= (dim-3)*torch.log(w_temp) + k*w_temp:
                    w[i] = w_temp
                    done = True
        return w

    def _householder_rotation(self, v, mu):
        u = F.normalize(v - mu, dim=-1)
        return v - 2 * (v * u).sum(-1, keepdim=True) * u

    def reparameterize(self, mu, kappa):
        batch_size = mu.shape[0]
        latent_dim = mu.shape[1]
        
        # Generate random direction: [batch_size, latent_dim]
        v = torch.randn(batch_size, latent_dim, device=mu.device)
        v = F.normalize(v, dim=-1)
        
        # Sample radial component: [batch_size]
        w = self._sample_vMF_radial(kappa.squeeze(-1), latent_dim)  # Remove extra dim from kappa
        
        # Householder rotation: [batch_size, latent_dim]
        z = self._householder_rotation(v, mu)
        
        # Correct broadcasting for w: [batch_size, 1] * [batch_size, latent_dim]
        z = z * w.unsqueeze(-1)
        
        return z

    def _vmf_norm_const(self, dim, kappa):
        """Compute vMF normalization constant
        Args:
            dim: scalar latent dimension
            kappa: [batch_size] concentration parameter
        Returns:
            log_c: [batch_size] log normalization constants
        """
        v = torch.tensor(dim/2 - 0.5, device=kappa.device)
        # Numerically stable computation
        sqrt_term = torch.sqrt(torch.pow(v, 2) + torch.pow(kappa, 2))
        log_c = torch.log((v + sqrt_term)/kappa)
        log_2pi = torch.log(torch.tensor(2*np.pi, device=kappa.device))
        return log_c - log_2pi

    def forward(self, x):
        mu, kappa = self.encode(x)
        z = self.reparameterize(mu, kappa)
        output = self.mlp(z)
        return output, mu, kappa

    def kl_divergence(self, mu, kappa):
    #     """KL divergence between vMF(mu, kappa) and uniform on sphere
    #     Args:
    #         mu: [batch_size, latent_dim] unit vectors
    #         kappa: [batch_size, 1] concentration parameters
    #     Returns:
    #         KL divergence scalar
    #     """
    #     dim = mu.shape[1]
    #     kappa = kappa.squeeze(-1)  # [batch_size]
        
    #     log_norm = self._vmf_norm_const(dim, kappa)
    #     log_2pi = torch.log(torch.tensor(2*np.pi, device=kappa.device))
        
    #     kld = kappa + (dim/2 - 1) * torch.log(kappa) - \
    #           (dim/2) * log_2pi - log_norm + log_2pi
        
    #     return torch.mean(kld)