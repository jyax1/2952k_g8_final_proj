import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import resnet_builder
from action_extractor.architectures.direct_resnet_mlp import ResNetMLP
import numpy as np
import math
from scipy.special import iv

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
        # --------------------------------------------------
        # 1) Build the ResNet backbone
        self.conv, resnet_out_dim = resnet_builder(
            resnet_version=resnet_version, 
            video_length=video_length, 
            in_channels=in_channels
        )

        # --------------------------------------------------
        # 2) Encoder outputs for mean direction (mu) and concentration (kappa)
        self.fc_mu = nn.Linear(resnet_out_dim, latent_dim)
        self.fc_kappa = nn.Linear(resnet_out_dim, 1)
        
        # --------------------------------------------------
        # 3) MLP to map the latent z -> final action output
        self.mlp = ResNetMLP(
            input_size=latent_dim,
            hidden_size=512,
            final_size=32,
            output_size=num_classes * action_length,
            num_layers=num_mlp_layers
        )

    def encode(self, x):
        """Encode input x -> (mu, kappa) for the vMF distribution."""
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        # mu in R^d, forced to unit vector
        mu = F.normalize(self.fc_mu(h), dim=-1)
        # kappa >= 1 to avoid kappa=0 corner (vMF becomes uniform). 
        # You might set + 0, but +1 is also fine for numerical stability.
        kappa = F.softplus(self.fc_kappa(h)) + 1
        return mu, kappa

    def _sample_vMF_radial(self, kappa, dim):
        """
        Sample the radial component 'w' from the vMF distribution, 
        using an accept-reject approach. 
        This is one valid way to handle the 'radius' part in vMF sampling.
        
        Args:
            kappa: [batch_size]
            dim:   scalar (latent dimension, d)
        Returns:
            w: [batch_size], each in [-1, 1], used to scale the random direction.
        """
        w = torch.zeros_like(kappa, device=kappa.device)
        for i in range(kappa.shape[0]):
            k = kappa[i]
            done = False
            while not done:
                z = torch.rand(1, device=k.device)
                # stable approach
                z = z * (1 - torch.exp(-2*k)) + torch.exp(-2*k)
                w_temp = 1 - torch.log(z) / k
                u = torch.rand(1, device=k.device)
                # accept-reject test
                if torch.log(u) <= (dim - 3) * torch.log(w_temp) + k * w_temp:
                    w[i] = w_temp
                    done = True
        return w

    def _householder_rotation(self, v, mu):
        """
        Householder rotation that reflects v onto mu (assuming v is 
        orthonormal to mu). 
        Actually used here to 'rotate' a random direction v so that 
        its mean direction is mu.
        """
        # Reflect v across the plane orthogonal to (v - mu)
        u = F.normalize(v - mu, dim=-1)  # the reflection axis
        return v - 2.0 * (v * u).sum(-1, keepdim=True) * u

    def reparameterize(self, mu, kappa):
        """
        Reparam trick for vMF: 
        1) Sample a random direction v uniformly on S^{d-1}.
        2) Sample radial component w from accept-reject above.
        3) Reflect the direction so that its mean direction is mu.
        4) Scale by w.
        """
        batch_size = mu.shape[0]
        latent_dim = mu.shape[1]
        
        # 1) random direction on the unit sphere
        v = torch.randn(batch_size, latent_dim, device=mu.device)
        v = F.normalize(v, dim=-1)
        
        # 2) sample radial part
        w = self._sample_vMF_radial(kappa.squeeze(-1), latent_dim)  # shape [batch_size]
        
        # 3) householder rotate
        z = self._householder_rotation(v, mu)
        
        # 4) scale by w
        z = z * w.unsqueeze(-1)
        
        return z

    def forward(self, x):
        """
        Full forward pass: 
        1) encode => (mu, kappa)
        2) reparameterize => z
        3) MLP => final output
        """
        mu, kappa = self.encode(x)
        z = self.reparameterize(mu, kappa)
        output = self.mlp(z)
        return output, mu, kappa

    def kl_divergence(self, mu, kappa):
        """
        Computes KL( vMF(mu, kappa) || Uniform(S^{d-1}) ).
        
        The formula (batch-wise) is:
          KL = [ (d/2 - 1)*log(kappa) - (d/2)*log(2*pi) - log(I_{d/2 - 1}(kappa)) ]
                + kappa * m(kappa)
                + log( surface_area(S^{d-1}) )
        
        where:
          m(kappa) = I_{d/2}(kappa) / I_{d/2 - 1}(kappa)
          surface_area(S^{d-1}) = 2 pi^{d/2} / Gamma(d/2).
          
        We'll return mean over the batch.
        """
        # 1) dimension: points lie on S^{d-1}, so d = mu.shape[1]
        d = mu.shape[1]

        # 2) Flatten kappa from [B,1] -> [B]
        kappa = kappa.squeeze(-1)
        # clamp to avoid log(0)
        kappa = torch.clamp(kappa, min=1e-10)

        # 3) Define terms for the log normalizing constant of vMF
        #    logC_vMF = (d/2 - 1)*log(kappa) - (d/2)*log(2*pi) - log(I_{(d/2)-1}(kappa))
        half_d = d / 2.0
        nu = half_d - 1.0  # order for the Bessel function
        log_2pi = math.log(2.0 * math.pi)

        # log(I_{nu}(kappa)) 
        # NOTE: for large d,kappa, consider scaled bessel or asymptotics
        log_bessel_nu = torch.log(iv(nu, kappa) + 1e-40)

        logC_vMF = ((half_d - 1.0)*torch.log(kappa)
                    - half_d*log_2pi
                    - log_bessel_nu)

        # 4) The mean resultant length: m(kappa) = I_{d/2}(kappa) / I_{d/2 - 1}(kappa)
        #    We'll compute log(I_{d/2}(kappa)) - log(I_{d/2 - 1}(kappa)) => log(m(kappa))
        nu_plus = half_d
        log_bessel_nu_plus = torch.log(iv(nu_plus, kappa) + 1e-40)
        log_m_kappa = log_bessel_nu_plus - log_bessel_nu
        m_kappa = torch.exp(log_m_kappa)

        # So the 'expected log-likelihood' part under vMF is => logC_vMF + kappa * E[mu^T z] = logC_vMF + kappa*m_kappa

        # 5) Uniform on S^{d-1} => constant density = 1 / |S^{d-1}|.
        #    log p_unif = -log(|S^{d-1}|)
        #    where |S^{d-1}| = 2 pi^{d/2} / Gamma(d/2)
        #    => log_surface_area = log(2) + (d/2)*log(pi) - lgamma(d/2)
        log_surface_area = (
            math.log(2.0)
            + half_d * math.log(math.pi)
            - torch.lgamma(torch.tensor(half_d, device=kappa.device))
        )

        # 6) Full KL => E[log p_vMF(z)] - E[log p_unif(z)]
        #              = [logC_vMF + kappa*m_kappa] - [- log_surface_area]
        #              = logC_vMF + kappa*m_kappa + log_surface_area
        kl_batch = logC_vMF + kappa*m_kappa + log_surface_area

        return kl_batch.mean()
