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


# --------------------------------------------------------------------------
# 1) Series-based approximation of log(I_v(x)) in pure PyTorch (GPU-friendly)
#
#    I_v(x) = \sum_{k=0}^\infty  \frac{(x/2)^{2k+v}}{k! \Gamma(k+v+1)}
#
#  We compute log(I_v(x)) by partial summation + log-sum-exp.
#  This is typically okay for moderate x, v. For large x or v, consider
#  scaled Bessel or asymptotic expansions to avoid overflow.

def log_i_v_series(v, x, max_iter=50):
    """
    Compute log(I_v(x)) via the series expansion, fully on the GPU.

    Args:
      v:  [B]  (can be float, e.g. v = d/2 - 1)
      x:  [B]  (kappa >= 0)
      max_iter: how many terms in the series expansion

    Returns:
      log_I_v: [B], log of the Bessel function I_v(x).
    """
    # Both v, x are shape [B]. We'll create a [max_iter, B] mesh
    B = x.shape[0]
    device = x.device
    dtype = x.dtype

    # k in [0, 1, 2, ..., max_iter-1]
    k_arange = torch.arange(max_iter, device=device, dtype=dtype)  # shape [max_iter]
    # Expand to [max_iter, B]
    k_mat = k_arange.unsqueeze(1).expand(max_iter, B)       # shape [max_iter, B]
    v_mat = v.unsqueeze(0).expand(max_iter, B)              # shape [max_iter, B]
    x_mat = x.unsqueeze(0).expand(max_iter, B)              # shape [max_iter, B]

    # (x/2)^(2k + v) => (2k + v)* log(x/2)
    # We'll do everything in log space to avoid large exponentials
    log_x_over_2 = torch.log(x_mat/2 + 1e-40)  # shape [max_iter, B]
    logTerm_top = (2*k_mat + v_mat) * log_x_over_2  # shape [max_iter, B]

    # Denominator: k! * Gamma(k+v+1)
    # => log(k!) + logGamma(k+v+1)
    # k! = Gamma(k+1)
    lgamma_k_plus_1 = torch.lgamma(k_mat + 1.0)
    lgamma_k_plus_v_plus_1 = torch.lgamma(k_mat + v_mat + 1.0)
    logTerm_bot = lgamma_k_plus_1 + lgamma_k_plus_v_plus_1

    # So the log of each term is:
    #   logTerm = (2k+v)*log(x/2) - [log(k!) + logGamma(k+v+1)]
    logTerm = logTerm_top - logTerm_bot

    # sum in exp space => log-sum-exp over dimension=0 (the k dimension)
    lse = torch.logsumexp(logTerm, dim=0)  # shape [B]
    return lse  # log(I_v(x))


def i_v_series(v, x, max_iter=50):
    """
    Return I_v(x) by exponentiating log_i_v_series(...).
    May overflow if x, v are large. In practice, consider scaled versions.
    """
    return torch.exp(log_i_v_series(v, x, max_iter=max_iter))


# --------------------------------------------------------------------------
# 2) Final HypersphericalResNet with pure-PyTorch Bessel for large-batch GPU

class ActionExtractionHypersphericalResNet(BaseVAE):
    def __init__(self, 
                 resnet_version='resnet18', 
                 video_length=2, 
                 in_channels=3, 
                 latent_dim=32, 
                 action_length=1, 
                 num_classes=7, 
                 num_mlp_layers=3,
                 bessel_max_iter=50):
        """
        bessel_max_iter: how many terms to use for the series expansion of I_v(x).
                         Increase for more accuracy if x or v can be large.
        """
        super(ActionExtractionHypersphericalResNet, self).__init__()
        self.latent_dim = latent_dim
        self.bessel_max_iter = bessel_max_iter

        # Build the ResNet backbone
        self.conv, resnet_out_dim = resnet_builder(
            resnet_version=resnet_version, 
            video_length=video_length, 
            in_channels=in_channels
        )

        # Encoder outputs for mean direction and concentration
        self.fc_mu = nn.Linear(resnet_out_dim, latent_dim)
        self.fc_kappa = nn.Linear(resnet_out_dim, 1)
        
        # MLP to map latent z -> final action output
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
        # mu in R^d, forced to unit vector
        mu = F.normalize(self.fc_mu(h), dim=-1)
        # kappa >= 1 to avoid kappa=0 (vMF ~ uniform). 
        # +1 helps ensure a stable positivity
        kappa = F.softplus(self.fc_kappa(h)) + 1
        return mu, kappa

    def _sample_vMF_radial(self, kappa, dim):
        """
        Sample the radial part 'w' from vMF with accept-reject.
        Vectorizing this accept-reject can be tricky if success rates differ,
        so we do a python loop. For large batches, consider a vectorized approach
        or a different sampling method.
        """
        w = torch.zeros_like(kappa)
        for i in range(kappa.shape[0]):
            k = kappa[i]
            done = False
            while not done:
                z = torch.rand(1, device=k.device)
                z = z * (1 - torch.exp(-2*k)) + torch.exp(-2*k)
                w_temp = 1.0 - torch.log(z) / k
                u = torch.rand(1, device=k.device)
                if torch.log(u) <= (dim - 3) * torch.log(w_temp) + k * w_temp:
                    w[i] = w_temp
                    done = True
        return w

    def _householder_rotation(self, v, mu):
        """
        Householder reflection to rotate v onto mu.
        """
        u = F.normalize(v - mu, dim=-1)  # reflection axis
        # reflect v across the plane orthogonal to u
        return v - 2.0 * (v * u).sum(-1, keepdim=True) * u

    def reparameterize(self, mu, kappa):
        """
        Reparam trick for vMF:
          1) sample v ~ Uniform(S^{d-1})
          2) sample radial part w
          3) reflect v so it has mean direction mu
          4) scale by w
        """
        batch_size = mu.shape[0]
        dim = mu.shape[1]

        # 1) random direction on the sphere
        v = torch.randn(batch_size, dim, device=mu.device)
        v = F.normalize(v, dim=-1)

        # 2) radial part
        kappa_1d = kappa.squeeze(-1)  # shape [B]
        w = self._sample_vMF_radial(kappa_1d, dim)

        # 3) rotate
        z = self._householder_rotation(v, mu)

        # 4) scale
        z = z * w.unsqueeze(-1)

        return z

    def forward(self, x):
        mu, kappa = self.encode(x)
        z = self.reparameterize(mu, kappa)
        out = self.mlp(z)
        return out, mu, kappa

    def kl_divergence(self, mu, kappa):
        """
        KL( vMF(mu, kappa) || Uniform(S^{d-1}) )

        In dimension d = mu.shape[1], the formula is:

          KL = [ (d/2 - 1)*log(kappa) - (d/2)*log(2*pi) - log(I_{(d/2)-1}(kappa)) ]
                + kappa * m(kappa)
                + log( surface_area(S^{d-1}) ),

          where m(kappa) = I_{d/2}(kappa) / I_{d/2 - 1}(kappa),
          surface_area(S^{d-1}) = 2 * pi^{d/2} / Gamma(d/2).

        We'll compute I_{(d/2)-1}(kappa) and I_{d/2}(kappa) via i_v_series(...).
        """
        d = mu.shape[1]
        kappa = kappa.squeeze(-1)  # [B]
        kappa = torch.clamp(kappa, min=1e-10)  # avoid log(0)

        half_d = d / 2.0
        nu = half_d - 1.0  # i.e. (d/2) - 1
        log_2pi = math.log(2.0 * math.pi)

        # 1) log(I_{nu}(kappa))
        log_i_nu = log_i_v_series(
            v=torch.full_like(kappa, nu), 
            x=kappa, 
            max_iter=self.bessel_max_iter
        )

        # -> logC_vMF = (d/2 - 1)*log(kappa) - (d/2)*log(2*pi) - log(I_{nu}(kappa))
        logC_vMF = ( (half_d - 1.0) * torch.log(kappa)
                     - half_d * log_2pi
                     - log_i_nu )

        # 2) mean resultant length: m(kappa) = I_{d/2}(kappa) / I_{d/2 - 1}(kappa)
        #    => log(m(kappa)) = log(I_{d/2}(kappa)) - log(I_{d/2 - 1}(kappa))
        log_i_nu_plus = log_i_v_series(
            v=torch.full_like(kappa, half_d), 
            x=kappa, 
            max_iter=self.bessel_max_iter
        )
        log_m_kappa = log_i_nu_plus - log_i_nu
        m_kappa = torch.exp(log_m_kappa)

        # 3) log surface area of S^{d-1}
        #    = log(2) + (d/2)*log(pi) - logGamma(d/2)
        log_surface_area = (
            math.log(2.0)
            + half_d * math.log(math.pi)
            - torch.lgamma(torch.tensor(half_d, device=kappa.device, dtype=kappa.dtype))
        )

        # 4) Final KL
        #    = logC_vMF + kappa*m_kappa + log_surface_area
        kl_vals = logC_vMF + kappa * m_kappa + log_surface_area
        return kl_vals.mean()