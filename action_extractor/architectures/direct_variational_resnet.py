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
    
############################################################
# 1) Optimized partial-sum approach for log(I_v(x))
############################################################

def log_i_v_series_optimized(v, x, max_iter=20):
    """
    Compute log(I_v(x)) using the series expansion:
       I_v(x) = sum_{k=0}^{max_iter-1} ( (x/2)^(2k+v ) ) / ( k! * Gamma(k+v+1 ) )
    in a memory-friendly partial summation manner.

    v, x : shape [B]
    Returns: shape [B] with log(I_v(x))
    """
    B = x.shape[0]
    device = x.device
    dtype = x.dtype

    # We'll accumulate the log-sum-exp in 'results'.
    # Initialize with log(0).
    results = torch.full((B,), float('-inf'), device=device, dtype=dtype)
    
    x_over_2 = x * 0.5
    eps = 1e-40  # for numeric clamps

    for k in range(max_iter):
        # exponent = (2k + v)*log(x/2)
        exponent = (2.0*k + v) * torch.log(torch.clamp(x_over_2, min=eps))

        # denominator = k! * Gamma(k+v+1)
        # => log_denom = lgamma(k+1) + lgamma(k+v+1)
        # We'll do them partially outside loop if we want; or inline:

        log_k_factor = math.lgamma(k+1.0)  # scalar
        # We'll do logGamma(k + v[b] + 1.0) as elementwise:
        log_gamma_vec = torch.lgamma(torch.tensor(k, device=device, dtype=dtype) + v + 1.0)

        log_term = exponent - (log_k_factor + log_gamma_vec)

        # results = logsumexp(results, log_term), do a stable in-place:
        max_val = torch.max(results, log_term)
        results = max_val + torch.log1p(
            torch.exp(results - max_val) + torch.exp(log_term - max_val)
        )

    return results

def log_i_v_asymptotic(v, x):
    """
    For large x, I_v(x) ~ exp(x) / sqrt(2*pi*x).
    log(I_v(x)) ~ x - 0.5*log(2*pi*x).
    (Ignoring v's effect for large x, or do a slightly refined version.)
    """
    eps = 1e-40
    return x - 0.5 * torch.log(2.0*math.pi*torch.clamp(x, min=eps))

############################################################
# 2) Single function that does partial-sum for smaller kappa,
#    and asymptotic approximation for large kappa.
############################################################

def log_i_v_mixed(v, x, max_iter=20, kappa_thresh=50.0):
    """
    Piecewise approach:
      If x > kappa_thresh, use asymptotic approx,
      else use partial-sum expansion.
    """
    # We do separate calls:
    large_mask = (x > kappa_thresh)
    log_series = log_i_v_series_optimized(v, x, max_iter=max_iter)
    log_asympt = log_i_v_asymptotic(v, x)
    return torch.where(large_mask, log_asympt, log_series)

############################################################
# 3) Householder-based rotation + Wood's algorithm for vMF
############################################################

def wood_sample_vMF(mu, kappa, max_tries=10):
    """
    Vectorized Wood's sampling of vMF(mu,kappa).
    mu: [B, d], kappa: [B] or [B,1].
    """
    device = mu.device
    B, d = mu.shape
    kappa = kappa.view(B)

    alpha = d - 1.0  # = d-1
    accepted = torch.zeros(B, dtype=torch.bool, device=device)
    w = torch.zeros(B, device=device)

    for _ in range(max_tries):
        # sample w in [-1,1]
        w_cand = 2.0*torch.rand(B, device=device) - 1.0
        one_minus_w2 = 1.0 - w_cand*w_cand
        one_minus_w2 = torch.clamp(one_minus_w2, min=1e-40)

        log_p = kappa*w_cand + 0.5*(alpha-2.0)*torch.log(one_minus_w2)
        # accept if log(r) + kappa < log_p
        log_r = torch.log(torch.rand(B, device=device) + 1e-40)
        newly_accepted = ((log_r + kappa) <= log_p) & (~accepted)

        w[newly_accepted] = w_cand[newly_accepted]
        accepted = accepted | newly_accepted
        if accepted.all():
            break

    w = torch.clamp(w, -1.0, 1.0)

    # sample direction in R^{d-1}
    v = torch.randn(B, d-1, device=device)
    v = F.normalize(v, dim=-1)
    sqrt_term = torch.sqrt(torch.clamp(1.0 - w*w, min=1e-40))

    # embed in [B,d]
    z_tilde = torch.zeros(B, d, device=device)
    z_tilde[:, 0:(d-1)] = (sqrt_term.unsqueeze(-1)*v)
    z_tilde[:, d-1] = w

    # Householder to rotate e_d => mu
    e_d = torch.zeros(d, device=device)
    e_d[-1] = 1.0
    e_d_batch = e_d.unsqueeze(0).expand(B, d)
    # reflection axis
    u = F.normalize(e_d_batch - mu, dim=-1)
    dot_val = (z_tilde*u).sum(dim=-1, keepdim=True)
    z = z_tilde - 2.0*dot_val*u
    return z

def sample_vMF_rejection(mu, kappa, max_tries=20):
    """
    Alternate approach using your radial vectorized acceptance.
    """
    B, d = mu.shape
    device = mu.device
    v = torch.randn(B, d, device=device)
    v = F.normalize(v, dim=-1)

    # radial accept-reject
    # vectorized approach for w:
    w = torch.zeros(B, device=device)
    accepted = torch.zeros(B, dtype=torch.bool, device=device)
    kappa = kappa.view(-1)

    for _ in range(max_tries):
        z = torch.rand(B, device=device)
        # stable approach
        # z in [exp(-2k), 1], ...
        z = z*(1.0 - torch.exp(-2*kappa)) + torch.exp(-2*kappa)
        w_temp = 1.0 - torch.log(z)/kappa

        u = torch.rand(B, device=device)
        test_log = (d-3.0)*torch.log(torch.clamp(w_temp, min=1e-40)) + kappa*w_temp
        compare_log = torch.log(torch.clamp(u, min=1e-40))
        newly = (compare_log <= test_log) & (~accepted)
        w[newly] = w_temp[newly]
        accepted = accepted | newly
        if accepted.all():
            break

    # rotate
    z_tilde = v  # shape [B,d]
    # Householder rotate z_tilde => direction mu
    # but we also scale by w
    # single reflection: we want z after reflection to have length w
    # Actually simpler: first reflect, then scale
    e_d = torch.zeros(d, device=device)
    e_d[-1] = 1.0
    e_d_batch = e_d.unsqueeze(0).expand(B, d)
    u = F.normalize(e_d_batch - mu, dim=-1)
    dot_val = (z_tilde*u).sum(dim=-1, keepdim=True)
    z_rot = z_tilde - 2.0*dot_val*u

    # scale by w
    z = z_rot*(w.unsqueeze(-1))
    return z

############################################################
# 4) The main HypersphericalResNet class (optimized)
############################################################

class ActionExtractionHypersphericalResNet(nn.Module):
    def __init__(self, 
                 resnet_version='resnet18',
                 video_length=2,
                 in_channels=3,
                 latent_dim=32,
                 action_length=1,
                 num_classes=7,
                 num_mlp_layers=3,
                 bessel_max_iter=20,
                 # reduce iteration
                 vMF_sample_method='wood',
                 max_tries_sampling=10,
                 approximate_bessel=True,
                 kappa_thresh=50.0):
        super().__init__()

        self.latent_dim = latent_dim
        self.bessel_max_iter = bessel_max_iter
        self.vMF_sample_method = vMF_sample_method.lower()
        self.max_tries_sampling = max_tries_sampling
        self.approximate_bessel = approximate_bessel
        self.kappa_thresh = kappa_thresh

        # Build conv backbone (just a placeholder for demonstration)
        # ResNet builder you'd define similarly, or a function you already have:
        self.conv, resnet_out_dim = resnet_builder(
            resnet_version=resnet_version, 
            video_length=video_length,
            in_channels=in_channels
        )

        self.fc_mu = nn.Linear(resnet_out_dim, latent_dim)
        self.fc_kappa = nn.Linear(resnet_out_dim, 1)

        # Suppose we define an MLP similarly
        self.mlp = ResNetMLP(
            input_size=latent_dim,
            hidden_size=512,
            final_size=32,
            output_size=num_classes*action_length,
            num_layers=num_mlp_layers
        )

    def encode(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        mu = F.normalize(self.fc_mu(h), dim=-1)
        kappa = F.softplus(self.fc_kappa(h)) + 1.0
        return mu, kappa

    def reparameterize(self, mu, kappa):
        if self.vMF_sample_method == 'wood':
            # Wood's algorithm
            z = wood_sample_vMF(mu, kappa, max_tries=self.max_tries_sampling)
        else:
            # rejection approach
            z = sample_vMF_rejection(mu, kappa, max_tries=self.max_tries_sampling)
        return z

    def forward(self, x):
        mu, kappa = self.encode(x)
        z = self.reparameterize(mu, kappa)
        out = self.mlp(z)
        return out, mu, kappa

    def kl_divergence(self, mu, kappa):
        """
        KL( vMF(mu, kappa) || Uniform(S^{d-1}) ).
        dimension d = mu.shape[1].
        """
        d = mu.shape[1]
        half_d = d/2.0
        nu = half_d - 1.0

        kappa = torch.clamp(kappa.squeeze(-1), min=1e-10)

        # 1) log(I_{nu}(kappa)), log(I_{nu+1}(kappa))
        log_i_nu = log_i_v_mixed(
            v=torch.full_like(kappa, nu),
            x=kappa,
            max_iter=self.bessel_max_iter,
            kappa_thresh=self.kappa_thresh
        )
        log_i_nu_plus = log_i_v_mixed(
            v=torch.full_like(kappa, nu+1.0),
            x=kappa,
            max_iter=self.bessel_max_iter,
            kappa_thresh=self.kappa_thresh
        )

        # logC_vMF
        log_2pi = math.log(2.0*math.pi)
        logC_vMF = ((half_d - 1.0)*torch.log(kappa)
                    - half_d*log_2pi
                    - log_i_nu)

        # m(kappa) = I_{nu+1}(kappa)/I_{nu}(kappa)
        log_m_kappa = log_i_nu_plus - log_i_nu
        m_kappa = torch.exp(log_m_kappa)

        # surface area of S^{d-1}, log
        # = log(2) + (d/2)*log(pi) - logGamma(d/2)
        log_surface = (math.log(2.0)
                       + half_d*math.log(math.pi)
                       - torch.lgamma(torch.tensor(half_d, device=kappa.device, dtype=kappa.dtype)))

        kl_vals = logC_vMF + kappa*m_kappa + log_surface
        return kl_vals.mean()