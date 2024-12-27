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
                 bessel_max_iter=50,
                 vMF_sample_method='wood'):
        """
        bessel_max_iter: how many terms to use for the series expansion of I_v(x).
                         Increase for more accuracy if x or v can be large.
        """
        super(ActionExtractionHypersphericalResNet, self).__init__()
        self.latent_dim = latent_dim
        self.bessel_max_iter = bessel_max_iter
        
        self.vMF_sample_method = vMF_sample_method.lower()

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

    def _sample_vMF_radial_vectorized(self, kappa, dim, max_tries=100):
        """
        Vectorized accept-reject for w in a single batch pass on the GPU.
        Args:
        kappa: [B] or [B,1]
        dim:   scalar
        max_tries: cap the loop to prevent infinite runs if acceptance is very low.

        Returns:
        w: [B]
        """
        device = kappa.device
        kappa = kappa.view(-1)  # [B]
        B = kappa.shape[0]

        w = torch.zeros(B, device=device)
        accepted = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_tries):
            # Draw all candidates at once
            z = torch.rand(B, device=device)
            # stable approach
            z = z * (1 - torch.exp(-2*kappa)) + torch.exp(-2*kappa)
            w_temp = 1.0 - torch.log(z) / kappa

            u = torch.rand(B, device=device)
            # accept-reject criterion
            test_log = (dim - 3) * torch.log(w_temp) + kappa * w_temp
            compare_log = torch.log(u)

            # mask of newly accepted
            newly_accepted = compare_log <= test_log
            # keep only for those not previously accepted
            newly_accepted = newly_accepted & (~accepted)

            # update w for newly accepted
            w[newly_accepted] = w_temp[newly_accepted]
            # mark them as accepted
            accepted = accepted | newly_accepted

            # break if all accepted
            if accepted.all():
                break

        # any that *never* got accepted remain zero; depending on your domain,
        # you could clamp or handle them. Usually, the acceptance rate is high enough.
        return w
    
    # ----------------------------------------------------------------------
    #  1) Vectorized Wood's Algorithm for vMF sampling (no per-sample loop)
    # ----------------------------------------------------------------------
    def _wood_sample_vMF(self, mu, kappa):
        """
        Vectorized sampling of z ~ vMF(mu, kappa) in R^d using Wood's algorithm.
        Returns: z, shape [B, d].
        
        Steps (batch of size B):
          1) Let d = mu.shape[1].
          2) We sample w in [-1,1] from a 1D distribution ~ e^{kappa w}(1-w^2)^{(d-3)/2}.
             This is done via a small accept-reject in batch form.
          3) Sample an orthonormal direction in R^{d-1}.
          4) Embed [w, sqrt(1-w^2)*v] into R^d.
          5) Rotate to match the mean direction mu in R^d.
        """
        device = mu.device
        B, d = mu.shape
        kappa = kappa.view(B)

        # We'll do an accept-reject for w, but fully vectorized:
        # pdf(w) ~ e^{kappa w} (1-w^2)^{(d-3)/2}, w in [-1,1].
        # We'll do a loop up to some max tries, each time generating "candidates".

        max_tries = 50  # you can adjust if acceptance is low
        accepted = torch.zeros(B, dtype=torch.bool, device=device)
        w = torch.zeros(B, device=device)

        alpha = d - 1.0  # = (d-1)

        for _ in range(max_tries):
            # uniform candidate in [-1,1]
            w_cand = 2.0*torch.rand(B, device=device) - 1.0
            # compute log pdf: 
            #   log p(w_cand) ~ kappa * w_cand + ((d-3)/2) * log(1 - w_cand^2)
            # We'll do shape-protection for d=2 => then (d-3)/2 = -0.5, still works.
            one_minus_w2 = 1.0 - w_cand*w_cand
            # avoid log(0)
            one_minus_w2 = torch.clamp(one_minus_w2, min=1e-40)
            log_p = kappa * w_cand + (0.5*(alpha - 2.0)) * torch.log(one_minus_w2)

            # we don't need the normalizing constant for accept-reject
            # just find max of log p theoretically. The maximum is around w=1 for large kappa,
            # but let's be safe. We'll sample u in [0,1].
            # define a bounding function M s.t. log_u < log_p - logM or so.
            # for simplicity, let's define M = 1 for all w => we need the max pdf < 1 => not trivial
            # Instead, we can do a ratio approach. We'll do a second approach:
            #  We'll compare log_u to [log_p - log_p_max].
            # But we need an upper bound. A simpler approach is:
            #   for each w_cand, also generate p_cand (some uniform(0, 1) scaled by the peak pdf).
            # This is "self-normalized" accept-reject:
            #   We sample another uniform(0,1), accept if log(u) < log_p - c, where c is a "shift".
            #
            # We'll shift by the max log p in the batch to keep stability, but we only do local AR.
            # This is standard: accept if u < exp(log_p - max_log_p), for a batch-based approach.

            # shift for numeric stability
            # But each sample might have a different max...
            # We'll do a single "global" shift that won't break correctness, but might reduce acceptance. 
            # Or do it individually per sample.
            # For best vectorization, do it per sample:
            max_log_p_cand = log_p  # since we do sample by sample anyway.
            # we can just do accept if u < exp(log_p_cand - log_p_cand) => exp(0) => 1 => trivial. That doesn't help.

            # Let's do a small bounding trick. A simpler approach is we can compare 
            #   log(u) with log_p - A
            # for some big A that ensures acceptance < 1 always. 
            # Or we skip trying to find an explicit M, and do:
            #   "We guess an upper bound. If that fails, we can fallback or just take the ratio approach."
            #
            # Actually, let's do a "self-normalized" approach: we also sample log v. If log v <= log_p - c, accept.

            # We'll pick c = 0 for simplicity, then we need p(w) <= 1 for all w, which might not hold if kappa large.
            # Let's do c = (kappa*1 + 0.5*(alpha-2)*log(1 - 1^2))?? That doesn't make sense. 
            # 
            # Let's just do a "secondary" random uniform r in [0,1], and accept if:
            #   r <= exp( log_p - max_possible_log_p )
            #
            # The maximum possible log_p might be near w=1 for large kappa. Let's estimate it as w=1 => 
            #   log_p_max ~ kappa*(1) + (0.5*(alpha-2))*log(1-1^2) => second term is log(0) => -inf => not good.
            # Actually the "peak" is typically near w=1 if kappa is large, or near w=0 if kappa=0, etc.
            #
            # We'll do a simpler bounding approach:
            #   log_p <= kappa + 0? Actually might still be negative. 
            #   We can be liberal and pick a bound like M = exp(kappa). 
            #
            # So we do accept if log(r) < log_p - kappa => log(r) + kappa < log_p.

            log_r = torch.log(torch.rand(B, device=device) + 1e-40)
            # accept if log_r + kappa < log_p
            accept_mask = (log_r + kappa) <= log_p

            newly_accepted = accept_mask & (~accepted)
            w[newly_accepted] = w_cand[newly_accepted]
            accepted = accepted | newly_accepted

            if accepted.all():
                break

        # Now w in [-1,1]. Next step: sample random direction in R^{d-1}.
        # We'll do that by sampling standard normal [B, d-1], normalizing.
        # Then embed as last coordinate = w, the rest is sqrt(1 - w^2) * v.

        w_clamped = torch.clamp(w, -1.0, 1.0)  # just to be safe
        v = torch.randn(B, d-1, device=device)
        v = F.normalize(v, dim=-1)  # each row is a direction in R^{d-1}

        # shape [B], sqrt(1 - w^2)
        sqrt_term = torch.sqrt(1.0 - w_clamped.pow(2).clamp(min=1e-40))
        # embed: we want a [B, d] vector where the last coordinate is w, 
        #        the first (d-1) coordinates are sqrt(1-w^2) * v
        # We'll store that in a new tensor "z_tilde"
        z_tilde = torch.zeros(B, d, device=device)
        # first d-1 coords
        z_tilde[:, 0:(d-1)] = (sqrt_term.unsqueeze(-1) * v)
        # last coord
        z_tilde[:, d-1] = w_clamped

        # Finally, we rotate z_tilde so that its "mean direction" is mu.
        # We'll do a Householder-based reflection that sends e_d -> mu, but we can also do:
        # "Reflect z_tilde so that [0,...,0,1] goes to mu."

        # Quick approach: note that z_tilde is currently oriented so that the "mean direction" 
        # in R^d is the last axis. We want to rotate that last axis to mu.
        # We'll do a Householder from e_d to mu. 
        # e_d = [0,0,...,1], so we'll define e_d - mu, reflect z_tilde across that plane.

        e_d = torch.zeros(d, device=device)
        e_d[-1] = 1.0
        # expand to batch
        e_d_batch = e_d.unsqueeze(0).expand(B, d)
        # reflection axis
        u = F.normalize(e_d_batch - mu, dim=-1)  # shape [B, d]

        # reflect z_tilde across that plane
        # z = z_tilde - 2 (z_tilde * u) u
        dot_val = (z_tilde * u).sum(dim=-1, keepdim=True)
        z = z_tilde - 2.0 * dot_val * u
        # z now ~ vMF(mu, kappa)

        return z
    
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
        if self.vMF_sample_method == 'wood':
            return self._wood_sample_vMF(mu, kappa)
        
        elif self.vMF_sample_method == 'rejection':
            batch_size = mu.shape[0]
            dim = mu.shape[1]

            # 1) random direction on the sphere
            v = torch.randn(batch_size, dim, device=mu.device)
            v = F.normalize(v, dim=-1)

            # 2) radial part
            kappa_1d = kappa.squeeze(-1)  # shape [B]
            w = self._sample_vMF_radial_vectorized(kappa_1d, dim)

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