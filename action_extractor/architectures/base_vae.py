import torch
import torch.nn as nn

class BaseVAE(nn.Module):
    def kl_divergence(self, *args):
        raise NotImplementedError