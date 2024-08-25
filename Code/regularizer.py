import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class KLRegularizer():
    def __init__(
        self,
        encoded_moments,
        scaling_factor = 0.18215
        #deterministic=False
    ):
        """
        Regularizer for latent space. Applies rescaling factor to stabilize latent space.
        Expects moments of encoded image.
        Returns posterior of encoded image.
        """
        
        self.encoded_moments = encoded_moments
        self.mean, self.logvar = torch.chunk(encoded_moments, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.var = torch.exp(self.logvar)
        self.std = torch.exp(0.5 * self.logvar)
        self.scaling_factor = scaling_factor

    def sample(self):
        sampleReg = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.encoded_moments.device)
        # historical constant from SD repo -> works well, but can recalculate separate constant for specific dataset
        # recalculation optionality is in train_vae.py
        sampleReg *= self.scaling_factor
        return sampleReg
    
    def mode(self):
        mean = self.mean * self.scaling_factor
        return mean
    
    def kl(
        self,
        other=None,
    ):
        if other is None:
            return 0.5 * torch.sum(torch.pow(self.mean, 2)
                    + self.var - 1.0 - self.logvar,
                    dim=[1, 2, 3])
        else:
            return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])
        
    def nll(
        self,
        sample,
        dims=[1,2,3]
    ):
        return 0.5 * torch.sum(
            np.log(2.0 * np.pi) + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)
    

def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    source: https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/losses.py#L12
    Compute the KL divergence between two gaussians.
    Shapes are automatically broadcasted, so batches can be compared to scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )