import torch
import numpy as np
import math

def make_beta_schedule(
    num_training_steps=1000,
    beta_schedule="linear",
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
):
    if beta_schedule == "linear":
        betas = torch.linspace(beta_start, beta_end, num_training_steps, dtype=torch.float32)
    # could add optionality for scaled-linear
        # same as "linear" optionality but (sqrt of beta_start and beta_end) **2
    elif beta_schedule == "cosine":
        betas = []
        beta_max = 0.999
        for i in range(num_training_steps):
            t1 = i / num_training_steps
            t1 = cosine_for_timestep(t1)
            t2 = (i+1) / num_training_steps
            t2 = cosine_for_timestep(t2)
            betas.append( min(1 - t2 / t1, beta_max) )

    return np.array(betas)


def cosine_for_timestep(
    timestep,
    cosine_s = 8e-3
):
    return math.cos( (timestep + cosine_s) / (1 + cosine_s) * math.pi / 2 )**2


# Rescales betas to have zero terminal SNR Based on https://arxiv.org/pdf/2305.08891.pdf (Algorithm 1)
def rescale_zero_terminal_snr(betas):
    # Convert betas to alphas_bar_sqrt
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(torch.from_numpy(alphas), dim=0)
    alphas_bar_sqrt = alphas_cumprod.sqrt()

    # Store old values
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so the last timestep is zero
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # Scale so the first timestep is back to the old value
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt**2  # Revert sqrt
    alphas = alphas_bar[1:] / alphas_bar[:-1]  # Revert cumprod
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas

    return betas