import torch
import numpy as np
import math
from schedulers.util import make_beta_schedule, rescale_zero_terminal_snr

class DDIMSampler:
    def __init__(
        self,
        generator: torch.Generator,
        num_training_steps=1000,
        beta_schedule="cosine",
        beta_start: float = 1e-4,
        beta_end: float = 1e-2,
        set_final_alpha_to_one = False,
        rescale_betas_zero_snr = False,
    ):
        assert beta_schedule in ["linear", "cosine"], f"Schedule argument {beta_schedule} is not a supported choice of linear or cosine"
        betas = make_beta_schedule(num_training_steps, beta_schedule, beta_start, beta_end)
        if rescale_betas_zero_snr:
            betas = rescale_zero_terminal_snr(betas)

        self.betas = betas
        alphas = 1.0 - self.betas
        self.alphas = torch.from_numpy(alphas)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.final_alpha_cumprod = torch.tensor(1.0) if set_final_alpha_to_one else self.alphas_cumprod[0]
        self.generator = generator

        self.num_training_timesteps = num_training_steps
        self.timesteps = torch.from_numpy(np.arange(0,num_training_steps)[::-1].copy())

    def _get_variance(
        self,
        timestep: int,
        prev_timestep: int
    ):
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = ( beta_prod_t_prev / beta_prod_t) * ( 1 - alpha_prod_t / alpha_prod_t_prev )
        return variance

    def set_inference_timesteps(
        self,
        num_inference_steps: int = 20,
    ):
        if (num_inference_steps > self.num_training_timesteps):
            raise ValueError(
                f"Inference steps {num_inference_steps} cannot exceed {self.num_training_timesteps}"
            )
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_training_timesteps // self.num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)

    def step(
        self,
        latents: torch.Tensor,
        timestep: int,
        ldm_output: torch.Tensor,
        eta: float = 0.0
    ):
        # Notation (<variable name>   ->   <name in paper>)
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointing to x_t"
        # - pred_prev_sample -> "x_t-1"

        prev_timestep = timestep - self.num_training_timesteps // self.num_inference_steps

        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t

        pred_x_zero = (latents - beta_prod_t **(0.5) * ldm_output) / alpha_prod_t **(0.5)
        pred_epsilon = ldm_output
        # can put threshold sample call here

        variance = self._get_variance(timestep, prev_timestep)
        sigma_t = eta * variance **(0.5)

        pred_x_t_direction = ( 1 - alpha_prod_t_prev - sigma_t**2 ) **(0.5) * pred_epsilon
        prev_x_t = alpha_prod_t_prev **(0.5) * pred_x_zero + pred_x_t_direction

        # DDIM when eta == 0
        if eta > 0:
            device = ldm_output.device
            variance_noise = torch.randn(ldm_output.shape, generator=self.generator, device=device, dtype=ldm_output.dtype)
            variance = sigma_t * variance_noise
            prev_x_t = prev_x_t + variance

        return prev_x_t