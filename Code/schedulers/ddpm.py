import torch
import numpy as np
import math
from schedulers.util import make_beta_schedule, rescale_zero_terminal_snr

class DDPMSampler:
    def __init__(
        self,
        generator: torch.Generator,
        num_training_steps=1000,
        beta_schedule="cosine",
        beta_start: float = 1e-4,
        beta_end: float = 1e-2,
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
        self.generator = generator

        self.num_training_timesteps = num_training_steps
        self.timesteps = torch.from_numpy(np.arange(0,num_training_steps)[::-1].copy())
    
    def _get_variance(
        self,
        timestep: int,
    ):
        prev_step = self._previous_timestep(timestep)
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_step] if prev_step >= 0 else torch.tensor(1.0)
        curr_alpha_t = alpha_prod_t / alpha_prod_t_prev
        curr_beta_t = 1 - curr_alpha_t

        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * curr_beta_t
        variance = torch.clamp(variance, min=1e-20)
        return variance

    def _previous_timestep(
        self,
        timestep: int,
    ):
      # return timestep - self.step_ratio # need to determine if function will be called while training
      return timestep - self.num_training_timesteps // self.num_inference_steps


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
    ):
        prev_step = self._previous_timestep(timestep)
        
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_step] if prev_step >= 0 else torch.tensor(1.0)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        curr_alpha_t = alpha_prod_t / alpha_prod_t_prev
        curr_beta_t = 1 - curr_alpha_t

        pred_x_zero = (latents - beta_prod_t **(0.5) * ldm_output) / alpha_prod_t **(0.5)
        # Determine coefficients for mu from Formula 7
        x_zero_coefficient = (alpha_prod_t_prev **(0.5) * curr_beta_t) / beta_prod_t # signal scaling, prevent exploding variances
        x_t_coefficient = (curr_alpha_t **(0.5) * beta_prod_t_prev) / beta_prod_t # variance scaling

        pred_mu_t = x_zero_coefficient * pred_x_zero + x_t_coefficient * latents

        variance = 0
        if timestep > 0:
            device = ldm_output.device
            variance_noise = torch.randn(ldm_output.shape, generator=self.generator, device=device, dtype=ldm_output.dtype)
            variance = (self._get_variance(timestep) **0.5) * variance_noise
        
        return pred_mu_t + variance