import torch
import numpy as np
from schedulers.util import make_beta_schedule, rescale_zero_terminal_snr

class EulerAncestralSampler:
    def __init__(
        self,
        generator: torch.Generator,
        num_training_steps=1000, # 4000-6000 generated decent results with cosine schedule, beta_start=1e-4 and beta_end=2e-2
        beta_schedule="linear",
        beta_start: float = 1e-4, # 1e-4,
        beta_end: float = 1e-2, # 1e-2,
        rescale_betas_zero_snr = False,
    ):
        assert beta_schedule in ["linear", "cosine"], f"Schedule argument {beta_schedule} is not a supported choice of linear or cosine"
        betas = make_beta_schedule(num_training_steps, beta_schedule, beta_start, beta_end)
        if rescale_betas_zero_snr:
            betas = rescale_zero_terminal_snr(betas)

        self.betas = betas
        alphas = 1.0 - self.betas
        self.alphas = alphas if rescale_betas_zero_snr else torch.from_numpy(alphas)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        if rescale_betas_zero_snr:
            self.alphas_cumprod[-1] = 2**(-24)

        sigmas = (( (1 - self.alphas_cumprod) / self.alphas_cumprod) **(0.5)).flip(0)
        sigmas = torch.cat([sigmas, torch.zeros(1)])
        self.sigmas = sigmas.to("cpu")

        timesteps = np.linspace(0, num_training_steps-1, num_training_steps, dtype=float)[::-1].copy()
        self.timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)
        self.num_training_steps = num_training_steps
        self.generator = generator

        self._step_index = None
        self._begin_index = None

    def init_noise_sigma(
        self,
    ):
        max_sigma = self.sigmas.max()
        return (max_sigma**2 + 1) ** (0.5)
    
    @property
    def step_index(
        self,
    ):
        return self._step_index

    @property
    def begin_index(
        self,
    ):
        return self._begin_index
    
    def set_begin_index(
        self,
        begin_index:int = 0,
    ):
        self._begin_index = begin_index

    def scale_model_input(
        self,
        latent,
        timestep: int,
    ):
        if self.step_index is None:
            self._init_step_index(timestep)

        sigma = self.sigmas[self.step_index]
        latent = latent / ((sigma**2 + 1) ** (0.5))

        return latent
    
    def set_inference_timesteps(
        self,
        num_inference_steps: int = 20,
    ):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_training_steps // self.num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.float32) # euler has np.int64

        sigmas = np.array( ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** (0.5))
        #"linear" interpolation type:
        sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
        sigmas = torch.from_numpy(sigmas).to(torch.float32)
        sigmas = torch.cat([sigmas, torch.zeros(1)]) # concat with sigma_last
        self.sigmas = sigmas.to("cpu")

        self.timesteps = torch.from_numpy(timesteps).to("cpu")
        self._step_index = None
        self._begin_index = None

    def _init_step_index(
        self,
        timestep: int,
    ):
        if self.begin_index is None:
            # self.index_for_timestep(timestep)
            indices = (timestep == self.timesteps).nonzero()
            # primary concern seems to be image-to-image, can directly assign this to 0 later
            pos = 1 if len(indices) > 1 else 0
            self._step_index = indices[pos].item()
        else:
            self._step_index = self._begin_index
    
    def step(
        self,
        latents: torch.Tensor,
        timestep: int,
        ldm_output: torch.Tensor,
    ):
        if self.step_index is None:
            self._init_step_index(timestep)
        
        # Can upcast to avoid precision issues
        # latents = latents.to(torch.float32)
        sigma = self.sigmas[self.step_index]

        # pred_original_sample = sample - sigma * model_output
        pred_x_zero = latents - sigma * ldm_output
        sigma_cur = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]
        sigma_up = (sigma_next**2 * (sigma_cur**2 - sigma_next**2) / sigma_cur**2) ** (0.5)
        sigma_down = (sigma_next**2 - sigma_up**2) ** (0.5)

        deriv_x = (latents - pred_x_zero) / sigma
        d_t = sigma_down - sigma
        prev_latent = latents + deriv_x * d_t
        #prev_latent = prev_latent.to(ldm_output.dtype)

        rand_noise = torch.randn(ldm_output.shape, generator=self.generator, device=ldm_output.device, dtype=ldm_output.dtype)
        prev_latent = prev_latent + rand_noise * sigma_up
        prev_latent = prev_latent.to(ldm_output.dtype)
        self._step_index += 1

        return prev_latent