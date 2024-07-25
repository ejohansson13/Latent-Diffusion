import torch
import numpy as np
from schedulers.util import make_beta_schedule, rescale_zero_terminal_snr

# Successful configurations:
# num_training_steps=(4000,6000), beta_start=1e-4, beta_end=2e-2; beta_schedule="cosine", rescale_betas_zero_snr=True

class EulerDiscreteSampler:
    def __init__(
        self,
        generator: torch.Generator,
        num_training_steps=4000, # 4000-6000 generated decent results with cosine schedule, beta_start=1e-4 and beta_end=2e-2
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
        #max_sigma = max(self.sigmas) # if self.sigmas is list
        # else 
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
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)

        sigmas = np.array( ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** (0.5))
        log_sigmas = np.log(sigmas)

        #if self.config.interpolation_type == "linear":
        sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
        #elif self.config.interpolation_type == "log_linear":
        #sigmas = torch.linspace(np.log(sigmas[-1]), np.log(sigmas[0]), num_inference_steps + 1).exp().numpy()

        # converting to karras sigmas
        sigma_min = sigmas[-1].item()
        sigma_max = sigmas[0].item()
        rho = 7.0
        min_inv_rho = sigma_min **(1 / rho)
        max_inv_rho = sigma_max **(1 / rho)
        sigmas = (max_inv_rho + np.linspace(0,1,num_inference_steps) * (min_inv_rho - max_inv_rho)) **rho

        timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas])
        self.timesteps = torch.from_numpy(timesteps.astype(np.float32))
        
        #sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32)
        sigma_last = ((1 - self.alphas_cumprod[0]) / self.alphas_cumprod[0]) ** 0.5
        #self.sigmas = torch.cat([sigmas, torch.zeros(1)]).to("cpu") # concat with sigma_last
        sigmas = np.concatenate([sigmas, [sigma_last]])
        self.sigmas = torch.from_numpy(sigmas).to("cpu", dtype=torch.float32)
        
        self._step_index = None
        self._begin_index = None

    def _sigma_to_t(
        self,
        sigma,
        log_sigmas,
    ):
        log_sigma = np.log(np.maximum(sigma, 1e-10))
        # get distribution
        dists = log_sigma - log_sigmas[:, np.newaxis]
        # get sigmas range
        low_idx = np.cumsum((dists >= 0), axis=0).argmax(axis=0).clip(max=log_sigmas.shape[0] - 2)
        high_idx = low_idx + 1
        low = log_sigmas[low_idx]
        high = log_sigmas[high_idx]

        # rescale sigmas
        w = (low - log_sigma) / (low - high)
        w = np.clip(w, 0, 1)
        # transform inerpolation to time range
        time_from_sigmas = (1 - w) * low_idx + w * high_idx
        time_from_sigmas = time_from_sigmas.reshape(sigma.shape)
        return time_from_sigmas
    
    def _init_step_index(
        self,
        timestep: int,
    ):
        # accounts for set_begin_index not being called from pipeline
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
        s_churn: float = 0.0, # [0.100]
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
    ):
        if self.step_index is None:
            self._init_step_index(timestep)
        
        # Can upcast to avoid precision issues
        # latents = latents.to(torch.float32)
        sigma = self.sigmas[self.step_index]
        gamma = min( s_churn / (len(self.sigmas)-1), 2**(0.5)-1 ) if s_tmin <= sigma <= s_tmax else 0.0
        rand_noise = torch.randn(ldm_output.shape, generator=self.generator, device=ldm_output.device, dtype=ldm_output.dtype)
        eps = rand_noise * s_noise
        sigma_hat = sigma * (gamma + 1)
        if gamma > 0:
            latents = latents + eps * (sigma_hat**2 - sigma**2) **(0.5)

        pred_x_zero = latents - sigma_hat * ldm_output
        deriv_x = (latents - pred_x_zero) / sigma_hat
        d_t = self.sigmas[self._step_index + 1] - sigma_hat

        prev_latent = latents + deriv_x * d_t
        prev_latent = prev_latent.to(ldm_output.dtype)
        self._step_index += 1
        
        return prev_latent