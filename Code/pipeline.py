import torch
import numpy as np
from tqdm import tqdm

from schedulers.ddpm import DDPMSampler
from schedulers.ddim import DDIMSampler
from schedulers.euler_discrete import EulerDiscreteSampler
from schedulers.euler_ancestral import EulerAncestralSampler


WIDTH = 512
HEIGHT = 512
LATENT_FACTOR = 8
LATENT_WIDTH = WIDTH // LATENT_FACTOR
LATENT_HEIGHT = HEIGHT // LATENT_FACTOR

def generate(
    prompt,
    uc_prompt=None,
    use_cfg=True,
    cfg_scale=7,
    sampler_name="ddpm",
    sample_euler = False,
    num_inference_steps=50,
    models={},
    seed=None,
    device=None,
    idle_device=None,
    tokenizer=None,
):
    with torch.no_grad():
        
        if idle_device: to_idle = lambda x: x.to(idle_device)
        else: to_idle = lambda x: x

        generator = torch.Generator(device=device)
        if seed is None: 
            generator.seed()
        else: 
            generator.manual_seed(seed)

        clip = models["conditioner"].to(device)

        if use_cfg:
            cond_tokens = tokenizer.batch_encode_plus( [prompt], padding="max_length", max_length=77 ).input_ids
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            conditioning = clip(cond_tokens)
            uc_tokens = tokenizer.batch_encode_plus( [uc_prompt], padding="max_length", max_length=77 ).input_ids
            uc_tokens = torch.tensor(uc_tokens, dtype=torch.long, device=device)
            uc_conditioning = clip(uc_tokens)
            conditioning = torch.cat([conditioning, uc_conditioning])

        else:
            tokens = tokenizer.batch_encode_plus( [prompt], padding="max_length", max_length = 77 ).input_ids
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            conditioning = clip(tokens)
        to_idle(clip)
        #print("Conditioning has been satisifed.")

        latents_shape = (1, 4, HEIGHT//LATENT_FACTOR, WIDTH//LATENT_FACTOR)
        # below only offers latent initialization for text2img option
        latents = torch.randn(latents_shape, generator=generator, device=device)

        assert sampler_name in ["ddpm", "ddim", "euler", "euler a"], f"Unknown sampler: {sampler_name}"
        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(num_inference_steps)
        elif sampler_name == "ddim":
            sampler = DDIMSampler(generator)
            sampler.set_inference_timesteps(num_inference_steps)
        elif sampler_name == "euler":
            sample_euler = True
            sampler = EulerDiscreteSampler(generator, beta_schedule="cosine", rescale_betas_zero_snr=True)
            sampler.set_inference_timesteps(num_inference_steps)
            #sampler.set_begin_index()
            init_noise = sampler.init_noise_sigma()
            latents = latents * init_noise
        elif sampler_name == "euler a":
            sample_euler = True
            sampler = EulerAncestralSampler(generator, beta_schedule="cosine", rescale_betas_zero_snr=True)
            sampler.set_inference_timesteps(num_inference_steps)
            #sampler.set_begin_index()
            init_noise = sampler.init_noise_sigma()
            latents = latents * init_noise
        #print("Sampler has been successfully created.")
        

        diffusion = models["diffusion"].to(device)
        #print("Initializing diffusion model.")

        sampler_timesteps = tqdm(sampler.timesteps)
        for idx, sampler_timestep in enumerate(sampler_timesteps):
            time_embedding = get_time_embedding(sampler_timestep).to(device)
            #print("Time embedding has been created.")

            if use_cfg:
                ldm_input = torch.cat([latents, latents], dim=0)
            if sample_euler:
                sampler.scale_model_input(latents, sampler_timestep)
                # seems to lead to worse model outputs

            pred_noise = diffusion(ldm_input, conditioning, time_embedding)
            #print("Diffusion is predicting noise")
            
            if use_cfg:
                output_cond, output_uc = pred_noise.chunk(2)
                ldm_output = cfg_scale * (output_cond - output_uc) + output_uc

            # can call rescale_noise_cfg
            if use_cfg and sample_euler:
                pred_noise = rescale_noise_cfg(pred_noise, output_cond, rescale_cfg=0.9) # guidance_rescale from https://arxiv.org/pdf/2305.08891 pg.4

            #print("Taking step through sampler")
            latents = sampler.step(latents, sampler_timestep, ldm_output)
        
        to_idle(diffusion)

        post_quant_conv = models["autoencoder"].post_quantum_conv
        post_quant_conv = post_quant_conv.to(device)
        decoder = models["autoencoder"].decoder
        decoder = decoder.to(device)

        latents /= 0.18215
        final_output = decoder(post_quant_conv(latents))
        to_idle(post_quant_conv)
        to_idle(decoder)

        final_output = rescale(final_output, (-1,1), (0,255), clamp=True)
        final_output = final_output.permute(0, 2, 3, 1)
        final_output = final_output.to("cpu", torch.uint8).numpy()
        return final_output[0]
    


def rescale(
        x, 
        old_range, 
        new_range, 
        clamp=False,
    ):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

# Rescale `noise_cfg` according to `guidance_rescale`
# https://arxiv.org/pdf/2305.08891.pdf Section 3.4
def rescale_noise_cfg(
    pred_noise,
    output_cond,
    rescale_cfg=0.75,
):
    std_cond = output_cond.std(dim=list(range(1, output_cond.ndim)), keepdim=True)
    std_cfg = pred_noise.std(dim=list(range(1, pred_noise.ndim)), keepdim=True)
    # rescale guidance results -> fix overexposure
    pred_noise_rescaled = pred_noise * (std_cond / std_cfg)
    # combine with original results to avoid "plain images"
    pred_noise = rescale_cfg * pred_noise_rescaled + (1 - rescale_cfg) * pred_noise
    return pred_noise

# Might move to a util file
def get_time_embedding(
    timestep,
    dim=320,
    max_period=10000
):
    half_dim = dim // 2
    frequency = torch.pow(max_period, -torch.arange(start=0, end=half_dim, dtype=torch.float32) / half_dim)
    x = torch.tensor([timestep], dtype=torch.float32) * frequency[None]

    return torch.cat([torch.cos(x), torch.sin(x)], dim=1)