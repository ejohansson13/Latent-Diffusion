import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from PIL import Image
from tqdm import tqdm

from transformers import CLIPTokenizer
import model_loader
from diffusion import Diffusion
from regularizer import KLRegularizer
from pipeline import get_time_embedding
from schedulers.ddim import DDIMSampler

"""
class DiffusionDataset(Dataset):
    def __init__(
        self,
        data_dir,
    ):
        return self
    
    def __len__(
        self
    ):
        return len(self.data_dir)
    
    def __getitem__(
        self, 
        index
    ):
        return {"image": , "caption": ""}
"""
        
def generate_timestep(latent_shape):
    # randomly sample timestep of noise to be added -> CompVis code suggests adding .long()
    # https://www.youtube.com/watch?v=T0Qxzf0eaio suggests log-normal distribution for sampling timesteps
    log_normal_timestep = torch.zeros(latent_shape[0]) # generate new tensor
    log_normal_timestep.log_normal_(2, 1) # fill it with log-normal distribution of timesteps
    timestep = torch.randint(0, num_train_timesteps, (latent_shape[0],), device="cpu").long() # samples from uniform distribution
    new_timesteps = []

    for i in range(latent_shape[0]):
        new_timestep = (timestep[i] + log_normal_timestep[i]) / 2 # average samples from log-normal and uniform distributions
        new_timesteps.append(new_timestep.round().item())

    return torch.tensor(new_timesteps).long()
    

def calc_added_noise(sampler_vals, timestep, latent_shape):
        noise_step = sampler_vals[timestep]
        return noise_step.reshape( timestep.shape[0], *((1,) * (len(latent_shape) - 1)) )


train_loader = [
    {"image": torch.randn((1,3,512,512)), "caption": "this is the 1st image"},
    {"image": torch.randn((1,3,512,512)), "caption": "this is the 2nd image"},
]


DEVICE="mps"
generator = torch.Generator(device=DEVICE)
# SCAFFOLDING VALUES
num_epochs=3
train_dataset = ""
ckpt_dir = ""
load_pretrained_model = False
use_prev_global_step = False

# use default pre-trained weights
tokenizer = CLIPTokenizer("data/tokenizer_vocab.json", merges_file="data/tokenizer_merges.txt")
model_path = "data/v1-5-pruned-emaonly.ckpt"
model_components = model_loader.load_model_weights(model_path, DEVICE)

# train with ddim sampler
sampler = DDIMSampler(generator)

#with torch.no_grad():
# load pre-trained autoencoder weights
# initialize pre-trained AutoEncoder to encode training images
clip = model_components["conditioner"].to(device="cpu").eval()
encoder = model_components["autoencoder"].encoder
encoder = encoder.to(device="cpu").eval()
quant_conv = model_components["autoencoder"].quantum_conv
quant_conv = quant_conv.to(device="cpu").eval()

# initialize U-Net and Diffusion components
# Running in eps mode -> we are only ever predicting noise, not x0
UNet = Diffusion().to(DEVICE)

# Option to load pre-trained U-Net weights
if load_pretrained_model:
    model_ckpt = "saved_diffusion_model"
    state_dict = torch.load(os.path.join(ckpt_dir, model_ckpt), map_location=DEVICE)
    UNet.load_state_dict(state_dict)

# parse model ckpt for global_step
if use_prev_global_step:
    global_step = model_ckpt.split("_")[3]
    global_step = int(global_step[:-4])
else:
    global_step = 0

# initialize optimizer for u-net
diffusion_opt = torch.optim.Adam(list(UNet.parameters()), lr=4.5e-6, betas=(0.5, 0.9))

num_train_timesteps = 1000
#train_loader = DataLoader(train_dataset, shuffle=True, batch_size=16)
UNet.train()

for epoch in tqdm(range(num_epochs)):
    for idx, batch in enumerate(train_loader):
        input_image = batch["image"].to(device="cpu")
        conditioning = batch["caption"]

        # encode prompts -> use pre-trained CLIP to encode prompt
        # conditioning should be occasionally and randomly dropped such that the model can also generate models unconditionally
        # create random probability that provided prompt is utilized or not: conditional vs unconditional
        # https://arxiv.org/pdf/2207.12598 suggests p = 0.1,0.2 perform the best and roughly equivalently to each other
        with torch.no_grad():
            cond_tokens = tokenizer.batch_encode_plus( [conditioning], padding="max_length", max_length=77 ).input_ids
            if torch.rand(1) <= 0.2:
                cond_tokens = tokenizer.batch_encode_plus( [""], padding="max_length", max_length=77 ).input_ids
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device="cpu")
            conditioning = clip(cond_tokens)

            # receive latent from encoder
            # .detach() may or may not be necessary
            # have to handle posterior first
            encoded_image = quant_conv(encoder(input_image)).detach()
            posterior = KLRegularizer(encoded_image)
        # scaling by pre-determined vae scale factor handled in .mode() function
        latent = posterior.mode()
        del posterior

        added_noise = torch.randn_like(latent).to(device="cpu")
        timestep = generate_timestep(latent.shape)
        
        alphas_cumprod = sampler.alphas_cumprod
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)

        # Add sampled noise according to Eq. 4 from https://arxiv.org/pdf/2010.02502 -> reparametrized on page 3
        x_start_coeff = calc_added_noise(sqrt_alphas_cumprod, timestep, latent.shape)
        x_start_coeff = x_start_coeff.to(device="cpu", dtype=torch.float32)
        noise_coeff = calc_added_noise(sqrt_one_minus_alphas_cumprod, timestep, latent.shape)
        noise_coeff = noise_coeff.to(device="cpu", dtype=torch.float32)
        
        noisy_latent = x_start_coeff*latent + noise_coeff*added_noise
        noisy_latent = noisy_latent.to(DEVICE)
        conditioning = conditioning.to(DEVICE)

        # pass in noisy latent, prompts, and sampled timesteps for noise
        timestep_emb = get_time_embedding(timestep)
        timestep_emb = timestep_emb.to(DEVICE)
        UNet.to(DEVICE)
        pred_noise = UNet(noisy_latent, conditioning, timestep_emb)

        # calculate loss: compare predicted noise to ground-truth 
        added_noise = added_noise.to(DEVICE)
        diffusion_loss = F.mse_loss(added_noise, pred_noise, reduction='none').mean([1,2,3])

        diffusion_loss.backward()
        UNet.to(device="cpu")

        diffusion_opt.step()
        diffusion_opt.zero_grad()
        global_step += 1

    # checkpoint and save weights
    if global_step % 10000 == 0:
        state_dict = UNet.state_dict()
        torch.save(state_dict, os.path.join(ckpt_dir, f"unet_ckpt_epoch_{global_step}.pth"))