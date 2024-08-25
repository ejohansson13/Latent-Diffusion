import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from PIL import Image
from tqdm import tqdm

from autoencoder import VariationalAutoEncoder
from train.loss_combined import LPIPSWithDiscriminator

"""
class AutoencoderDataset(Dataset):
    def __init__(
        self,
        data_dir,
    ):
    
    def __len__(
        self
    ):
        return len(self.data_dir)
    
    def __getitem__(
        self, 
        index
    ):
        return {"image": image}
"""

# configure optimizers for learning vae parameters according to perceptual loss, kl-regularization, and patch-based adversarial loss
def config_optimizers(
    vae: VariationalAutoEncoder,
    loss: LPIPSWithDiscriminator,
    lr = 0,
):
    lr = lr# have to initialize this somewhere
    vae_optimizer = torch.optim.Adam(list(vae.encoder.parameters())+
                                list(vae.decoder.parameters())+
                                list(vae.quantum_conv.parameters())+
                                list(vae.post_quantum_conv.parameters()),
                                lr=lr, betas=(0.5, 0.9))
    disc_optimizer = torch.optim.Adam(loss.discriminator.parameters(),
                                        lr=lr, betas=(0.5, 0.9))
    
    return vae_optimizer, disc_optimizer


DEVICE="mps"
# SCAFFOLDING VALUES
num_epochs=3
train_dataset = ""
ckpt_dir = ""
load_pretrained_model = False
use_prev_global_step = False

# initialize VAE -> all below are initialized along with VAE
    # initialize perceptual loss (VGG16 loss) -> initialized in LPIPS call
    # initialize patch-based adversarial loss -> initialized in LPIPS call
    # have to figure out configuration of decaying lr schedule
AutoencoderKL = VariationalAutoEncoder().to(DEVICE)

# Option to load pre-trained VAE weights
    # read in previous global step and assign it here
if load_pretrained_model:
    model_ckpt = "saved_vae_model"
    state_dict = torch.load(os.path.join(ckpt_dir, model_ckpt), map_location=DEVICE)
    AutoencoderKL.load_state_dict(state_dict)

# parse model ckpt for global_step
if use_prev_global_step:
    global_step = model_ckpt.split("_")[3]
    global_step = int(global_step[:-4])
else:
    global_step = 0

# initialize loss metric
# perceptual loss and patch-based adversarial loss are both handled in LPIPSWithDiscriminator class
loss = LPIPSWithDiscriminator().to(DEVICE)

# initialize dual optimizers for vae objectives
vae_opt, disc_opt = config_optimizers(AutoencoderKL, loss, lr=4.5e-6)

# step through VAE -> all below should be handled in VAE
    # pass image through encoder
    # perform latent-space regularization (KL)
    # pass latent through decoder
    # update loss values
AutoencoderKL.train()

# dataset loader for torch-compatible training
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=16)

# current latent rescaling factor was calculated by original LDM authors: https://arxiv.org/pdf/2112.10752
# optionality to recalculate latent rescaling factor to better align with custom user dataset
calc_rescaling_factor = False
if calc_rescaling_factor:
    all_latents_std = []

for epoch in tqdm(range(num_epochs)):
    for idx, batch in enumerate(train_loader):
        original_image = batch["image"].to(DEVICE)
        encoding_image = batch["image"].to(DEVICE)
        # iterate over VAE with training steps
        posteriors, reconstructions = AutoencoderKL(encoding_image)
        reconstructions = reconstructions.to(DEVICE)

        if calc_rescaling_factor:
            std_latents = posteriors.sample()
            all_latents_std.extend(std_latents)

        # calculate patch-based adversarial loss
        vae_loss = loss(
            original_image, reconstructions, posteriors, optimizer_idx=0, global_step=global_step, last_layer=AutoencoderKL.get_last_layer()
        )

        vae_opt.zero_grad()
        vae_loss.backward()
        vae_opt.step()

        # calculate perceptual loss
        disc_loss = loss(
            original_image, reconstructions, posteriors, optimizer_idx=1, global_step=global_step, last_layer=AutoencoderKL.get_last_layer()
        )
        disc_loss.backward()
        disc_opt.step()
        disc_opt.zero_grad()

        global_step += 1

    # checkpoint and save model weights
    if global_step % 10000 == 0:
        state_dict = AutoencoderKL.state_dict()
        torch.save(state_dict, os.path.join("data", "checkpoints", f"vae_ckpt_epoch_{global_step}.pth"))


# update latent constant multipler in accordance with variance
# https://github.com/huggingface/diffusers/issues/437
if calc_rescaling_factor:
    all_latents_std_tensor = torch.cat(all_latents_std)
    std = all_latents_std_tensor.std().item()
    rescaling_factor = 1 / std
    print(f"normalizer = {rescaling_factor}")