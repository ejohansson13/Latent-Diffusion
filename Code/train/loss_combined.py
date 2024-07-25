import torch
import torch.nn as nn
import torch.nn.functional as F

from train.lpips import LPIPS
from train.patch_discriminator import NLayerDiscriminator
from regularizer import KLRegularizer

def hinge_d_loss(logits_real, logits_fake):
    """
    Hinge loss function taken from https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/vqperceptual.py
    """
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


# Overarching LPIPSWithDiscriminator
class LPIPSWithDiscriminator(nn.Module):
    def __init__(
        self,
        disc_start=50001,
        init_logvar = 0.0,
        kl_weight = 1e-6,
        pixel_loss_weight = 1.0,
        disc_in_channels = 3,
        num_layers_disc = 3,
        disc_factor = 1.0,
        disc_weight = 0.5,
        perceptual_weight = 1.0,
    ):
        """
        Overall VAE loss framework. 
        Expects original and reconstructed images, index for optimizer to be updated, and overall training step.
        Calculates reconstruction and adversarial losses.
        Returns overall loss.
        """
        
        super().__init__()

        self.kl_weight = kl_weight
        self.pixel_weight = pixel_loss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * init_logvar)
        self.discriminator = NLayerDiscriminator(num_channels_input=disc_in_channels, num_conv_layers_disc=num_layers_disc)
        self.disc_start_threshold = disc_start
        self.disc_factor = disc_factor
        self.disc_weight = disc_weight
    
    # calculate loss coefficients lambda according to page 4 of https://arxiv.org/pdf/2012.09841.pdf
    def calculate_adaptive_weight(
        self,
        nll_loss,
        generator_loss,
        last_layer=None
    ):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(generator_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.disc_weight
        return d_weight
        

    def forward(
        self,
        targets: torch.Tensor,
        reconstructions: torch.Tensor,
        posteriors: KLRegularizer,
        optimizer_idx,
        global_step,
        last_layer=None,
        split="train",
        #weights=None,
    ):
        rec_loss = torch.abs( targets.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            rec_loss = rec_loss + self.perceptual_weight * self.perceptual_loss(targets.contiguous(), reconstructions.contiguous())

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # GAN loss

        if optimizer_idx == 0:
            logits_fake = self.discriminator(reconstructions.contiguous())
            generator_loss = -torch.mean(logits_fake)
            
            if self.disc_factor > 0.0:
                d_weight = self.calculate_adaptive_weight(nll_loss, generator_loss, last_layer)
            else:
                d_weight = torch.tensor(0.0)

            if global_step < self.disc_start_threshold:
                disc_factor = 0

            loss = nll_loss + self.kl_weight*kl_loss + d_weight*disc_factor*generator_loss
            
            return loss
        
        if optimizer_idx == 1:
            logits_real = self.discriminator(targets.contiguous().detach())
            logits_fake = self.discriminator(reconstructions.contiguous().detach())

            if global_step < self.disc_start_threshold:
                disc_factor = 0
            
            d_loss = disc_factor * hinge_d_loss(logits_real, logits_fake)

            return d_loss