import torch
from torch import nn
from torch.nn import functional as F

from helper_blocks import ResnetBlock, UNETAttnBlock, TimestepBlock
from schedulers.ddpm import DDPMSampler
from schedulers.ddim import DDIMSampler

MAX_TIMESTEPS = 1280/4

class UNET_Encoder(nn.Module):
    def __init__(
        self,
        init_channels=4,
        num_channels=320
    ):
        """
        Encoder portion of U-Net. Receives 32x32 latents and downsamples to 8x8.
        Expects latent, conditioning information, and timestep.
        Returns latent.
        """
        super().__init__()

        # each iteration of the U-Net uses the same timestep information at every ResNet block
        # timestep doesn't change until one pass through the U-Net has been completed

        self.conv_in = nn.Conv2d(init_channels, num_channels, kernel_size=3, padding=1)
        self.initial_res1 = ResnetBlock(num_channels, num_channels, use_time=True)
        self.initial_res2 = ResnetBlock(num_channels, num_channels, use_time=True)
        self.attn_1 = UNETAttnBlock(8, 40)
        self.attn_2 = UNETAttnBlock(8, 40)
        self.down_conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=2, padding=1)

        self.down1_res1 = ResnetBlock(num_channels, 2*num_channels, use_time=True)
        self.down1_res2 = ResnetBlock(2*num_channels, 2*num_channels, use_time=True)
        self.down1_attn1 = UNETAttnBlock(8, 80)
        self.down1_attn2 = UNETAttnBlock(8, 80)
        self.down_conv2 = nn.Conv2d(num_channels*2, num_channels*2, kernel_size=3, stride=2, padding=1)

        self.down2_res1 = ResnetBlock(2*num_channels, 4*num_channels, use_time=True)
        self.down2_res2 = ResnetBlock(4*num_channels, 4*num_channels, use_time=True)
        self.down2_attn1 = UNETAttnBlock(8, 160)
        self.down2_attn2 = UNETAttnBlock(8, 160)
        self.down_conv3 = nn.Conv2d(num_channels*4, num_channels*4, kernel_size=3, stride=2, padding=1)

        self.down3_res1 = ResnetBlock(4*num_channels, 4*num_channels, use_time=True)
        self.down3_res2 = ResnetBlock(4*num_channels, 4*num_channels, use_time=True)
    
    def forward(
        self,
        x,
        prompt,
        timesteps,
    ):
        skip_connections = []
        x = self.conv_in(x); skip_connections.append(x)
        x = self.initial_res1(x, timesteps)
        x = self.attn_1(x, prompt); skip_connections.append(x)
        x = self.initial_res2(x, timesteps)
        x = self.attn_2(x, prompt); skip_connections.append(x)
        x = self.down_conv1(x); skip_connections.append(x)

        x = self.down1_res1(x, timesteps)
        x = self.down1_attn1(x, prompt); skip_connections.append(x)
        x = self.down1_res2(x, timesteps)
        x = self.down1_attn2(x, prompt); skip_connections.append(x)
        x = self.down_conv2(x); skip_connections.append(x)

        x = self.down2_res1(x, timesteps)
        x = self.down2_attn1(x, prompt); skip_connections.append(x)
        x = self.down2_res2(x, timesteps)
        x = self.down2_attn2(x, prompt); skip_connections.append(x)
        x = self.down_conv3(x); skip_connections.append(x)

        x = self.down3_res1(x, timesteps); skip_connections.append(x)
        x = self.down3_res2(x, timesteps); skip_connections.append(x)

        return x, skip_connections
    

    
class UNET_Bottleneck(nn.Module):
    def __init__(
        self,
        num_channels=1280,
    ):
        """
        Bottleneck of UNet. Propagates 8x8 latents to UNet decoder.
        Expects latent, conditioning information, and timestep.
        Returns latent.
        """
        super().__init__()
        self.bottle_res1 = ResnetBlock(num_channels, num_channels, use_time=True)
        self.bottle_attn = UNETAttnBlock(8, 160)
        self.bottle_res2 = ResnetBlock(num_channels, num_channels, use_time=True)

    def forward(
        self,
        x,
        prompt,
        timesteps,
    ):
        x = self.bottle_res1(x, timesteps)
        x = self.bottle_attn(x, prompt)
        x = self.bottle_res2(x, timesteps)
        
        return x


class UNET_Decoder(nn.Module):
    def __init__(
        self,
        init_channels=4,
        num_channels=320,
    ):
        """
        Decoder portion of UNet. Expects 8x8 latents, upsamples to 32x32.
        Expects latent, conditioning information, and timestep.
        Returns latent.
        """
        super().__init__()

        self.initial_res1 = ResnetBlock(8*num_channels, 4*num_channels, use_time=True) # with concatenations, reaches 2x num_channels
        self.initial_res2 = ResnetBlock(8*num_channels, 4*num_channels, use_time=True)
        self.initial_res3 = ResnetBlock(8*num_channels, 4*num_channels, use_time=True)
        self.up_conv1 = nn.Conv2d(4*num_channels, 4*num_channels, kernel_size=3, padding=1)

        self.up1_res1 = ResnetBlock(8*num_channels, 4*num_channels, use_time=True)
        self.up1_res2 = ResnetBlock(8*num_channels, 4*num_channels, use_time=True)
        self.up1_res3 = ResnetBlock(6*num_channels, 4*num_channels, use_time=True)
        self.up1_attn1 = UNETAttnBlock(8, 160)
        self.up1_attn2 = UNETAttnBlock(8, 160)
        self.up1_attn3 = UNETAttnBlock(8, 160)
        self.up_conv2 = nn.Conv2d(4*num_channels, 4*num_channels, kernel_size=3, padding=1)

        self.up2_res1 = ResnetBlock(6*num_channels, 2*num_channels, use_time=True)
        self.up2_res2 = ResnetBlock(4*num_channels, 2*num_channels, use_time=True)
        self.up2_res3 = ResnetBlock(3*num_channels, 2*num_channels, use_time=True)
        self.up2_attn1 = UNETAttnBlock(8, 80)
        self.up2_attn2 = UNETAttnBlock(8, 80)
        self.up2_attn3 = UNETAttnBlock(8, 80)
        self.up_conv3 = nn.Conv2d(2*num_channels, 2*num_channels, kernel_size=3, padding=1)

        self.up3_res1 = ResnetBlock(3*num_channels, num_channels, use_time=True)
        self.up3_res2 = ResnetBlock(2*num_channels, num_channels, use_time=True)
        self.up3_res3 = ResnetBlock(2*num_channels, num_channels, use_time=True)
        self.up3_attn1 = UNETAttnBlock(8, 40)
        self.up3_attn2 = UNETAttnBlock(8, 40)
        self.up3_attn3 = UNETAttnBlock(8, 40)

        # UNET Output Layer
        self.normalize = nn.GroupNorm(num_groups=32, num_channels=num_channels, eps=1e-6)
        self.nonlinearity = nn.SiLU()
        self.conv_out = nn.Conv2d(num_channels, init_channels, kernel_size=3, padding=1)

    def forward(
        self,
        x,
        prompt,
        timesteps,
        skip_connections,
    ):
        # above blocks
        x = torch.cat((x, skip_connections.pop()), dim=1); x = self.initial_res1(x, timesteps)
        x = torch.cat((x, skip_connections.pop()), dim=1); x = self.initial_res2(x, timesteps)
        x = torch.cat((x, skip_connections.pop()), dim=1); x = self.initial_res3(x, timesteps)
        x = F.interpolate(x, scale_factor=2, mode="nearest"); x = self.up_conv1(x)

        x = torch.cat((x, skip_connections.pop()), dim=1); x = self.up1_res1(x, timesteps)
        x = self.up1_attn1(x, prompt)
        x = torch.cat((x, skip_connections.pop()), dim=1); x = self.up1_res2(x, timesteps)
        x = self.up1_attn2(x, prompt)
        x = torch.cat((x, skip_connections.pop()), dim=1); x = self.up1_res3(x, timesteps)
        x = self.up1_attn3(x, prompt)
        x = F.interpolate(x, scale_factor=2, mode="nearest"); x = self.up_conv2(x)

        x = torch.cat((x, skip_connections.pop()), dim=1); x = self.up2_res1(x, timesteps)
        x = self.up2_attn1(x, prompt)
        x = torch.cat((x, skip_connections.pop()), dim=1); x = self.up2_res2(x, timesteps)
        x = self.up2_attn2(x, prompt)
        x = torch.cat((x, skip_connections.pop()), dim=1); x = self.up2_res3(x, timesteps)
        x = self.up2_attn3(x, prompt)
        x = F.interpolate(x, scale_factor=2, mode="nearest"); x = self.up_conv3(x)

        x = torch.cat((x, skip_connections.pop()), dim=1); x = self.up3_res1(x, timesteps)
        x = self.up3_attn1(x, prompt)
        x = torch.cat((x, skip_connections.pop()), dim=1); x = self.up3_res2(x, timesteps)
        x = self.up3_attn2(x, prompt)
        x = torch.cat((x, skip_connections.pop()), dim=1); x = self.up3_res3(x, timesteps)
        x = self.up3_attn3(x, prompt)
        
        # UNET Output Layer
        x = self.normalize(x)
        x = self.nonlinearity(x)
        x = self.conv_out(x)

        return x
    

    
class UNET(nn.Module):
    def __init__(
        self,
        init_channels=4,
        out_channels=320,
    ):
        """
        Overall UNet framework. Progresses latent through denoising UNet.
        Expects latent, conditioning information, and timestep.
        Returns predicted quantity of noise.
        """

        super().__init__()
        self.encoder = UNET_Encoder(init_channels=init_channels, num_channels=out_channels)
        self.bottleneck = UNET_Bottleneck(num_channels=out_channels*4)
        self.decoder = UNET_Decoder(init_channels=init_channels, num_channels=out_channels)

    def forward(
        self,
        x,
        prompt,
        timesteps,
    ):
        x, skip_connections = self.encoder(x, prompt, timesteps)
        x = self.bottleneck(x, prompt, timesteps)
        x = self.decoder(x, prompt, timesteps, skip_connections)
        return x



class Diffusion(nn.Module):
    def __init__(
        self,
    ):
        """
        Overall diffusion framework. Responsible for predicting magnitude of noise to remove from latent.
        Expects latent, conditioning information, and timestep.
        Returns predicted quantity of noise.
        """

        super().__init__()
        self.timestep_embedding = TimestepBlock(320)
        self.unet = UNET(init_channels=4, out_channels=320)
    
    def forward(
        self,
        latent,
        conditioning,
        timestep,
    ):
        # latent: (Batch_Size, 4, Height / 8, Width / 8)
        # conditioning: (Batch_Size, Seq_Len, Dim)
        # time_step: (1, 320)
        time_step = self.timestep_embedding(timestep)
        pred_noise = self.unet(latent, conditioning, time_step)

        return pred_noise