import torch
from torch import nn
from torch.nn import functional as F

from helper_blocks import ResnetBlock, AttnBlock
from regularizer import KLRegularizer

class VAE_Encoder(nn.Sequential):
    def __init__(
        self,
        num_groups=32,
        eps=1e-6,
    ):
        """
        Encoder half of VAE. Responsible for encoding pixel-space images to latent representations.
        Expects pixel-space image.
        Returns non-quantized, compressed moments of image.
        """
        super().__init__()

        self.conv_in = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.initial_res1 = ResnetBlock(128, 128)
        self.initial_res2 = ResnetBlock(128, 128)
        # insert padding asymmetrically prior to downsampling operation to preserve equal numbers of rows and columns
        self.down_conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0)

        self.down1_res1 = ResnetBlock(128, 256)
        self.down1_res2 = ResnetBlock(256, 256)
        self.down_conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0)

        self.down2_res1 = ResnetBlock(256, 512)
        self.down2_res2 = ResnetBlock(512, 512)
        self.down_conv3 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0)

        self.down3_res1 = ResnetBlock(512, 512)
        self.down3_res2 = ResnetBlock(512, 512)

        self.middle_res1 = ResnetBlock(512, 512)
        self.attn = AttnBlock(512)
        self.middle_res2 = ResnetBlock(512, 512)

        self.normalize = nn.GroupNorm(num_groups=num_groups, num_channels=512, eps=eps)
        self.nonlinear = nn.SiLU()
        self.conv_final = nn.Conv2d(512, 8, kernel_size=3, stride=1, padding=1) 

    def forward(
        self,
        x
    ):
        x = self.conv_in(x)
        x = self.initial_res1(x)
        x = self.initial_res2(x)
        # insert padding asymmetrically prior to downsampling operation to preserve equal numbers of rows and columns
        x = F.pad(x, (0,1,0,1))
        x = self.down_conv1(x)

        x = self.down1_res1(x)
        x = self.down1_res2(x)
        x = F.pad(x, (0,1,0,1))
        x = self.down_conv2(x)

        x = self.down2_res1(x)
        x = self.down2_res2(x)
        x = F.pad(x, (0,1,0,1))
        x = self.down_conv3(x)

        x = self.down3_res1(x)
        x = self.down3_res2(x)

        x = self.middle_res1(x)
        x = self.attn(x)
        x = self.middle_res2(x)

        x = self.normalize(x)
        x = self.nonlinear(x)
        x = self.conv_final(x)
        return x


########################################################################################################################
#################################################    END OF ENCODER    #################################################
########################################################################################################################


class VAE_Decoder(nn.Sequential):
    def __init__(
        self,
        num_groups=32,
        eps=1e-6,
    ):
        """
        Decoder half of VAE. Responsible for upsampling latent to pixel-space.
        Expects unquantized latent.
        Returns pixel-space image.
        """
        super().__init__()

        self.conv_in = nn.Conv2d(4, 512, kernel_size=3, padding=1)
        self.middle_res1 = ResnetBlock(512, 512)
        self.attn = AttnBlock(512)
        self.middle_res2 = ResnetBlock(512, 512)

        self.initial_res1 = ResnetBlock(512, 512)
        self.initial_res2 = ResnetBlock(512, 512)
        self.initial_res3 = ResnetBlock(512, 512)
        self.up_1 = nn.Upsample(scale_factor=2)
        self.up_conv1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.up1_res1 = ResnetBlock(512, 512)
        self.up1_res2 = ResnetBlock(512, 512)
        self.up1_res3 = ResnetBlock(512, 512)
        self.up_2 = nn.Upsample(scale_factor=2)
        self.up_conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.up2_res1 = ResnetBlock(512, 256)
        self.up2_res2 = ResnetBlock(256, 256)
        self.up2_res3 = ResnetBlock(256, 256)
        self.up_3 = nn.Upsample(scale_factor=2)
        self.up_conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.up3_res1 = ResnetBlock(256, 128)
        self.up3_res2 = ResnetBlock(128, 128)
        self.up3_res3 = ResnetBlock(128, 128)
        self.normalize = nn.GroupNorm(num_groups=num_groups, num_channels=128, eps=eps)
        self.nonlinear = nn.SiLU()
        self.conv_final = nn.Conv2d(128, 3, kernel_size=3, padding=1)

    def forward(
        self,
        x
    ):
        x = self.conv_in(x)
        x = self.middle_res1(x)
        x = self.attn(x)
        x = self.middle_res2(x)

        x = self.initial_res1(x)
        x = self.initial_res2(x)
        x = self.initial_res3(x)
        x = self.up_1(x)
        x = self.up_conv1(x)

        x = self.up1_res1(x)
        x = self.up1_res2(x)
        x = self.up1_res3(x)
        x = self.up_2(x)
        x = self.up_conv2(x)

        x = self.up2_res1(x)
        x = self.up2_res2(x)
        x = self.up2_res3(x)
        x = self.up_3(x)
        x = self.up_conv3(x)

        x = self.up3_res1(x)
        x = self.up3_res2(x)
        x = self.up3_res3(x)
        x = self.normalize(x)
        x = self.normalize(x)
        x = self.nonlinear(x)
        x = self.conv_final(x)
        return x
    

########################################################################################################################
#################################################    END OF DECODER    #################################################
########################################################################################################################


class VariationalAutoEncoder(nn.Module):
    def __init__(
        self,
    ):
        """
        Overall VAE framework. 
        Responsible for encoding pixel-space image, quantizing, regularizing latent space, de-quantizing, and decoding latent.
        Expects pixel-sapce image.
        Returns pixel-space image.
        """
        super().__init__()
        self.encoder = VAE_Encoder(num_groups=32, eps=1e-6) # encodes pixel-space image
        self.quantum_conv = nn.Conv2d(8, 8, kernel_size=1) # quantizes moments
        self.post_quantum_conv = nn.Conv2d(4, 4, kernel_size=1) # un-quantizes latent
        self.decoder = VAE_Decoder(num_groups=32, eps=1e-6) # decodes latent to pixel-space image
        
        self.scale_factor = 0.18215
        # self.embed_dim

    # forward step through encoder of VAE
    def encode(
        self,
        input_image,
    ):
        latent = self.encoder(input_image)
        moments = self.quantum_conv(latent)
        posterior = KLRegularizer(moments)
        return posterior
    
    # forward step through decoder of VAE
    def decode(
        self,
        latent,
    ):
        latent /= self.scale_factor
        latent = self.post_quantum_conv(latent)
        output_image = self.decoder(latent)
        return output_image
    
    # forward step through entire autoencoder
    def forward(
        self,
        input_image,
        sample_posterior=True,
    ):
        posterior = self.encode(input_image)
        latent = posterior.sample()
        # if not sample_posterior -> latent = posterior.mean()
        output_image = self.decode(latent)

        return posterior, output_image

    def get_last_layer(
        self
    ):
        return self.decoder.conv_final.weight