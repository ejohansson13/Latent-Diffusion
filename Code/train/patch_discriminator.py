import torch
import torch.nn as nn

class NLayerDiscriminator(nn.Module):
    def __init__(
        self,
        num_channels_input=3, # the number of channels in input images
        num_filters_last_conv=64, # the number of filters in the last conv layer
        num_conv_layers_disc=3, # the number of conv layers in the discriminator
    ):
        """
        Patch-based adversarial loss from https://github.com/CompVis/taming-transformers/blob/master/taming/modules/discriminator/model.py .
        Expects reconstructed image after pass through VAE.
        Returns reconstructed image with scalar patch of added noise.
        """
        super().__init__()
        normalize_batch = nn.BatchNorm2d
        num_filters_mult = 1
        num_filters_mult_prev = 1

        sequence_ops = [
            nn.Conv2d(num_channels_input, num_filters_last_conv, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for n in range(1, num_conv_layers_disc):
            num_filters_mult_prev = num_filters_mult
            num_filters_mult = min(2**n, 8)
            sequence_ops += [
                nn.Conv2d(num_filters_last_conv*num_filters_mult_prev, num_filters_last_conv*num_filters_mult, kernel_size=4, stride=2, padding=1, bias=False),
                normalize_batch(num_filters_last_conv*num_filters_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        num_filters_mult_prev = num_filters_mult
        sequence_ops += [
            nn.Conv2d(num_filters_last_conv*num_filters_mult_prev, num_filters_last_conv*num_filters_mult, kernel_size=4, stride=1, padding=1, bias=False),
            normalize_batch(num_filters_last_conv*num_filters_mult),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_filters_last_conv*num_filters_mult, 1, kernel_size=4, stride=1, padding=1)
        ]

        self.sequence_ops = nn.Sequential(*sequence_ops)

    def forward(
        self,
        x
    ):
        return self.sequence_ops(x)