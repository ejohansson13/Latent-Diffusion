import torch
import torch.nn as nn
import torchvision
from collections import namedtuple

def normalize_tensor(x,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2,dim=1,keepdim=True))
    return x/(norm_factor+eps)


# LPIPS - using pre-trained model for semantic comparison between original and reconstructed image of autoencoder
class LPIPS(nn.Module):
    def __init__(
        self,
    ):
        """
        Perceptual loss implemenation from https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/lpips.py .
        Expects original image and reconstructed image after pass through VAE.
        Returns perceptual loss.
        """
        
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.channels = [64, 128, 256, 512, 512]
        self.vgg_network = vgg16()
        self.lins = nn.ModuleList(
            [nn.Conv2d(in_channels=ch, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False) for ch in self.channels]
        )
        for param in self.parameters():
            param.requires_grad = False

    def forward(
        self,
        reconstruction, # reconstructed image
        target, # original image
    ):
        input_rec, input_target = (self.scaling_layer(reconstruction), self.scaling_layer(target))
        output_rec, output_target = self.vgg_network(input_rec), self.vgg_network(input_target)
        feats_rec, feats_target, sq_diffs = {}, {}, {}
        res = 0

        for idx in range(len(self.channels)):
            feats_rec[idx], feats_target[idx] = normalize_tensor(output_rec[idx]), normalize_tensor(output_target[idx])
            sq_diffs[idx] = (feats_rec[idx] - feats_target[idx])**2

        for idx, lin_conv in enumerate(self.lins):
            res += lin_conv(sq_diffs[idx]).mean([2,3], keepdim=True)
        
        return res


class ScalingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class vgg16(torch.nn.Module):
    def __init__(
        self, 
        requires_grad=False, 
        pretrained=True
    ):
        """
        Loading VGG16 model for semantic comparison between original and reconstructed image.
        """
        
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg16(pretrained=pretrained).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out