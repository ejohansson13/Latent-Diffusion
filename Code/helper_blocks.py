import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention

class ResnetBlock (nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        #dropout,
        num_groups=32,
        eps=1e-6,
        use_time=False,
        num_timesteps=1280,
    ):
        """
        ResNet block implementation serving as the building block for the Variational Auto-Encoder and U-Net.
        Expects a minimum argument containing the number of input channels and output channels.
        Returns features propagated by the network.
        If using for U-Net, make sure to include use_time argument, allowing the temporal information to be embedded into the U-Net.
        """
        super().__init__()

        self.nonlinearity = nn.SiLU() # activation function, F.silu = x*torch.sigmoid(x)
        self.normalize_group_1 = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.use_time = use_time
        if self.use_time:
            self.time_linear = nn.Linear(num_timesteps, out_channels)
            
        self.normalize_group_2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels, eps=eps, affine=True)
        # dropout to regularize learning across all neurons
        #self.dropout = torch.nn.Dropout(dropout) # dependent on size of training set, may not be necessary
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.residual_path = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        else:
            self.residual_path = nn.Identity()

    def forward(
        self,
        x,
        timestep=None,
    ) -> torch.Tensor:
        
        # separate residual
        residual = x
        x = self.normalize_group_1(x)
        x = self.nonlinearity(x)
        x = self.conv_1(x)

        # implement temporal information if using ResNet block for UNet
        if self.use_time:
            timestep = self.nonlinearity(timestep)
            timestep = self.time_linear(timestep)
            timestep = timestep.unsqueeze(-1).unsqueeze(-1) # dimension compatibility
            x = x + timestep
        
        x = self.normalize_group_2(x)
        x = self.nonlinearity(x)
        #x = self.dropout(x)
        x = self.conv_2(x)
        
        return x + self.residual_path(residual) # reimplement residual
    

class AttnBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        num_groups=32,
        eps=1e-6,
    ):
        """
        Simplified attention block. Includes feature normalization before Self-Attention operation.
        Transforms dimensions to align with Self-Attention.
        Returns self-attended features.
        """
        super().__init__()
        self.normalize_group = nn.GroupNorm(num_groups, in_channels, eps)
        self.attention = SelfAttention(1, in_channels, num_groups)
    
    def forward(
        self,
        x: torch.Tensor,
    ):
        residual = x
        # normalize
        x = self.normalize_group(x)
        batch_size, channels, height, width = x.shape
        x = x.reshape((batch_size, channels, height*width))
        x = x.transpose(2,1)

        # call self.attention
        x = self.attention(x)
        x = x.transpose(2,1)
        x = x.reshape(batch_size, channels, height, width)

        x += residual
        return x
    
class UNETAttnBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_embeddings: int,
        dim_conditioner = 768,
        num_groups=32,
        eps=1e-6,
    ):
        """
        Attention block implementation for U-Net.
        Expects minimum argument of number of heads and number of embeddings for attention operations.
        Performs cross-attention between image feature embeddings and conditioning.
        Returns advanced features.
        """
        super().__init__()
        num_channels = num_heads * num_embeddings

        self.normalize_group = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=eps, affine=True)
        self.conv_in = nn.Conv2d(num_channels, num_channels, kernel_size=1, padding=0)

        self.normalize_layer_1 = nn.LayerNorm(num_channels)
        self.attn = SelfAttention(num_heads, num_channels, in_proj_bias=False)
        
        self.normalize_layer_2 = nn.LayerNorm(num_channels)
        self.cross_attn = CrossAttention(num_heads, num_channels, dim_conditioner, in_proj_bias=False)
        
        self.normalize_layer_3 = nn.LayerNorm(num_channels)
        self.geglu_nonlinearity_1 = nn.Linear(num_channels, 4*num_channels*2)
        self.nonlinearity = nn.GELU()
        self.geglu_nonlinearity_2 = nn.Linear(4*num_channels, num_channels)
        self.conv_out = nn.Conv2d(num_channels, num_channels, kernel_size=1, padding=0)

    def forward(
        self,
        x: torch.Tensor,
        conditioner,
    ):
        residual_long = x
        x = self.normalize_group(x)
        x = self.conv_in(x)

        batch_size, channels, height, width = x.shape
        x = x.view((batch_size, channels, height*width))
        x = x.transpose(-1,-2)

        residual_short = x
        x = self.normalize_layer_1(x)
        x = self.attn(x)
        x += residual_short

        residual_short = x
        x = self.normalize_layer_2(x)
        x = self.cross_attn(x, conditioner)
        x += residual_short

        residual_short = x
        x = self.normalize_layer_3(x)
        x, gate = self.geglu_nonlinearity_1(x).chunk(2,dim=-1)
        x = x * self.nonlinearity(gate)
        x = self.geglu_nonlinearity_2(x)
        x += residual_short

        x = x.transpose(-1,-2)
        x = x.view((batch_size, channels, height, width))
        return self.conv_out(x) + residual_long

class TimestepBlock(nn.Module):
    def __init__(
        self,
        num_embeddings
    ):
        """
        Converts timestep to timestep embedding for UNet compatibility.
        Expects embedding size for timestep embedding.
        Returns dimensionally compatible timestep information for UNet.
        """
        super().__init__()

        self.linear_1 = nn.Linear(num_embeddings, 4*num_embeddings)
        self.nonlinearity = nn.SiLU()
        self.linear_2 = nn.Linear(4*num_embeddings, 4*num_embeddings)

    def forward(
        self,
        x
    ):
        x = self.linear_1(x)
        x = self.nonlinearity(x)
        x = self.linear_2(x)

        return x