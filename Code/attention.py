import torch
from torch import nn
from torch.nn import functional as F

class SelfAttention(nn.Module):
    def __init__(
        self,
        num_heads,
        dim_embeddings,
        in_proj_bias=True,
        out_proj_bias=True,
    ):
        """
        Self Attention mechanism allowing holistic understanding of feature embeddings.
        Expects number of attention heads and embedding dimensions.
        Returns self-attended image features.
        """
        super().__init__()

        self.num_heads = num_heads
        self.dim_heads = dim_embeddings // num_heads
        # combines Wq, Wk, and Wv operations into one matrix
        self.proj_in = nn.Linear(dim_embeddings, 3*dim_embeddings, bias=in_proj_bias)
        self.proj_out = nn.Linear(dim_embeddings, dim_embeddings, bias=out_proj_bias)

    def forward(
        self,
        x,
        causal_mask=False, # needed for text conditioner
    ):
        # once fully functional, can eliminate repetitive code of input_shape var
        batch_size, sequence_length, dim_embeddings = x.shape
        
        # (Batch_Size, Seq_Len, H, Dim / H)
        interim_shape = (batch_size, sequence_length, self.num_heads, self.dim_heads)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim * 3) -> 3 tensor of shape (Batch_Size, Seq_Len, Dim)
        query, key, value = self.proj_in(x).chunk(3, dim=2) # can also be dim=-1

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
        query = query.view(interim_shape).transpose(1,2)
        value = value.view(interim_shape).transpose(1,2)
        # (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, H, Dim / H, Seq_Len)
        key = key.view(interim_shape).permute(0,2,3,1)

        # compute attention
        # (Batch_Size, H, Seq_Len, Dim / H) @ (Batch_Size, H, Dim / H, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weight = query@key

        if causal_mask:
            # Mask with upper triangle of 1
            mask = torch.ones(weight.size(), dtype=torch.bool).triu(1)
            mask = mask.to(weight.device)
            weight.masked_fill_(mask, -torch.inf)
            del mask

        weight *= (int(self.dim_heads)**(-0.5)) # weight = weight*(int(channels)**(-0.5))
        weight = F.softmax(weight, dim=-1)

        x = weight @ value
        x = x.transpose(1,2)
        x = x.reshape((batch_size, sequence_length, dim_embeddings))

        x = self.proj_out(x)
        return x
        

class CrossAttention(nn.Module):
    def __init__(
        self,
        num_heads,
        dim_mat1,
        dim_mat2,
        in_proj_bias=True,
        out_proj_bias=True,
    ):
        """
        Cross Attention mechanism responsible for U-Net understanding conditioning information.
        Expects the number of attention heads, two matrices representing feature and conditioning information respectively.
        Image features act as the query, conditioning acts as key and value matrices.
        Returns cross-attention product of image embeddings and conditioning.
        """
        super().__init__()

        self.query_proj  = nn.Linear(dim_mat1, dim_mat1, bias=in_proj_bias)
        self.key_proj    = nn.Linear(dim_mat2, dim_mat1, bias=in_proj_bias)
        self.value_proj  = nn.Linear(dim_mat2, dim_mat1, bias=in_proj_bias)

        self.proj_out = nn.Linear(dim_mat1, dim_mat1, bias=out_proj_bias)
        self.num_heads = num_heads
        self.dim_heads = dim_mat1 // num_heads

    def forward(
        self,
        mat_latent,
        mat_conditioner,
    ):
        batch_size, sequence_length, dim_embeddings = mat_latent.shape
        interim_shape = (batch_size, -1, self.num_heads, self.dim_heads)

        query = self.query_proj(mat_latent)
        key = self.key_proj(mat_conditioner)
        value = self.value_proj(mat_conditioner)

        query = query.view(interim_shape).transpose(1,2)
        value = value.view(interim_shape).transpose(1,2)
        key = key.view(interim_shape).permute(0,2,3,1)

        weight = query @ key
        weight *= (int(self.dim_heads)**(-0.5)) # weight = weight*(int(channels)**(-0.5))
        weight = F.softmax(weight, dim=-1)

        x = weight@value
        x = x.transpose(1,2).contiguous()
        x = x.view((batch_size, sequence_length, dim_embeddings))

        x = self.proj_out(x)
        return x