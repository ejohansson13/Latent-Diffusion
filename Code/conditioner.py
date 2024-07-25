import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class CLIPEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_embeddings: int,
        num_tokens: int,
    ):
        """
        Embeds conditioning tokens.
        Expects total vocabulary size, number of embeddings, and maximum amount of tokens per prompt.
        Returns embedded tokens.
        """
        super().__init__()

        self.token_embeddings = nn.Embedding(vocab_size, num_embeddings)
        self.positional_embeddings = nn.Parameter(torch.zeros((num_tokens, num_embeddings)))

    def forward(
        self,
        tokens,
    ):
        # x = self.token_embeddings(tokens)
        # x += self.positional_embeddings
        return self.token_embeddings(tokens) + self.positional_embeddings
    


class CLIPLayer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_embeddings: int,
    ):
        """
        Abstraction of layers of CLIP model following ViT-L/14 from https://github.com/mlfoundations/open_clip .
        Expects number of heads and number of embeddings.
        Returns the output of one CLIP layer.
        """
        super().__init__()

        # Pre-attention norm
        self.normalize_layer_1 = nn.LayerNorm(num_embeddings)
        self.attn = SelfAttention(num_heads, num_embeddings)
        self.normalize_layer_2 = nn.LayerNorm(num_embeddings)

        self.linear_1 = nn.Linear(num_embeddings, 4*num_embeddings)
        self.activation = nn.GELU()
        self.linear_2 = nn.Linear(4*num_embeddings, num_embeddings)

    def forward(
        self,
        x
    ):
        residual = x
        x = self.normalize_layer_1(x)
        x = self.attn(x, causal_mask=True)
        x += residual

        residual = x
        x = self.normalize_layer_2(x)
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.linear_2(x)

        x += residual
        return x
    
class CLIP(nn.Module):
    def __init__(
        self,
    ):
        """
        Embeds conditioning tokens, applies 12 CLIP layers, and normalizes textual embeddings.
        Expects tokenized conditioning information.
        Returns textual embeddings.
        """
        super().__init__()
        CLIP_VOCAB_SIZE = 49408
        CLIP_NUM_EMBEDDINGS = 768
        CLIP_MAX_TOKENS = 77

        self.embeddings = CLIPEmbedding(CLIP_VOCAB_SIZE, CLIP_NUM_EMBEDDINGS, CLIP_MAX_TOKENS)
        self.tokenizer_layers = nn.ModuleList([
            CLIPLayer(12, 768) for i in range(12)
        ])
        self.normalize_layer = nn.LayerNorm(CLIP_NUM_EMBEDDINGS)

    def forward(
        self,
        tokens: torch.LongTensor,
    ):
        tokens = tokens.long()
        embedded = self.embeddings(tokens)

        for layer in self.tokenizer_layers:
            embedded = layer(embedded)

        output = self.normalize_layer(embedded)
        return output