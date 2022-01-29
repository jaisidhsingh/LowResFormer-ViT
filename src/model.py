from vit_pytorch import ViT
from config import *
import torch
import torch.nn as nn
from einops import reduce, rearrange, repeat
from einops.layers.torch import Rearrange, Reduce


class CLSToken(nn.Module):
    def __init__(self, embedding_dim=DIM, device=DEVICE):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embedding_dim)).to(device)

    def forward(self, x):
        BATCH_SIZE = x.shape[0]

        # repeat the parameter to accomodate the batch size, and then concatenate along columns
        # leads to shape (batch, p x p + 1, embedding_dim)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=BATCH_SIZE)

        concat_tokens = torch.cat([cls_tokens, x], dim=1)
        return concat_tokens


class PositionalEncoding(nn.Module):
    def __init__(
            self, 
            img_size=IMAGE_SIZE, 
            patch_size=PATCH_SIZE, 
            embedding_dim=DIM, 
            device=DEVICE
        ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim

        # making sure shapes are correct to add cls_token appended embeddings
        self.positional_encoding = nn.Parameter(torch.randn(
            (self.img_size // self.patch_size)**2+1, self.embedding_dim)).to(device)

    def forward(self, x):
        return self.positional_encoding + x


class PatchEmbedding(nn.Module):
    def __init__(
            self, 
            img_size=IMAGE_SIZE, 
            in_channels=IN_CHANNELS, 
            patch_size=PATCH_SIZE, 
            embedding_dim=DIM
        ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.img_size = img_size

        self.projection = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.embedding_dim,
                kernel_size=self.patch_size,
                stride=self.patch_size
            ),
            Rearrange(' b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x):
        x = self.projection(x)  # shape (batch, p x p, embedding_dim)
        x = CLSToken(self.embedding_dim)(x)
        x = PositionalEncoding(
            self.img_size, self.patch_size, self.embedding_dim)(x)
        return x   # shape: (batch, p x p + 1, embdding_dim)


# x = torch.randn((1, 3, 32, 32))
# p = PatchEmbedding(img_size=32, in_channels=3, patch_size=4, embedding_dim=768)
# print(p(x).shape)


class MultiHeadAttention(nn.Module):
    def __init__(
            self, 
            embedding_dim=DIM, 
            num_heads=HEADS, 
            scaling=True, 
            dropout=0.0
        ):
        super().__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.dropout = nn.Dropout(dropout)
        self.scaling = scaling

        self.queries_keys_values = nn.Linear(embedding_dim, 3*embedding_dim)
        self.projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        splits = rearrange(self.queries_keys_values(
            x), 'b n (h d qkv) -> (qkv) b h n d', qkv=3, h=self.num_heads)
        queries, keys, values = splits[0], splits[1], splits[2]

        attention = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        attention = nn.functional.softmax(attention, dim=-1)
        if self.scaling:
            attention = attention / (self.embedding_dim**0.5)

        attention = self.dropout(attention)

        output = torch.einsum('bhad, bhdv -> bhav', attention, values)
        output = rearrange(output, 'b h a v -> b a (h v)')
        return self.projection(output)


class AdditiveSkipConnection(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        tmp = x
        x = self.layer(x)
        return x + tmp


class MLP(nn.Module):
    def __init__(
        self, 
        embedding_dim=DIM, 
        expansion_factor=EXPANSION_FACTOR, 
        dropout=DROPOUT
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.expansion_factor = expansion_factor
        self.droput = nn.Dropout(dropout)

        self.mlp = nn.Linear(
            self.embedding_dim, 
            self.embedding_dim*self.expansion_factor
        )
        self.activation = nn.GELU()
        self.mlp2 = nn.Linear(
            self.embedding_dim*self.expansion_factor, 
            self.embedding_dim
        )

    def forward(self, x):
        x = self.mlp(x)
        x = self.activation(x)
        x = self.mlp2(x)
        x = self.droput(x)
        return x


class Block(nn.Module):
    def __init__(
            self, 
            embedding_dim=DIM, 
            expansion_factor=EXPANSION_FACTOR,
            dropout=DROPOUT,
        ):
        super().__init__()
        self.embedding_dim = embedding_dim,
        self.expansion_factor = expansion_factor
        self.dropout = dropout

    def forward(self, x):
        tmp = x
        x = nn.LayerNorm(self.embedding_dim)(x)
        x = MultiHeadAttention()(x)
        x += tmp

        tmp = x
        x = nn.LayerNorm(self.embedding_dim)(x)
        x = MLP()(x)
        x += tmp
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self, 
        embedding_dim=DIM, 
        expansion_factor=EXPANSION_FACTOR, 
        dropout=DROPOUT, 
        depth=DEPTH
    ):
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.expansion_factor = expansion_factor
        self.dropout = dropout

    def forward(self, x):
        for _ in range(self.depth):
            x = Block(
                )(x)

        return x


class ClassificationHead(nn.Module):
    def __init__(self, embedding_dim=DIM, intermediate_dim=INT_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.intermediate_dim = intermediate_dim
        self.projection = nn.Linear(embedding_dim*intermediate_dim, 512)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, self.num_classes)

    def forward(self, x):
        b, n, e = x.shape
        x = x.view((b, n*e))
        x = self.projection(x)
        x = nn.functional.relu(x)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x

class AttributeEmbeddingNetwork(nn.Module):
    def __init__(self, input_size=NUM_ATTR, embedding_dim=DIM, output_size=ATTR_EMB):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.input_size = input_size
        self.output_size = output_size
        # self.embedding = nn.Embedding(input_size, embedding_dim)
        self.projection = nn.Linear(input_size, 512)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, output_size)
    
    def forward(self, x):
        # x = self.embedding(x).long()
        x = self.projection(x)
        x = nn.functional.relu(x)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x

class AttributeHead(nn.Module):
    def __init__(self, embedding_dim=DIM, intermediate_dim=INT_DIM, num_attr=ATTR_EMB):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_attr = num_attr
        self.intermediate_dim = intermediate_dim

        dim = self.embedding_dim*self.intermediate_dim
        self.projection = nn.Linear(dim, 512)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, self.num_attr)

    def forward(self, x):
        b, n, e = x.shape
        x = x.view((b, n*e))
        x = self.projection(x)
        x = nn.functional.relu(x)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x


class FinalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pe = PatchEmbedding()
        self.trans = VisionTransformer()
        self.cls_head = ClassificationHead()
        self.attr_head = AttributeHead()
        self.attr_emb = AttributeEmbeddingNetwork()
    
    def forward(self, x, a):
        x = self.pe(x)
        x = self.trans(x)
        ae = self.attr_emb(a)
        c = self.cls_head(x)
        ap = self.attr_head(x)

        return (c, ap, ae)

