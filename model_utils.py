import torch
import torch.nn as nn
from einops import reduce, rearrange, repeat
from einops.layers.torch import Rearrange, Reduce


class CLSToken(nn.Module):
	def __init__(self, embedding_dim):
		super().__init__()
		self.embedding_dim = embedding_dim
		self.cls_token = nn.Parameter(torch.randn(1, 1, self.embedding_dim))

	def forward(self, x):
		BATCH_SIZE = x.shape[0]
		
		# repeat the parameter to accomodate the batch size, and then concatenate along columns
		# leads to shape (batch, p x p + 1, embedding_dim)
		cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=BATCH_SIZE)
			
		concat_tokens = torch.cat([cls_tokens, x], dim=1)
		return concat_tokens


class PositionalEncoding(nn.Module):
	def __init__(self, img_size=224, patch_size=14, embedding_dim=764):
		super().__init__()
		self.img_size = img_size
		self.patch_size = patch_size
		self.embedding_dim = embedding_dim

		# making sure shapes are correct to add cls_token appended embeddings
		self.positional_encoding = nn.Parameter(torch.randn((self.img_size // self.patch_size)**2+1, self.embedding_dim))

	def forward(self, x):
		return self.positional_encoding + x
	

class PatchEmbedding(nn.Module):
	def __init__(self, img_size=224, in_channels=3, patch_size=16, embedding_dim=768):
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
		x = self.projection(x) # shape (batch, p x p, embedding_dim)
		x = CLSToken(self.embedding_dim)(x)
		x = PositionalEncoding(self.img_size, self.patch_size, self.embedding_dim)(x)
		return x   # shape: (batch, p x p + 1, embdding_dim)


class MultiHeadAttention(nn.Module):
	def __init__(self, embedding_dim=768, num_heads=8, scaling=True, dropout=0.0):
		super().__init__()
		self.num_heads = num_heads
		self.embedding_dim = embedding_dim
		self.dropout = nn.Dropout(dropout)
		self.scaling = scaling

		self.queries_keys_values = nn.Linear(embedding_dim, 3*embedding_dim)
		self.projection = nn.Linear(embedding_dim, embedding_dim)
	
	def forward(self, x):
		splits = rearrange(self.queries_keys_values(x), 'b n (h d qkv) -> (qkv) b h n d', qkv=3, h=self.num_heads)
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
	def __init__(self, embedding_dim=768, expansion_factor=4, dropout=0.0):
		super().__init__()
		self.embedding_dim = embedding_dim
		self.expansion_factor = expansion_factor
		self.droput = nn.Dropout(dropout)

		self.mlp = nn.Linear(embedding_dim, embedding_dim*expansion_factor)
		self.activation = nn.GELU()
		self.mlp2 = nn.Linear(embedding_dim*expansion_factor, embedding_dim)

	def forward(self, x):
		x = self.mlp(x)
		x = self.activation(x)
		x = self.mlp2(x)
		x = self.droput(x)
		return x	


class Block(nn.Module):
	def __init__(self, embedding_dim=768, expansion_factor=4, dropout=0.0):
		super().__init__()
		self.embedding_dim = embedding_dim,
		self.expansion_factor = expansion_factor

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
	def __init__(self, depth=12, name="ViT"):
		super().__init__()
		self.name = name
		self.depth = depth

	def forward(self, x):
		for _ in range(self.depth):
			x = Block()(x)
		
		return x