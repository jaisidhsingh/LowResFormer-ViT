import torch
from PIL import Image
from vit_pytorch import ViT
from model import MODEL
from config import *
from einops import *

x = torch.randn((1, IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE))

model = MODEL()
scale = 1/8
heads = 8
x = model.pe(x)
x = model.dropout(x)

att_mat = []

for att, ff in model.trans.layers:
	att_layer = att.fn
	qkv = att_layer.to_qkv(x).chunk(3, dim = -1)
	q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = heads), qkv)

	dots = torch.matmul(q, k.transpose(-1, -2)) * scale

	attn = att_layer.attend(dots)
	att_mat.append(attn)

att_mat = torch.tensor(att_mat)
print(att_mat.shape)
print(att_mat)
