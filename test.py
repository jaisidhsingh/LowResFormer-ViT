import torch
from model_utils import PatchEmbedding, Block, VisionTransformer

def test_block():
	x = torch.randn((1, 3, 224, 224))
	pe = PatchEmbedding()(x)

	b = Block()(pe)
	return b.shape

def test_ViT():
	x = torch.randn((1, 3, 224, 224))
	pe = PatchEmbedding()(x)

	t = VisionTransformer()(pe)
	return t.shape

assert torch.Size([1, 197, 768]) == test_block() , "Tensor shapes do not match, check forward pass"
assert torch.Size([1, 197, 768]) == test_ViT() , "Tensor shapes do not match, check forward pass"

