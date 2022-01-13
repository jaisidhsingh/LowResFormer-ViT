from vit_pytorch import ViT
from config import *

model = ViT(
    image_size=IMAGE_SIZE,
    patch_size=PATCH_SIZE,
    num_classes=NUM_CLASSES,
    dim=DIM,
    heads=HEADS,
    depth=DEPTH,
    mlp_dim=MLP_DIM
)