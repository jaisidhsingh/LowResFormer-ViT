import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from data import DeepFashionDataset
from config import *
from utils import *
from model import *
import warnings
warnings.simplefilter('ignore')


model = MODEL().to(DEVICE)
# model.load_state_dict(torch.load("./checkpoints/model_deepfashion_32_1.pth", map_location=DEVICE))

print(next(model.parameters()).is_cuda)

RAND_AUGMENT, ops = rand_augment()
train_dataset = DeepFashionDataset(files=TRAIN_FILES, cap=[0, 5000], split='train', transforms=RAND_AUGMENT)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

for (i, a, l) in train_loader:
    print(i.shape, a.shape, l.shape)
print(len(train_dataset))