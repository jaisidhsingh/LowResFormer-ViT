import torch
from model import *
from config import *
from losses import CustomLoss
from utils import *
from data import *
from cubconfig import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

RAND_AUGMENT, ops = rand_augment()
train_dataset = CUBDataset(CUB_FILES, split='train', transforms=RAND_AUGMENT)
print("data init")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

for (i, a, t) in train_loader:
	i = i.cpu().detach().numpy()[0]
	i = i.reshape(i.shape[1], i.shape[2], i.shape[0])
	plt.imshow(i)
	plt.show()
	print(a)
	print(a.shape)
	print(t)
	print(t.shape)
	break
