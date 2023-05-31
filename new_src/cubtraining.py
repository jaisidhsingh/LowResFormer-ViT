from losses import CustomLoss
from utils import * 
from data import * 
from cubmodel import *
# from config import *
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from cubconfig import *
import argparse

parse = argparse.ArgumentParser()
parse.add_argument(
	'--epochs', 
	type=int, 
	default=EPOCHS
)
parse.add_argument(
	'--batch-size', 
	type=int, 
	default=BATCH_SIZE
)
parse.add_argument(
	'--learning-rate', 
	type=float, 
	default=LEARNING_RATE
)
args = parse.parse_args()

RAND_AUGMENT, ops = rand_augment()

train_dataset = CUBDataset(CUB_FILES, split='train', transforms=RAND_AUGMENT)
test_dataset = CUBDataset(CUB_FILES, split='test', transforms=RAND_AUGMENT)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


torch.cuda.empty_cache()

# MODEL_PATH = "../new-checkpoints/awa2_32_992.pth"

model = MODEL().to(DEVICE)
# model.load_state_dict(torch.load(MODEL_PATH))
# model = MODEL().to(DEVICE)
criterion = CustomLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
print("flag")

def main():
	for epoch in range(args.epochs):
		# with open('logging.txt', 'a') as f:
			# f.write(f"EPOCH: {epoch} \n")
		
		train_fn(model, train_loader, epoch, criterion, optimizer, DEVICE)
		
		# if epoch % 10  == 9:
		check_accuracy(model, train_loader, split='train', device=DEVICE)
		check_accuracy(model, test_loader, split='test', device=DEVICE)
			# print("\n")
		
		# if epoch == 20:
			# save_checkpoint(model, epoch, train_dataset)
			# print('checkpoint saved \n')

main()

# 
# check_accuracy(model, train_loader, split='train', device=DEVICE)
# check_accuracy(model, test_loader, split='test', device=DEVICE)

