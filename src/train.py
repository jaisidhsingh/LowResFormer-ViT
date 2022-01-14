from utils import * 
from data import AwA2Dataset
from model import model
from config import *
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
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

train_dataset = AwA2Dataset(IMAGES_DIR, LABEL_FILE, split='train', transforms=TRAIN_TRANSFORMS)
test_dataset = AwA2Dataset(IMAGES_DIR, LABEL_FILE, split='test', transforms=TEST_TRANSFORMS)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
scaler = torch.cuda.amp.GradScaler()

def main():
	for epoch in range(args.epochs):
		train_fn(model, train_loader, epoch, criterion, optimizer, scaler, DEVICE)
		check_accuracy(model, train_loader, split='train', device=DEVICE)
		check_accuracy(model, test_loader, split='test', device=DEVICE)
		print("\n")
		
		if epoch % 10 == 9:
			save_checkpoint(model, epoch, train_dataset)
			print('checkpoint saved \n')


main()