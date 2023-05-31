from losses import CustomLoss
from utils import * 
from data import AwA2Dataset
from model import *
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

RAND_AUGMENT, ops = rand_augment()

train_dataset = AwA2Dataset(AWA_IMAGES_DIR, AWA_LABEL_FILE, AWA_ATTRIBUTE_FILE, split='train', transforms=RAND_AUGMENT)
test_dataset = AwA2Dataset(AWA_IMAGES_DIR, AWA_LABEL_FILE, AWA_ATTRIBUTE_FILE, split='test', transforms=RAND_AUGMENT)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

MODEL_PATH = "../new-checkpoints/awa2_16_2_3.pth"
model = torch.load(MODEL_PATH)
# print(model.children)
# model = MODEL().to(DEVICE)
# model.load_state_dict(torch.load(MODEL_PATH))
# model = MODEL().to(DEVICE)
criterion = CustomLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)


print("loaded")
# check_accuracy(model, test_loader, split='test', device=DEVICE)
def main():
	for epoch in range(args.epochs):
		with open('logging.txt', 'a') as f:
			f.write(f"EPOCH: {epoch} \n")
		
		train_fn(model, train_loader, epoch, criterion, optimizer, DEVICE)

		# if epoch % 10 == 9:
		check_accuracy(model, test_loader, split='test', device=DEVICE)
			# print("\n")
		
		# if epoch % 10 == 9:
		# save_model(model, epoch, train_dataset)
			# print('checkpoint saved \n')

main()
# check_accuracy(model, train_loader, split='train', device=DEVICE)
