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

train_dataset = AwA2Dataset(IMAGES_DIR, LABEL_FILE, split='train', transforms=train_transforms)
test_dataset = AwA2Dataset(IMAGES_DIR, LABEL_FILE, split='test', transforms=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scaler = torch.cuda.amp.GradScaler()


def check_accuracy(model, loader, split='train'):
	correct = 0
	total = 0

	with torch.no_grad():
		for data in loader:
			images, labels = data
			outputs = model(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

	print(f'{split.upper()} Accuracy: {100 * correct / total} %')

def train_loop(model, train_loader, epoch, criterion, optimizer, scaler, device):
	loop = tqdm(train_loader)
	model.train()
	
	for batch_idx, (inputs, targets) in enumerate(loop):
		inputs = inputs.to(device)
		targets = targets.unsqueeze(1).to(device)

		with torch.cuda.amp.autocast():
			predictions = model(inputs).to(device)
			loss = criterion(predictions, targets.long())
		
		optimizer.zero_grad()
		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()

		loop.set_postfix(loss=loss.item())
	print(f'Loss for epoch: {epoch} is {loss.item()}')
	model.eval()	

for epoch in range(EPOCHS):
	train_loop(model, train_loader, epoch, criterion, optimizer, scaler, DEVICE)
	check_accuracy(model, train_loader, split='train')
	check_accuracy(model, test_loader, split='test')