import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import *
from tqdm import tqdm


TRAIN_TRANSFORMS = A.Compose(
	[
		A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
		A.Normalize(
         mean=[NORMALIZE_MEAN for _ in range(3)],
         std=[NORMALIZE_STD for _ in range(3)],
      	max_pixel_value=MAX_PIXEL_VALUE,
      ),
		ToTensorV2()
	]
)

TEST_TRANSFORMS = A.Compose(
	[
		A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
		A.Normalize(
         mean=[NORMALIZE_MEAN for _ in range(3)],
         std=[NORMALIZE_STD for _ in range(3)],
      	max_pixel_value=MAX_PIXEL_VALUE,
      ),
		ToTensorV2()
	]
)

def save_checkpoint(model, epoch, data):
	torch.save(model.state_dict(), f'../checkpoints/{data.name}_{IMAGE_SIZE}_{epoch}.pth')

def check_accuracy(model, loader, split, device):
	correct = 0
	total = 0

	with torch.no_grad():
		for data in loader:
			images, labels = data
			images, labels = images.to(device), labels.to(device)
			outputs = model(images).to(device)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

	print(f'{split.upper()} accuracy: {100 * correct / total} %')

def train_fn(model, train_loader, epoch, criterion, optimizer, scaler, device):
	loop = tqdm(train_loader)
	model.train()
	
	for batch_idx, (inputs, targets) in enumerate(loop):
		inputs = inputs.to(device)
		targets = targets.long().to(device)

		optimizer.zero_grad()
		
		with torch.cuda.amp.autocast():
			predictions = model(inputs).to(device)
			loss = criterion(predictions, targets)

		loss.backward()
		optimizer.step()	
		# scaler.scale(loss).backward()
		# scaler.step(optimizer)
		# scaler.update()

		loop.set_postfix(loss=loss.item())
	print(f'Loss for epoch: {epoch+1} is {loss.item()}')
	model.eval()	

