from losses import CustomLoss
import json
from utils import *
from data import *
from model import *
from config import *
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import argparse
import warnings
warnings.simplefilter('ignore')
Image.MAX_IMAGE_PIXELS = 933120000

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

train_dataset = DeepFashionDataset(TRAIN_FILES, cap=TRAIN_CAPsplit, ='train', transforms=RAND_AUGMENT)
val_dataset = DeepFashionDataset(VAL_FILES, cap=VAL_CAP, split='val', transforms=RAND_AUGMENT)
test_dataset = DeepFashionDataset(TEST_FILES, cap=TEST_CAP, split='test', transforms=RAND_AUGMENT)

train_loader = DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(
    test_dataset, batch_size=args.batch_size, shuffle=True)

print('Starting training and computing results on validation and test splits')
print('train data size: ', train_dataset.__len__())
print('val data size: ', val_dataset.__len__())
print('test data size: ', test_dataset.__len__())

model = MODEL().to(DEVICE)
criterion = CustomLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

torch.cuda.empty_cache()

with open(LOG_FILE) as f:
    prev_epochs = json.load(f)['epochs_completed']


def main():
    for epoch in range(args.epochs):
        train_info = train_fn(model, train_loader, epoch,
                              criterion, optimizer, DEVICE)
        val_acc = check_accuracy(model, val_loader, split='val', device=DEVICE)
        test_acc = check_accuracy(
            model, test_loader, split='test', device=DEVICE)

        train_info['val_accuracy'] = val_acc
        train_info['test_accuracy'] = test_acc
        
        print(" ")
        print(" ")
        with open(LOG_FILE, 'r+') as f:
            data = json.load(f)
            data['epochs_completed'] += 1
            train_info['epoch'] = data['epochs_completed']
            data['logs'].append(train_info)
            f.seek(0)
            json.dump(data, f)

        if (epoch+1) % 50 == 0:
            path = f'model_deepfashion_{IMAGE_SIZE}_{epoch+1}.pth'
            save_checkpoint(model, path)
            print(f'checkpoint saved at epoch: {epoch+1}')


main()
