import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import *
from tqdm import tqdm
import torch
import numpy as np
import os


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


def save_checkpoint(model, path):
    torch.save(model.state_dict(), os.path.join(CKPT_DIR, path))


def save_model(model, epoch, data):
    torch.save(
        model, f'../new-checkpoints/{data.name}_{IMAGE_SIZE}_{PATCH_SIZE}_{epoch+1}.pth')


def check_accuracy(model, loader, split, device):
    correct = 0
    total = 0
    images, attributes, labels = None, None, None
    with torch.no_grad():
        for data in loader:
            images, attributes, labels = data
            images, attributes, labels = images.float().to(
                device), attributes.float().to(device), labels.long().to(device)
            outputs = model(images, attributes)
            c = outputs[0]
            _, predicted = torch.max(c.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    del images
    del attributes
    del labels
    del outputs
    out = f'{split.upper()} accuracy: {round(correct / total, 4)} %'
    print(out)
    return out


def train_fn(model, train_loader, epoch, criterion, optimizer, device):
    loop = tqdm(train_loader)
    model.train()
    correct = 0
    total = 0

    for batch_idx, (inputs, attributes, targets) in enumerate(loop):
        inputs = inputs.float().to(device)
        targets = targets.long().to(device)
        attributes = attributes.float().to(device)

        optimizer.zero_grad()

        predictions = model(inputs, attributes)
        (c, ap, ae) = predictions
        _, cls_pred = torch.max(c.data, 1)
        total += targets.size(0)
        correct += (cls_pred == targets).sum().item()
        loss = criterion.compute(c, targets, ap, ae)

        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())
    print(f'Loss for epoch: {epoch+1} is {loss.item()}')
    print(f'Train accuracy for epoch: {epoch+1} is {round(correct/total, 4)}')

    del inputs
    del attributes
    del targets
    del predictions
    model.eval()

    return {"epoch": epoch+1, "loss": loss.item(), "train_accuracy": round(correct/total, 4)}


def rand_augment(N=4, M=0, p=0.5, mode="all", cut_out=False):
    # Magnitude(M) search space
    shift_x = np.linspace(0, 150, 10)
    shift_y = np.linspace(0, 150, 10)
    rot = np.linspace(0, 30, 10)
    shear = np.linspace(0, 10, 10)
    sola = np.linspace(0, 256, 10)
    post = [4, 4, 5, 5, 6, 6, 7, 7, 8, 8]
    cont = [np.linspace(-0.8, -0.1, 10), np.linspace(0.1, 2, 10)]
    bright = np.linspace(0.1, 0.7, 10)
    shar = np.linspace(0.1, 0.9, 10)
    cut = np.linspace(0, 60, 10)
  # Transformation search space
    Aug = [  # 0 - geometrical
        A.ShiftScaleRotate(
            shift_limit_x=shift_x[M], rotate_limit=0,   shift_limit_y=0, shift_limit=shift_x[M], p=p),
        A.ShiftScaleRotate(
            shift_limit_y=shift_y[M], rotate_limit=0, shift_limit_x=0, shift_limit=shift_y[M], p=p),
        A.IAAAffine(rotate=rot[M], p=p),
        A.IAAAffine(shear=shear[M], p=p),
        A.InvertImg(p=p),
        # 5 - Color Based
        A.Equalize(p=p),
        A.Solarize(threshold=sola[M], p=p),
        A.Posterize(num_bits=post[M], p=p),
        A.RandomContrast(limit=[cont[0][M], cont[1][M]], p=p),
        A.RandomBrightness(limit=bright[M], p=p),
        A.IAASharpen(alpha=shar[M], lightness=shar[M], p=p)]
    # Sampling from the Transformation search space
    if mode == "geo":
        ops = np.random.choice(Aug[0:5], N)
    elif mode == "color":
        ops = np.random.choice(Aug[5:], N)
    else:
        ops = np.random.choice(Aug, N)

    if cut_out:
        ops.append(A.Cutout(num_holes=8, max_h_size=int(
            cut[M]),   max_w_size=int(cut[M]), p=p))
    trans_list = [
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.Normalize(
            mean=[NORMALIZE_MEAN for _ in range(3)],
            std=[NORMALIZE_STD for _ in range(3)],
            max_pixel_value=MAX_PIXEL_VALUE,
        ),
        ToTensorV2(),
    ]

    for item in ops:
        trans_list.insert(0, item)

    transforms = A.Compose(trans_list)
    return transforms, ops


def print2lines():
    print(" ")
    print(" ")
