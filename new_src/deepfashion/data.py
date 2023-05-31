import torch
import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset
import random
from config import *
Image.MAX_IMAGE_PIXELS = 933120000


class DeepFashionDataset(Dataset):
    def __init__(self, files, cap, split, transforms=None):
        self.files = files
        self.cap = cap
        self.split = split
        self.transforms = transforms

        self.image_paths = []
        self.attributes = []
        self.labels = []

        with open(self.files['attr']) as f:
            for c, line in enumerate(f.readlines()):
                if c < 2:
                    continue
                line = line.split(" ")
                img_path = line[0][3:]
                attribute = line[1:]
                img_path = os.path.join(IMAGE_FOLDER, img_path)
                self.image_paths.append(img_path)
                self.attributes.append(attribute)
        
        with open(self.files['cats']) as f:
            for c, line in enumerate(f.readline()):
                if c < 2:
                    continue
                line = line.split(" ")
                label = float(line[1])
                self.labels.append(label)        

        self.image_paths = self.image_paths[self.cap[0], self.cap[1]]
        self.attributes = self.attributes[self.cap[0], self.cap[1]]
        self.labels = self.labels[self.cap[0], self.cap[1]]

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        attribute = torch.tensor(self.attributes[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        
        return image, attribute, label


class DFDataset(Dataset):
    def __init__(self, dirname, split, transforms=None):
        self.dir = dirname
        self.split = split
        self.transforms = transforms

        self.img_paths = []
        self.labels = []
        self.attributes = []

        with open(os.path.join(self.dir, "main.txt")) as f:
            for line in f.readlines():
                path = line[4:].strip()
                path = os.path.join(IMAGE_FOLDER, path)
                self.img_paths.append(path)

        with open(os.path.join(self.dir, "cate.txt")) as f:
            for line in f.readlines():
                label = int(line.strip())
                self.labels.append(label)

        with open(os.path.join(self.dir, "attr.txt")) as f:
            for line in f.readlines():
                attr = [float(x) for x in line.split()]
                self.attributes.append(attr)

        self.attributes = torch.from_numpy(np.array(self.attributes))


        if self.split == 'train':
            self.img_paths = self.img_paths[:10000]
            self.labels = self.labels[:10000]
            self.attributes = self.attributes[:10000]
        else:
            self.img_paths = self.img_paths[:1000]
            self.labels = self.labels[:1000]
            self.attributes = self.attributes[:1000]
    
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        label = self.labels[idx]
        attribute = self.attributes[idx]

        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        return img, attribute, label
