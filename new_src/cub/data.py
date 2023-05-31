from cub.utils import TRAIN_TRANSFORMS
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class CUBDataset(Dataset):
    def __init__(self, files, split, transform=None):
        self.files = files
        self.split = split
        self.transforms = transform

        self.image_path = []
        self.attributes = []
        self.labels = []
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        attribute = self.attribute[idx]
        img = np.array(img)

        if self.transform is not None:
            img = self.transform(image=img)['image']
        
        return img, attribute, label
