import torch
import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class AwA2Dataset(Dataset):
	def __init__(self, images_dir, label_file, split=None, transforms=None):
		self.images_dir = images_dir
		self.label_file = label_file
		self.transforms = transforms
		self.split = split
		self.images = []
		self.labels = []
		with open(self.label_file, 'r') as f:
			for line in f.readlines():
					class_id = line.split()[1]
					self.labels.append(class_id)

		for item in os.listdir(self.images_dir):
			for f in os.listdir(self.images_dir+item+"/"):
					self.images.append(self.images_dir+item+'/'+f)
		
		self.train_images = self.images[:2500]
		self.test_images = self.images[2500:3500]
		self.dump_train_image_paths('../Animals_with_Attributes2/train_image_paths.txt')
		self.dump_test_image_paths('../Animals_with_Attributes2/test_image_paths.txt')

	def dump_train_image_paths(self, filename):
		with open(filename, 'w') as f:
			for img_path in self.train_images:
					f.write(img_path+'\n')
	
	def dump_test_image_paths(self, filename):
		with open(filename, 'w') as f:
			for img_path in self.test_images:
				f.write(img_path+'\n')
			
	def __len__(self):
		if self.split == 'train':
			return len(self.train_images)
		
		if self.split == 'test':
			return len(self.test_images)

	def __getitem__(self, idx):
		image, one_hot = None, None

		if self.split == 'train':
			img_path = self.train_images[idx]
			cls_label = img_path.split('/')[3]
			label = float(self.labels.index(cls_label))
			# one_hot = [0.0 for _ in range(len(self.labels))]
			# one_hot[label] = 1.0
			one_hot = label

		if self.split == 'test':
			img_path = self.test_images[idx]
			cls_label = img_path.split('/')[3]
			label = float(self.labels.index(cls_label))
			# one_hot = [0.0 for _ in range(len(self.labels))]
			# one_hot[label] = 1.0
			one_hot = label

		one_hot = torch.tensor(one_hot)
		image = np.array(Image.open(img_path).convert('RGB'))
		if self.transforms is not None:
			image = self.transforms(image=image)['image']
		return image, one_hot


