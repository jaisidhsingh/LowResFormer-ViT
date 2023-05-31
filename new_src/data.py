import torch
import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from config import *
import random
from cubconfig import *

class AwA2Dataset(Dataset):
	def __init__(self, images_dir, label_file, attribute_file, train_cap=AWA_TRAIN_CAP, test_cap=AWA_TEST_CAP, split=None, transforms=None):
		self.name = 'awa2'
		self.images_dir = images_dir
		self.label_file = label_file
		self.transforms = transforms
		self.split = split
		self.train_cap = train_cap
		self.test_cap = test_cap
		self.images = []
		self.labels = []
		self.attributes = []

		with open(self.label_file, 'r') as f:
			for line in f.readlines():
					class_id = line.split()[1]
					self.labels.append(class_id)

		for item in os.listdir(self.images_dir):
			for f in os.listdir(self.images_dir+item+"/"):
					self.images.append(self.images_dir+item+'/'+f)

		with open(attribute_file, 'r') as f:
			for line in f.readlines():
					attributes = line.split()
					attributes = [float(x) for x in attributes]
					self.attributes.append(attributes)
		
		random.shuffle(self.images)		
		self.train_images = self.images[: self.train_cap]
		self.test_images = self.images[self.train_cap : self.train_cap + self.test_cap]
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
		attribute = None

		if self.split == 'train':
			img_path = self.train_images[idx]
			cls_label = img_path.split('/')[3]
			label = int(self.labels.index(cls_label))
			attribute = self.attributes[label]
			one_hot = float(label)

		if self.split == 'test':
			img_path = self.test_images[idx]
			cls_label = img_path.split('/')[3]
			label = int(self.labels.index(cls_label))
			attribute = self.attributes[label]
			one_hot = float(label)

		one_hot = torch.tensor(one_hot)
		image = np.array(Image.open(img_path).convert('RGB'))
		if self.transforms is not None:
			image = self.transforms(image=image)['image']
		return image, torch.tensor(attribute), one_hot


class CUBDataset(Dataset):
	def __init__(self, files, split, train_cap=CUB_TRAIN_CAP, test_cap=CUB_TEST_CAP, transforms=None):
		self.name = 'cub200'
		self.dataset_dir = files['dataset_dir']
		self.image_file = files['image_file']
		self.label_file = files['label_file']
		self.attribute_file = files['attribute_file']
		self.image_keys = files['image_keys']
		self.train_cap = train_cap
		self.test_cap = test_cap
		self.split = split
		self.transforms = transforms

		self.images = []
		self.labels = []
		self.attribute_dim = 312

		with open (self.image_file, 'r') as f:
			for line in f.readlines():
				self.images.append(line.strip())
		
		with open(self.label_file, 'r') as f:
			for line in f.readlines():
				self.labels.append(line.strip())

		self.img_keys = {}
		
		with open (self.image_keys, 'r') as f:
			for line in f.readlines():
				line = line.split(" ")
				self.img_keys[line[1].strip()] = int(line[0])

		self.attributes = torch.zeros((len(self.img_keys), self.attribute_dim))		
		with open(self.attribute_file, 'r') as f:
			for line in f.readlines():
				line = line.split(" ")
				img_key = int(line[0])
				attr_key = int(line[1])
				present = int(line[2])
				self.attributes[img_key][attr_key] = present


		random.shuffle(self.images)
		self.train_images = self.images[0 : self.train_cap]
		self.test_images = self.images[self.train_cap : self.train_cap + self.test_cap]	

	def __len__(self):
		if self.split == 'train':
			return len(self.train_images)
		
		if self.split == 'test':
			return len(self.test_images)

	def __getitem__(self, idx):
		img_path = None
		one_hot = None
		attribute = None

		if self.split == 'train':
			img_path = self.images[idx]
			cls_label = img_path.split('/')[0]
			label_idx = self.labels.index(cls_label)
			attr_idx = self.img_keys[img_path]
			attribute = self.attributes[attr_idx]
			one_hot = float(label_idx)	

		if self.split == 'test':
			img_path = self.test_images[idx]
			cls_label = img_path.split('/')[0]	
			label_idx = self.labels.index(cls_label)
			attr_idx = self.img_keys[img_path]
			attribute = self.attributes[attr_idx]
			one_hot = float(label_idx)
		
		one_hot = torch.tensor(one_hot)
		image = np.array(Image.open(self.dataset_dir+img_path).convert('RGB'))
		if self.transforms is not None:
			image = self.transforms(image=image)['image']
		return image, attribute, one_hot

