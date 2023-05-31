import torch.nn as nn
import torch
from config import *
from warnings import simplefilter
simplefilter('ignore')


"""
METHOD:

> Pass the softmaxed outputs as Xn into bias_loss
> Pass torch.ones_like(Xn) as Yn into bias_loss
> Yields sum(-log(Xn))
"""

class CustomLoss():
	def __init__(self):

		self.attr_loss = nn.KLDivLoss()
		self.cls_loss = nn.CrossEntropyLoss()
		self.bias_loss = nn.BCELoss(reduction='sum')

	def compute(self, 
		cls_preds, 
		cls_targets, 
		attr_preds, 
		attr_targets
	):

		cls_loss = self.cls_loss(cls_preds, cls_targets)
		soft_attr_preds = torch.softmax(attr_preds, dim=1)
		soft_attr_targets = torch.softmax(attr_targets, dim=1)

		attr_loss = self.attr_loss(soft_attr_preds, soft_attr_targets)
		softmaxed_cls_preds = torch.softmax(cls_preds, dim=1)
		bias_loss = self.bias_loss(softmaxed_cls_preds, torch.ones_like(softmaxed_cls_preds))

		return ALPHA*cls_loss + BETA*attr_loss + GAMMA*bias_loss
