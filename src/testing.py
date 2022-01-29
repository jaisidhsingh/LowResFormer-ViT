import torch
from model import *
from config import *
from losses import CustomLoss

cls_targets = torch.tensor([3]).long()
attr_targets = torch.randn((1, 64))
criterion = CustomLoss()

x = torch.randn((1, 3, 32, 32))
a = torch.randn((1, 85))

model = FinalModel()
(c, ap, ae) = model(x, a)

loss = criterion.compute(c, cls_targets, ap, ae)
print("Loss: ", loss.item())