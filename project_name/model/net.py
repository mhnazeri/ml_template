import torch
import torch.nn as nn


class CustomModel(nn.Module):

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self.layer1 = nn.Conv2d(cfg.in_features, 16, kernel_size=3, bias=False, padding=1)
		self.bn1 = nn.BatchNorm2d(16)
		self.layer2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
		self.fc = nn.Linear(8 * 28 * 28, cfg.out_features)
		
	def forward(self, x):
		bsz = x.size(0)
		x = torch.relu(self.bn1(self.layer1(x)))
		x = torch.relu(self.layer2(x))
		x = x.view(bsz, -1)
		return self.fc(x)
