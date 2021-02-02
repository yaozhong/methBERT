import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=2)
		self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2)

		self.pool = nn.MaxPool1d(2, stride=1)

		self.bn1 = nn.BatchNorm1d(num_features=32)
		self.bn2 = nn.BatchNorm1d(num_features=64)
		
		self.fc1 = nn.Linear(64*1*86 , 128)
		self.fc2 = nn.Linear(128, 2)

	def forward(self, x):

		x = self.pool(F.relu(self.bn1(self.conv1(x))))
		x = self.pool(F.relu(self.bn2(self.conv2(x))))

		## checking the view function
		x = x.view(x.shape[0], -1)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		
		return x



# use the convolutional part network structure form Deepsignal
class DS_conv(nn.Module):
	def __init__(self):
		super(DS_conv, self).__init__()
		self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1)
		self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

		self.pool = nn.MaxPool1d(2, stride=1)

		self.bn1 = nn.BatchNorm1d(num_features=64)
		#self.bn2 = nn.BatchNorm1d(num_features=64)
		
		self.fc1 = nn.Linear(64*1*348 , 128)
		self.fc2 = nn.Linear(128, 2)

	def forward(self, x):

		x = self.pool(F.relu(self.bn1(self.conv1(x))))
		x = self.pool(F.relu(self.bn1(self.conv2(x))))
		x = self.pool(F.relu(self.bn1(self.conv2(x))))
		x = self.pool(F.relu(self.bn1(self.conv2(x))))

		## checking the view function
		x = x.view(x.shape[0], -1)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		
		return x


# todo tranform the time series data to the image data and use image classification techniques

