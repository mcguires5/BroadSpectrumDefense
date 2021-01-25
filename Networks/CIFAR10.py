import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F

class StandardCIFAR10(nn.Module):
    def __init__(self):
        super(StandardCIFAR10, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 128, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 5 * 5, 256)
        #self.fc1 = nn.Linear(128 * 24 * 24, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)
        self.dropout1 = nn.Dropout(.5)
        self.dropout2 = nn.Dropout(.5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(F.relu(self.conv2(x)))
        #x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool2(F.relu(self.conv4(x)))
        #x = F.relu(self.conv4(x))
        x = x.reshape(-1, 128 * 5 * 5)
        #x = x.view(-1, 128 * 24 * 24)
        x = F.relu(self.dropout1(self.fc1(x)))
        x = F.relu(self.dropout2(self.fc2(x)))
        x = self.fc3(x)
        x_pred = self.softmax(x)
        return x

class MagNet(nn.Module):
    def __init__(self):
        super(MagNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, 3)
        self.conv2 = nn.Conv2d(96, 96, 3)
        self.conv3 = nn.Conv2d(96, 96, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(96, 192, 3)
        self.conv5 = nn.Conv2d(192, 192, 3)
        self.conv6 = nn.Conv2d(192, 192, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv7 = nn.Conv2d(192, 192, 3)
        self.conv8 = nn.Conv2d(192, 192, 1)
        self.conv9 = nn.Conv2d(192, 10, 1)
        self.globpool = nn.AvgPool2d(10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool1(x)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool2(x)
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = x.view(-1, 10)
        x_pred = self.softmax(x)
        return x

