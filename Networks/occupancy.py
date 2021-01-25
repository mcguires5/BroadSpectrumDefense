import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F


class occupancy_classifier(nn.Module):
    def __init__(self, data_length, num_classes):
        super(occupancy_classifier, self).__init__()
        self.fcc1 = nn.Linear(data_length, 32)
        #self.fcc2 = nn.Linear(128, 64)
        self.fcc3 = nn.Linear(32, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.fcc1(x))
        #x = F.relu(self.fcc2(x))
        x = self.fcc3(x)
        x_pred = self.softmax(x)
        return x