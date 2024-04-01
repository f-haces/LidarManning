# NEURAL NETWORK
import torch
import torch.nn as nn
import torch.nn.functional as F

class MRNN(nn.Module):
    # ORIGINALLY ADAPTED FROM https://github.com/fxia22/pointnet.pytorch/blob/master/pointnet/model.py UNDER MIT LICENSE
    def __init__(self, num_outputs=1, transform=False):
        super(MRNN, self).__init__()
        self.use_transforms = transform
        if self.use_transforms:
            self.transform = Transform()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 256, 1)
        self.conv3 = nn.Conv1d(256, 2048, 1)
        self.conv4 = nn.Conv1d(2048, 2048, 1)
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_outputs)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(2048)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        if self.use_transforms:
            x = self.transform(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv4(F.relu(self.bn3(self.conv3(x))))
        
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 2048)
        
        x = self.fc1(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        return x