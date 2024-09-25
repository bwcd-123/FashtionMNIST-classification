import torch
from torch import nn


class ResNetBlock(nn.Module):
    """ResNet block"""
    def __init__(self, in_c, hidden_c, out_c, kernel_size=3, stride=1, padding=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_c, hidden_c, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(hidden_c)
        self.conv2 = nn.Conv2d(hidden_c, out_c, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.use_conv = False
        if in_c != out_c:
            self.use_conv = True
            self.conv = nn.Conv2d(in_c, out_c, 1, 1, 0)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.use_conv:
            x = self.conv(x)
        out += x
        return torch.relu(out)
    

class ResNet12(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet12, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.resblocks = nn.Sequential(
            ResNetBlock(64, 64, 64),
            ResNetBlock(64, 64, 64),
            ResNetBlock(64, 64, 64),
            ResNetBlock(64, 64, 64),
            ResNetBlock(64, 64, 64)
        )
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fully_connected = nn.Linear(64, num_classes)


    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.pool1(out)
        out = self.resblocks(out)
        out = self.avg(out)
        out = self.flatten(out)
        out = self.fully_connected(out)
        return out
    

class ResNet4(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet4, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.resblock = ResNetBlock(64, 64, 64)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fully_connected = nn.Linear(64, num_classes)


    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.pool1(out)
        out = self.resblock(out)
        out = self.avg(out)
        out = self.flatten(out)
        out = self.fully_connected(out)
        return out
    

class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(6)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(16)
        self.flatten = nn.Flatten()
        self.full_connected1 = nn.Linear(256, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.full_connected2 = nn.Linear(64, num_classes)


    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.maxpool1(out)
        out = torch.relu(self.bn2(self.conv2(out)))
        out = self.maxpool1(out)
        out = self.flatten(out)
        out = torch.relu(self.bn3(self.full_connected1(out)))
        out = self.full_connected2(out)
        return out


class MLP2(nn.Module):
    """多层感知机，两层"""
    def __init__(self, hidden_units:int=196,
                 num_features=784, num_classes=10):
        super(MLP2, self).__init__()
        self.flatten = nn.Flatten()
        self.full_connected1 = nn.Linear(num_features, hidden_units)
        self.bn1 = nn.BatchNorm1d(hidden_units)
        self.full_connected2 = nn.Linear(hidden_units, num_classes)
    
    def forward(self, x):
        out = self.flatten(x)
        out = torch.relu(self.bn1(self.full_connected1(out)))
        out = self.full_connected2(out)
        return out


class MLP5(nn.Module):
    """多层感知机，五层"""
    def __init__(self, hidden_units:list=[2048, 512, 128, 32],
                 num_features=784, num_classes=10):
        super(MLP5, self).__init__()
        self.flatten = nn.Flatten()
        self.full_connected1 = nn.Linear(num_features, hidden_units[0])
        self.bn1 = nn.BatchNorm1d(hidden_units[0])
        self.full_connected2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.bn2 = nn.BatchNorm1d(hidden_units[1])
        self.full_connected3 = nn.Linear(hidden_units[1], hidden_units[2])
        self.bn3 = nn.BatchNorm1d(hidden_units[2])
        self.full_connected4 = nn.Linear(hidden_units[2], hidden_units[3])
        self.bn4 = nn.BatchNorm1d(hidden_units[3])
        self.full_connected5 = nn.Linear(hidden_units[3], num_classes)


    def forward(self, x):
        out = self.flatten(x)
        out = torch.relu(self.bn1(self.full_connected1(out)))
        out = torch.relu(self.bn2(self.full_connected2(out)))
        out = torch.relu(self.bn3(self.full_connected3(out)))
        out = torch.relu(self.bn4(self.full_connected4(out)))
        out = self.full_connected5(out)
        return out
