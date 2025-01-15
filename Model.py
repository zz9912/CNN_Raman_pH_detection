
from torch import nn
from torch.nn import functional as F, init
import time
import torch
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = shortcut

    def forward(self, x):
        residual = x if self.shortcut is None else self.shortcut(x)
        out = self.left(x)
        out += residual
        return F.relu(out)

class ResNet(nn.Module):
    def __init__(self, layers,num_classes1=1, num_classes2 = 1):
        super(ResNet, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )


        self.layer1 = self.make_layer(64, 64, layers[0])
        self.layer2 = self.make_layer(64, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(128, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(256, 512, layers[3], stride=2)

        self.dropout = nn.Dropout(0.5)


        self.fc1 = nn.Linear(512, num_classes1)
        self.fc2 = nn.Linear(512, num_classes2)

    def make_layer(self, in_channels, out_channels, block_num, stride=1):
        shortcut = None
        if stride != 1 or in_channels != out_channels:
            shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, shortcut))
        for _ in range(1, block_num):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)

        output1 = self.fc1(x)
        output2 = self.fc2(x)


        return output1, output2


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.01)
                init.constant_(m.bias, 0)

def resnet(layer_number,num1=1,num2=1):
    return ResNet(layer_number, num_classes1=num1,num_classes2=num2)

