import torch.nn as nn
import torch

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=3, stride=1, padding=1),  # input[3, 32, 32]  output[48, 32, 32]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # output[48, 16, 16]
            nn.Conv2d(48, 96, kernel_size=3, stride=1, padding=1),  # output[96, 16, 16]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # output[96, 8, 8]
            nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1),  # output[192, 8, 8]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),  # output[192, 8, 8]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 96, kernel_size=3, stride=1, padding=1),  # output[96, 8, 8]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # output[96, 4, 4]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(96*4*4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            #输出的类别总共有10个
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x