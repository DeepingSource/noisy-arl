import torch.nn as nn
import torch


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class fc(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(fc, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_ch, out_ch),
            nn.ReLU(inplace=True),
            nn.Linear(out_ch, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class Classifier(nn.Module):
    def __init__(self, n_channels, num_classes, is_cifar10=False):
        super(Classifier, self).__init__()
        self.down1 = down(n_channels, 256)
        self.max_pool = nn.MaxPool2d(2)
        self.down2 = down(256, 512)
        self.down3 = down(512, 512)
        if is_cifar10:
            self.down4 = fc(512, 4096)
        else:
            self.down4 = fc(12800, 4096)
        self.down5 = nn.Linear(4096, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.down1(x)
        x = self.max_pool(x)
        x = self.down2(x)
        x = self.max_pool(x)
        x = self.down3(x)
        x = self.max_pool(x)
        x = torch.flatten(x, 1)
        x = self.down4(x)
        x = self.down5(x)

        return x
