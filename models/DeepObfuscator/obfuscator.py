import torch.nn as nn


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)

        return x


class Encoder(nn.Module):
    def __init__(self, n_channels):
        super(Encoder, self).__init__()
        self.down1 = down(n_channels, 64)
        self.down2 = down(64, 64)
        self.max_pool = nn.MaxPool2d(2)
        self.down3 = down(64, 128)
        self.down4 = down(128, 128)

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.max_pool(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.max_pool(x)

        return x
