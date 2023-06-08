import torch.nn as nn


class up(nn.Module):
    '''(deconv => BN => ReLU)'''

    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)

        return x


class VarDecoder(nn.Module):
    def __init__(self, n_ch):
        super(VarDecoder, self).__init__()
        self.upsample1 = nn.Upsample(size=(89, 89), mode='bilinear')
        self.up1 = up(n_ch, 128)
        self.up2 = up(128, 64)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up3 = up(64, 64)
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=3, padding=1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.upsample1(x)  # torch.Size([None, 64, 89, 89])
        x = self.up1(x)  # torch.Size([None, 128, 89, 89])
        x = self.up2(x)  # torch.Size([None, 64, 89, 89])
        x = self.upsample2(x)  # torch.Size([None, 64, 178, 178])
        x = self.up3(x)  # torch.Size([None, 64, 178, 178])
        x = self.up4(x)  # torch.Size([None, 3, 178, 178])
        x = self.sigmoid(x)

        return x
