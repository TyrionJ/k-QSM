import torch.nn as nn

ks = (3, 3, 3)
sd = (1, 1, 1)
pd = (1, 1, 1)


class LFDQSM(nn.Module):
    def __init__(self):
        super(LFDQSM, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=32, kernel_size=ks, stride=sd, padding=pd),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        self.res_blocks = nn.Sequential()
        for i in range(0, 8):
            self.res_blocks.add_module(name=f'block{i+1}', module=WideResBlock(32))

        self.layer2 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=ks, stride=sd, padding=pd),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=ks, stride=sd, padding=pd),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.layer4 = nn.Conv3d(in_channels=32, out_channels=2, kernel_size=ks, stride=sd, padding=pd)

    def forward(self, X):
        out = self.layer1(X)
        del X
        out = self.res_blocks(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        return out


class WideResBlock(nn.Module):

    def __init__(self, channels):
        super(WideResBlock, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=ks, stride=sd, padding=pd)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.dropout = nn.Dropout3d(p=0.2)

        self.conv2 = nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=ks, stride=sd, padding=pd)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, X_in):
        out = self.conv1(X_in)
        out = self.relu1(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.relu2(out + X_in)

        return out
