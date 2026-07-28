import torch
from torch import nn
from torch.nn import functional as F
from train_option import opt


class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=8):
        super(Conv_Block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.GroupNorm(num_groups=min(num_groups, out_channels), num_channels=out_channels),
            nn.Dropout2d(0.1),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.Dropout2d(0.1),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layer(x)

class DownSample(nn.Module):
    def __init__(self, channel):
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=2, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)

class UpSample(nn.Module):
    def __init__(self, channel):
        super(UpSample, self).__init__()
        self.layer = nn.Conv2d(channel, channel//2, kernel_size=1, stride=1)

    def forward(self, x, feature_map):
        # up = F.interpolate(x, scale_factor=2, mode='nearest')
        up = F.interpolate(x, size=feature_map.shape[-2:], mode='nearest')
        out = self.layer(up)

        # 如果仍有±1误差就再pad/crop
        # diffY = feature_map.size()[2] - out.size()[2]
        # diffX = feature_map.size()[3] - out.size()[3]
        # if diffX or diffY:

        return torch.cat((out, feature_map), dim=1)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, n_feats=64, num_groups=8):
        super(UNet, self).__init__()

        # in_channels = 3 * opt.frame
        # out_channels = opt.num_pixels * opt.num_wavelengths
        # n_feats = 64

        self.c1 = Conv_Block(in_channels, n_feats)
        self.d1 = DownSample(n_feats)
        self.c2 = Conv_Block(n_feats, n_feats * 2)
        self.d2 = DownSample(n_feats * 2)
        self.c3 = Conv_Block(n_feats * 2, n_feats * 4)
        self.d3 = DownSample(n_feats * 4)
        self.c4 = Conv_Block(n_feats * 4, n_feats * 8)
        self.d4 = DownSample(n_feats * 8)
        self.c5 = Conv_Block(n_feats * 8, n_feats * 16)

        self.u1 = UpSample(n_feats * 16)
        self.c6 = Conv_Block(n_feats * 16, n_feats * 8)
        self.u2 = UpSample(n_feats * 8)
        self.c7 = Conv_Block(n_feats * 8, n_feats * 4)
        self.u3 = UpSample(n_feats * 4)
        self.c8 = Conv_Block(n_feats * 4, n_feats * 2)
        self.u4 = UpSample(n_feats * 2)
        self.c9 = Conv_Block(n_feats * 2, n_feats)

        self.out = nn.Conv2d(n_feats, out_channels, kernel_size=1, stride=1)
        self.Th = nn.ReLU()


    def forward(self, x):
        R1 = self.c1(x)
        R2 = self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        R5 = self.c5(self.d4(R4))
        '''
        if not hasattr(self, 'debug_printed'):  # 只打印一次
            print('DEBUG R4 shape:', R4.shape)
            self.debug_printed = True
        '''
        O1 = self.c6(self.u1(R5, R4))
        O2 = self.c7(self.u2(O1, R3))
        O3 = self.c8(self.u3(O2, R2))
        O4 = self.c9(self.u4(O3, R1))

        return self.Th(self.out(O4))


class UNetCascade(nn.Module):
    def __init__(self, opt):
        super(UNetCascade, self).__init__()
        self.unet1 = UNet(in_channels=3 * opt.frame, out_channels=opt.num_pixels * opt.num_wavelengths, n_feats= opt.unetfeats)
        self.unet2 = UNet(in_channels=opt.num_pixels * opt.num_wavelengths, out_channels=opt.num_pixels * opt.num_wavelengths, n_feats=opt.unetfeats)

    def forward(self, x):
        out1 = self.unet1(x)
        out2 = self.unet2(out1)
        return out2

'''
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


class UNet(nn.Module):
    def __init__(self, conv=default_conv):
        super(UNet, self).__init__()

        n_channels = 3 * opt.frame
        n_classes = opt.num_pixels * opt.num_wavelengths
        n_feats = 4
        kernel_size = 3
        act = nn.ReLU(True)

        # Encoder
        self.inc = BasicBlock(conv, n_channels, n_feats, kernel_size, act=act)
        self.down1 = BasicBlock(conv, n_feats, 2*n_feats, kernel_size, stride=2, act=act)
        self.down2 = BasicBlock(conv, 2*n_feats, 4*n_feats, kernel_size, stride=2, act=act)


        # Decoder
        self.up1 = BasicBlock(conv, 4*n_feats, 2*n_feats, kernel_size, act=act)
        self.up2 = BasicBlock(conv, 2*n_feats, n_feats, kernel_size, act=act)
        self.outc = conv(n_feats, n_classes, kernel_size)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3)
        x = self.up2(x + x2)
        x = self.outc(x + x1)
        return x
'''