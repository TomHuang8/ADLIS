import torch.nn as nn

from test_code.test_option import opt


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
        n_classes = opt.num_pixels * opt.wavelengths
        n_feats = 64
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
