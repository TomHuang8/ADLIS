import torch.nn as nn
import torch
import torch.nn.functional as F
from train_option import opt
from torch.nn import init


class CNN(nn.Module):
    def __init__(self, in_channels=opt.frame * 3, out_channels=opt.num_pixels * opt.num_wavelengths, dropout=False,
                 num_blocks=30):
        super(CNN, self).__init__()
        self.use_dropout = dropout
        self.device = None

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

        dfus_blocks = [dfus_block(dim=256 + 32 * i) for i in range(num_blocks)]
        self.dfus_blocks = nn.Sequential(*dfus_blocks)

        self.conv_out = nn.Conv2d(256 + 32 * num_blocks, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

        if dropout:
            self.dropout = nn.Dropout2d(p=0.5)

        self.apply(self._weight_init)

    @staticmethod
    def _weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.zeros_(m.bias)

    def forward(self, x):
        if self.device is None:
            self.device = x.device
            self.to(self.device)

        feats = self.features(x)

        if self.use_dropout and self.training:
            feats = self.dropout(feats)

        fused_feats = self.dfus_blocks(feats)
        out = self.conv_out(fused_feats)
        return out


class dfus_block(nn.Module):
    def __init__(self, dim):
        super(dfus_block, self).__init__()
        self.conv1 = nn.Conv2d(dim, 128, 1, 1, 0, bias=False)

        self.conv_up1 = nn.Conv2d(128, 32, 3, 1, 1, bias=False)
        self.conv_up2 = nn.Conv2d(32, 16, 1, 1, 0, bias=False)

        self.conv_down1 = nn.Conv2d(128, 32, 3, 1, 1, bias=False)
        self.conv_down2 = nn.Conv2d(32, 16, 1, 1, 0, bias=False)

        self.conv_fution = nn.Conv2d(96, 32, 1, 1, 0, bias=False)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.relu(self.conv1(x))
        feat_up1 = self.relu(self.conv_up1(feat))
        feat_up2 = self.relu(self.conv_up2(feat_up1))
        feat_down1 = self.relu(self.conv_down1(feat))
        feat_down2 = self.relu(self.conv_down2(feat_down1))
        feat_fution = torch.cat([feat_up1, feat_up2, feat_down1, feat_down2], dim=1)
        feat_fution = self.relu(self.conv_fution(feat_fution))
        out = torch.cat([x, feat_fution], dim=1)
        return out
