import torch
import torch.nn as nn
import scipy.io as scio
import numpy as np
from ADMM import A, At

'''
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   
'''


class double_conv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(double_conv, self).__init__()
        self.d_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.d_conv(x)
        return x


class Unet(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(Unet, self).__init__()

        self.dconv_down1 = double_conv(in_ch, 32)
        self.dconv_down2 = double_conv(32, 64)
        self.dconv_down3 = double_conv(64, 128)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.dconv_up2 = double_conv(64 + 64, 64)
        self.dconv_up1 = double_conv(32 + 32, 32)

        self.conv_last = nn.Conv2d(32, out_ch, 1)
        self.afn_last = nn.Tanh()

    def forward(self, x):
        inputs = x
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)

        x = self.upsample2(conv3)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample1(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        x = self.conv_last(x)
        x = self.afn_last(x)
        out = x + inputs

        return out


def generate_masks(mask_path):
    mask = scio.loadmat(mask_path + '/mask.mat')
    mask = mask['mask']
    mask = np.transpose(mask, [2, 0, 1])
    mask_s = np.sum(mask, axis=0)
    index = np.where(mask_s == 0)
    mask_s[index] = 1
    mask_s = mask_s.astype(np.uint8)
    mask = torch.from_numpy(mask)
    mask = mask.float()
    mask = mask.cuda()
    mask_s = torch.from_numpy(mask_s)
    mask_s = mask_s.float()
    mask_s = mask_s.cuda()
    return mask, mask_s


def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename

def A(x,Phi):
    temp = x*Phi
    y = torch.sum(temp,1)
    return y

def At(y,Phi):
    temp = torch.unsqueeze(y, 1).repeat(1,Phi.shape[1],1,1)
    x = temp*Phi
    return x


class ADMM_net(nn.Module):

    def __init__(self):
        super(ADMM_net, self).__init__()

        self.unet1 = Unet(8, 8)
        self.unet2 = Unet(8, 8)
        self.unet3 = Unet(8, 8)
        self.unet4 = Unet(8, 8)
        self.unet5 = Unet(8, 8)
        self.unet6 = Unet(8, 8)
        self.unet7 = Unet(8, 8)
        self.unet8 = Unet(8, 8)
        self.unet9 = Unet(8, 8)
        self.gamma1 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma2 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma3 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma4 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma5 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma6 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma7 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma8 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma9 = torch.nn.Parameter(torch.Tensor([0]))

    def forward(self, y, Phi, Phi_s):
        x_list = []
        theta = At(y, Phi)
        b = torch.zeros_like(Phi)
        ### 1-3
        yb = A(theta + b, Phi)
        x = theta + b + At(torch.div(y - yb, Phi_s + self.gamma1), Phi)
        x1 = x - b
        theta = self.unet1(x1)
        b = b - (x - theta)
        x_list.append(theta)
        yb = A(theta + b, Phi)
        x = theta + b + At(torch.div(y - yb, Phi_s + self.gamma2), Phi)
        x1 = x - b
        theta = self.unet2(x1)
        b = b - (x - theta)
        x_list.append(theta)
        yb = A(theta + b, Phi)
        x = theta + b + At(torch.div(y - yb, Phi_s + self.gamma3), Phi)
        x1 = x - b
        theta = self.unet3(x1)
        b = b - (x - theta)
        x_list.append(theta)
        ### 4-6
        yb = A(theta + b, Phi)
        x = theta + b + At(torch.div(y - yb, Phi_s + self.gamma4), Phi)
        x1 = x - b
        theta = self.unet4(x1)
        b = b - (x - theta)
        x_list.append(theta)
        yb = A(theta + b, Phi)
        x = theta + b + At(torch.div(y - yb, Phi_s + self.gamma5), Phi)
        x1 = x - b
        theta = self.unet5(x1)
        b = b - (x - theta)
        x_list.append(theta)
        yb = A(theta + b, Phi)
        x = theta + b + At(torch.div(y - yb, Phi_s + self.gamma6), Phi)
        x1 = x - b
        theta = self.unet6(x1)
        b = b - (x - theta)
        x_list.append(theta)
        ### 7-9
        yb = A(theta + b, Phi)
        x = theta + b + At(torch.div(y - yb, Phi_s + self.gamma7), Phi)
        x1 = x - b
        theta = self.unet7(x1)
        b = b - (x - theta)
        x_list.append(theta)
        yb = A(theta + b, Phi)
        x = theta + b + At(torch.div(y - yb, Phi_s + self.gamma8), Phi)
        x1 = x - b
        theta = self.unet8(x1)
        b = b - (x - theta)
        x_list.append(theta)
        yb = A(theta + b, Phi)
        x = theta + b + At(torch.div(y - yb, Phi_s + self.gamma9), Phi)
        x1 = x - b
        theta = self.unet9(x1)
        b = b - (x - theta)
        x_list.append(theta)

        output_list = x_list[-3:]
        return output_list