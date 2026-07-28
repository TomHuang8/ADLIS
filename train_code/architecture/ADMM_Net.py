import torch
import torch.nn as nn
import torch.nn.functional as F
from train_option import opt

def A(x,Phi):
    temp = x*Phi
    y = torch.sum(temp,1)
    return y

'''def At(y,Phi):
    temp = torch.unsqueeze(y, 1).repeat(1,Phi.shape[1],1,1)
    x = temp*Phi
    return x'''
def At(y, Phi):
    """
    y: (B,H,W)      —— 老格式
       (B,C,H,W)    —— 新格式，且 C == Phi.shape[1]
    """
    if y.dim() == 4:
        return y * Phi                    # 直接逐元素乘
    elif y.dim() == 3:
        return y.unsqueeze(1).repeat(1, Phi.shape[1], 1, 1) * Phi
    else:
        raise ValueError(f'y 维度异常: {y.shape}')

class double_conv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(double_conv, self).__init__()
        self.d_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.d_conv(x)
        return x


class Unet(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(Unet, self).__init__()

        n_feats = 32

        # self.inp_proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

        self.dconv_down1 = double_conv(in_ch, n_feats)
        self.dconv_down2 = double_conv(n_feats, n_feats * 2)
        self.dconv_down3 = double_conv(n_feats * 2, n_feats * 4)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(n_feats * 4, n_feats * 2, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(n_feats * 2, n_feats, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.dconv_up2 = double_conv(n_feats * 2 + n_feats * 2, n_feats * 2)
        self.dconv_up1 = double_conv(n_feats + n_feats, n_feats)

        self.conv_last = nn.Conv2d(n_feats, out_ch, 1)
        self.afn_last = nn.Tanh()

    def forward(self, x):
        b, c, h_inp, w_inp = x.shape
        # inputs = x
        hb, wb = 8, 8
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
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

        '''if x.shape != inputs.shape:
            inp_pad = F.pad(inputs, [0, pad_w, 0, pad_h], mode='reflect')
            inp_proj = self.inp_proj(inp_pad)
            out = x + inp_proj
        else:'''
        out = x + inputs

        return out[:, :, :h_inp, :w_inp]

def shift_3d(inputs,step=2):
    [bs, nC, row, col] = inputs.shape
    for i in range(nC):
        inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:], shifts=step*i, dims=2)
    return inputs

def shift_back_3d(inputs,step=2):
    [bs, nC, row, col] = inputs.shape
    for i in range(nC):
        inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:], shifts=(-1)*step*i, dims=2)
    return inputs

class ADMM_net(nn.Module):
    """
        - 测量域 C_meas = 3*opt.frame = 6
        - 潜表示域 C_lat  = feats (保持统一，默认 96)
        - 输出域   C_out  = opt.num_pixels * opt.num_wavelengths = 225
        ADMM 运算全部在测量域完成；U‑Net 在潜表示域完成。
    """
    def __init__(self):
        super(ADMM_net, self).__init__()

        # opt.admm_unet_feats = 96
        feats = opt.admm_unet_feats
        in_channel = 3 * opt.frame
        out_channel = opt.num_pixels * opt.num_wavelengths

        # --- 域间映射 (共享) ---
        # 测量域 -> 潜表示域
        self.lift_m2l = nn.Conv2d(in_channel, feats, 1, bias=False)
        # 潜表示域 -> 测量域
        self.drop_l2m = nn.Conv2d(feats, in_channel, 1, bias=False)
        # 潜表示域 -> 输出域 (最终重建)
        self.head_l2out = nn.Conv2d(feats, out_channel, 1, bias=False)

        self.unet1 = Unet(feats, feats)
        self.unet2 = Unet(feats, feats)
        self.unet3 = Unet(feats, feats)
        self.unet4 = Unet(feats, feats)
        self.unet5 = Unet(feats, feats)
        self.unet6 = Unet(feats, feats)
        self.unet7 = Unet(feats, feats)
        self.unet8 = Unet(feats, feats)
        self.unet9 = Unet(feats, feats)
        self.gamma1 = torch.nn.Parameter(torch.Tensor([0]))  # 初始化值为 0 的可学习参数
        self.gamma2 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma3 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma4 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma5 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma6 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma7 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma8 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma9 = torch.nn.Parameter(torch.Tensor([0]))

        self.eps = 1e-8  # 避免除零

    def forward(self, y, input_mask=None):
        # 解析 y 的形状，并在需要时把 4‑D → 3‑D
        if y.dim() == 4:  # (B, C, H, W)
            B, C, H_inp, W_inp = y.shape  # 先记下原始通道数
            y = y.sum(dim=1)  # → (B, H, W) ＊核心改动＊
        elif y.dim() == 3:  # (B, H, W)
            B, H_inp, W_inp = y.shape
            C = 3 * opt.frame  # 通道数只能从配置里取
        else:
            raise ValueError(f'y 维度异常: {y.shape}')

        dev = y.device

        if input_mask is None:
            '''Phi = torch.rand((1, 28, 256, 310)).cuda()
            Phi_s = torch.rand((1, 256, 310)).cuda()'''
            Phi = torch.ones(B, C, H_inp, W_inp, device=dev)
            Phi_s = torch.ones(B, H_inp, W_inp, device=dev) * C
        else:
            Phi, Phi_s = input_mask
            # Phi.shape[1] = C  # 防止以后通道数变化
            assert Phi.shape[1] == C, \
                f"Phi 通道数 {Phi.shape[1]} 与配置测量通道 {C} 不一致"
        x_list = []
        # 归一化
        # scale = float(C)
        # y = y / scale

        theta = At(y,Phi)
        b = torch.zeros_like(Phi)
        ### 1-3
        yb = A(theta+b,Phi)
        # y_dot = y - yb
        # a = torch.div(y_dot,Phi_s+self.gamma1)
        # x = theta + b + At(a, Phi)
        x = theta+b + At(torch.div(y-yb,Phi_s+self.gamma1),Phi)
        x1 = x-b
        x1 = shift_back_3d(x1)
        x1 = self.lift_m2l(x1)
        theta = self.unet1(x1)
        theta = shift_3d(theta)
        theta = self.drop_l2m(theta)
        b = b- (x-theta)
        x_list.append(theta)
        yb = A(theta+b,Phi)
        x = theta+b + At(torch.div(y-yb,Phi_s+self.gamma2),Phi)
        x1 = x-b
        x1 = shift_back_3d(x1)
        x1 = self.lift_m2l(x1)
        theta = self.unet2(x1)
        theta = shift_3d(theta)
        theta = self.drop_l2m(theta)
        b = b- (x-theta)
        x_list.append(theta)
        yb = A(theta+b,Phi)
        x = theta+b + At(torch.div(y-yb,Phi_s+self.gamma3),Phi)
        x1 = x-b
        x1 = shift_back_3d(x1)
        x1 = self.lift_m2l(x1)
        theta = self.unet3(x1)
        theta = shift_3d(theta)
        theta = self.drop_l2m(theta)
        b = b- (x-theta)
        x_list.append(theta)
        ### 4-6
        yb = A(theta+b,Phi)
        x = theta+b + At(torch.div(y-yb,Phi_s+self.gamma4),Phi)
        x1 = x-b
        x1 = shift_back_3d(x1)
        x1 = self.lift_m2l(x1)
        theta = self.unet4(x1)
        theta = shift_3d(theta)
        theta = self.drop_l2m(theta)
        b = b- (x-theta)
        x_list.append(theta)
        yb = A(theta+b,Phi)
        x = theta+b + At(torch.div(y-yb,Phi_s+self.gamma5),Phi)
        x1 = x-b
        x1 = shift_back_3d(x1)
        x1 = self.lift_m2l(x1)
        theta = self.unet5(x1)
        theta = shift_3d(theta)
        theta = self.drop_l2m(theta)
        b = b- (x-theta)
        x_list.append(theta)
        yb = A(theta+b,Phi)
        x = theta+b + At(torch.div(y-yb,Phi_s+self.gamma6),Phi)
        x1 = x-b
        x1 = shift_back_3d(x1)
        x1 = self.lift_m2l(x1)
        theta = self.unet6(x1)
        theta = shift_3d(theta)
        theta = self.drop_l2m(theta)
        b = b- (x-theta)
        x_list.append(theta)
        ### 7-9
        yb = A(theta+b,Phi)
        x = theta+b + At(torch.div(y-yb,Phi_s+self.gamma7),Phi)
        x1 = x-b
        x1 = shift_back_3d(x1)
        x1 = self.lift_m2l(x1)
        theta = self.unet7(x1)
        theta = shift_3d(theta)
        theta = self.drop_l2m(theta)
        b = b- (x-theta)
        x_list.append(theta)
        yb = A(theta+b,Phi)
        x = theta+b + At(torch.div(y-yb,Phi_s+self.gamma8),Phi)
        x1 = x-b
        x1 = shift_back_3d(x1)
        x1 = self.lift_m2l(x1)
        theta = self.unet8(x1)
        theta = shift_3d(theta)
        theta = self.drop_l2m(theta)
        b = b- (x-theta)
        x_list.append(theta)
        yb = A(theta+b,Phi)
        x = theta+b + At(torch.div(y-yb,Phi_s+self.gamma9),Phi)
        x1 = x-b
        x1 = shift_back_3d(x1)
        x1 = self.lift_m2l(x1)
        theta = self.unet9(x1)
        theta = shift_3d(theta)
        out = self.head_l2out(theta)

        # theta = shift_3d(theta) * scale
        return out[:, :, :H_inp, :W_inp]
