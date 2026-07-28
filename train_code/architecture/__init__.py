import torch
from train_option import opt
from .edsr import EDSR
from .HDNet import HDNet
from .hinet import HINet
from .hrnet import SGN
from .HSCNN_Plus import HSCNN_Plus
from .MIRNet import MIRNet
from .MPRNet import MPRNet
from .MST import MST
from .MST_Plus_Plus import MST_Plus_Plus
from .Restormer import Restormer
from .AWAN import AWAN
from .UNet import UNet
from .CNN import CNN
from .UNet import UNetCascade
from .CSST import CSST, Csst
from .CST import CST
from .S2Transformer import S2Transformer
from .ADMM_Net import ADMM_net
from .BiSRNet import BiSRNet

def model_generator(method, pretrained_model_path=None):
    if method == 'mirnet':
        model = MIRNet(n_RRG=3, n_MSRB=1, height=3, width=1).cuda()
    elif method == 'mst_plus_plus':
        model = MST_Plus_Plus().cuda()
    elif method == 'mst':
        model = MST(dim=279, stage=2, num_blocks=[4, 7, 5]).cuda()
    elif method == 'hinet':
        model = HINet(depth=4).cuda()
    elif method == 'mprnet':
        model = MPRNet(num_cab=4).cuda()
    elif method == 'restormer':
        model = Restormer().cuda()
    elif method == 'edsr':
        model = EDSR().cuda()
    elif method == 'hdnet':
        model = HDNet().cuda()
    elif method == 'hrnet':
        model = SGN().cuda()
    elif method == 'hscnn_plus':
        model = HSCNN_Plus().cuda()
    elif method == 'awan':
        model = AWAN().cuda()
    elif method == 'unet':
        model = UNetCascade(opt).cuda()
    elif method == 'cnn':
        model = CNN().cuda()
    elif method == 'csst':
        model = Csst().cuda()
        # model = CSST(num_iterations=5).cuda()
    elif method == 'cst':
        model = CST(num_blocks=[1, 1, 2], sparse=True).cuda()
    elif method == 's2transformer':
        model = S2Transformer().cuda()
    elif method == 'admm':
        model = ADMM_net().cuda()
    elif method == 'bisrnet':
        model = BiSRNet().cuda()
    else:
        print(f'Method {method} is not defined !!!!')
    if pretrained_model_path is not None:
        print(f'load model from {pretrained_model_path}')
        checkpoint = torch.load(pretrained_model_path)
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()},
                              strict=True)
    return model
