import torch
from train_option import opt
from .edsr import EDSR
from .hinet import HINet
from .HSCNN_Plus import HSCNN_Plus
from .MPRNet import MPRNet
from .Restormer import Restormer

def model_generator(method, pretrained_model_path=None):
    if method == 'hinet':
        model = HINet(depth=4).cuda()
    elif method == 'mprnet':
        model = MPRNet(num_cab=4).cuda()
    elif method == 'restormer':
        model = Restormer().cuda()
    elif method == 'edsr':
        model = EDSR().cuda()
    elif method == 'hscnn_plus':
        model = HSCNN_Plus().cuda()

    else:
        print(f'Method {method} is not defined !!!!')
    if pretrained_model_path is not None:
        print(f'load model from {pretrained_model_path}')
        checkpoint = torch.load(pretrained_model_path)
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()},
                              strict=True)
    return model
