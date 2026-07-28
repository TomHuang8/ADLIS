import argparse
import os
import torch
import os
from datetime import datetime

def save_model_params(model, path, filename):
    params = model.state_dict()
    filepath = os.path.join(path, filename)
    with open(filepath, 'w') as f:
        for k, v in params.items():
            f.write(f'{k} {v}\n')


def save_options(opt, path, filename):
    filepath = os.path.join(path, filename)
    with open(filepath, 'w') as f:
        for k, v in vars(opt).items():
            f.write(f'{k}: {v}\n')


parser = argparse.ArgumentParser(description="Spectral Recovery Toolbox")
parser.add_argument('--method', type=str, default='restormer')
# parser.add_argument('--pretrained_model_path', type=str, default='./model_zoo/mst_plus_plus.pth')
parser.add_argument('--pretrained_model_path', type=str, default='../../train_record/dataset_real_9_25/train/restormer/frame/2_2025_05_16_06_37_50/net_90epoch.pth')
parser.add_argument("--batch_size", type=int, default=1, help="batch size")  # 测试时设为1
parser.add_argument("--end_epoch", type=int, default=300, help="number of epochs")
parser.add_argument("--init_lr", type=float, default=1e-4, help="initial learning rate")
parser.add_argument("--outf", type=str, default='../../exp_9_200_200_36/test/restormer/Test0', help='path log files')
# parser.add_argument("--data_root", type=str, default='../../MST-plus-plus-master/dataset/')
parser.add_argument("--root", type=str, default="../../dataset/dataset/")
parser.add_argument("--data_name", type=str, default="dataset_real_9_25/", help='dataset_qiao_crop_9, ')
parser.add_argument("--patch_size", type=int, default=100, help="patch size")
parser.add_argument("--stride", type=int, default=50, help="stride")
parser.add_argument("--gpu_id", type=str, default='7', help='path log files')
parser.add_argument("--alpha", type=float, default=7.1 * torch.pi / 4, help='光轴夹角')
parser.add_argument("--thita", type=float, default=1 * torch.pi / 4, help='第一帧检偏夹角')
parser.add_argument("--thita2", type=float, default=3 * torch.pi / 4, help='第二帧检偏夹角')
parser.add_argument("--frame", type=int, default=1, help='测量值帧数')
parser.add_argument("--num_pixels", type=int, default=9, help='number of views')
parser.add_argument("--num_wavelengths", type=int, default=36, help='number of wavelengths')
parser.add_argument("--n_o", type=float, default=1.5440, help='o光折射率')
parser.add_argument("--n_e", type=float, default=1.5519, help='e光折射率')
parser.add_argument("--sizes", type=int, default=58, help='gt尺寸')
parser.add_argument("--aperature_mode", type=str, default='REAL', help='CCFA, DO')
parser.add_argument("--optics_d", type=int, default=[647.8,659.1,682.3,583.1,679.2,625.6,898.0,839.0,608.6,], help='d_values')


opt = parser.parse_args()
if opt.aperature_mode == 'CCFA' :
    opt.frame = 1

elif opt.aperature_mode == 'DO' :
    opt.frame = opt.frame

elif opt.aperature_mode == 'REAL' :
    opt.frame = opt.frame

opt.data_root = os.path.join(opt.root, opt.data_name)
last_folder = os.path.basename(os.path.normpath(opt.data_root))
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
opt.outf = os.path.join('../../test_record/', last_folder, 'test/', opt.method, current_time)
if opt.data_name == 'dataset_qiao_crop/' :
    opt.num_pixels = 25
    opt.num_wavelengths = 36
    opt.sizes = 400
elif opt.data_name == 'dataset_qiao_crop_9/' :
    opt.num_pixels = 9
    opt.num_wavelengths = 36
    opt.sizes = 400
elif opt.data_name == 'dataset_qiao_crop_plus/' :
    opt.num_pixels = 25
    opt.num_wavelengths = 36
    opt.sizes = 200
elif opt.data_name == 'dataset_qiao_crop_9_plus/' :
    opt.num_pixels = 9
    opt.num_wavelengths = 36
    opt.sizes = 200
elif opt.data_name == 'dataset_lv/' :
    opt.num_pixels = 9
    opt.num_wavelengths = 31
    opt.sizes = 58
elif opt.data_name == 'dataset_lv_25/' :
    opt.num_pixels = 25
    opt.num_wavelengths = 31
    opt.sizes = 58
elif opt.data_name == 'dataset_lv_49/' :
    opt.num_pixels = 49
    opt.num_wavelengths = 31
    opt.sizes = 58
elif opt.data_name == 'dataset_real_9_25/' :
    opt.num_pixels = 9
    opt.num_wavelengths = 25
    opt.sizes = 400
elif opt.data_name == 'dataset_lv_9_plus/' or 'dataset_9_31_200/' or 'dataset_9_31_200_re/':
    opt.num_pixels = 9
    opt.num_wavelengths = 31
    opt.sizes = 200

