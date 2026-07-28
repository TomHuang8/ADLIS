import argparse
import os
import torch


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
parser.add_argument('--pretrained_model_path', type=str, default=None)
parser.add_argument("--batch_size", type=int, default=8, help="batch size")  # 训练时设置为4
parser.add_argument("--end_epoch", type=int, default=3000, help="number of epochs")
parser.add_argument("--DO_start_iter", type=int, default=300, help="start of do-iter")
parser.add_argument("--DO_end_iter", type=int, default=50000, help="end of do-iter")
parser.add_argument("--init_lr", type=float, default=0.0001, help="initial recon-learning rate")
parser.add_argument("--do_lr", type=float, default=0.002, help="initial do-learning rate")
parser.add_argument("--k_loss_e", type=float, default=0.001, help="initial do-learning rate")
parser.add_argument("--k_loss_d", type=float, default=10, help="initial do-learning rate")
parser.add_argument("--k_loss_g", type=float, default=0, help="initial grad rate")
parser.add_argument("--outf", type=str, default='../../exp_400_400_36/train/unet_2f/Test0', help='path log files')
# parser.add_argument("--data_root", type=str, default='../../MST-plus-plus-master/dataset/')
parser.add_argument("--root", type=str, default="../../slf_dataset/")
parser.add_argument("--data_name", type=str, default="dataset_9_36/", help='dataset_9_36, ')
parser.add_argument("--patch_size", type=int, default=100, help="patch size")
parser.add_argument("--stride", type=int, default=4, help="stride")
parser.add_argument("--gpu_id", type=str, default='1', help='path log files')
parser.add_argument("--alpha", type=float, default=7.1 * torch.pi / 4, help='光轴夹角')
parser.add_argument("--frame", type=int, default=1, help='单帧测量值')
parser.add_argument("--thita", type=float, default=1 * torch.pi / 4, help='第一帧检偏夹角')
parser.add_argument("--thita2", type=float, default=3 * torch.pi / 4, help='第二帧检偏夹角')
parser.add_argument("--thita3", type=float, default=0.6 * torch.pi, help='第三帧检偏夹角')
parser.add_argument("--num_pixels", type=int, default=0, help='number of views')
parser.add_argument("--num_wavelengths", type=int, default=0, help='number of wavelengths')
parser.add_argument("--n_o", type=float, default=1.5440, help='o光折射率')
parser.add_argument("--n_e", type=float, default=1.5519, help='e光折射率')
parser.add_argument("--sizes", type=int, default=0, help='gt尺寸')
parser.add_argument("--seed", type=int, default=1, help='random seed')
parser.add_argument("--canny_mode", type=str, default='REGION', help='region')
parser.add_argument("--aperture_mode", type=str, default='DO', help='CCFA, DO, REAL')
parser.add_argument("--canny_size", type=int, default=15 , help='canny kernel size')
parser.add_argument("--canny_threshold", type=int, default=[5,15], help='canny window')
parser.add_argument("--salt_noise", type=float, default=0.0, help='salt_noise')
parser.add_argument("--gaussian_std", type=float, default=0.01, help='salt_noise')
parser.add_argument("--optics_d", type=int, default=[644.0,671.0,692.0,587.0,683.0,645.0,905.0,854.0,597.0], help='')
parser.add_argument("--norm", type=int, default=1, help='canny kernel size')

parser.add_argument("--admm_unet_feats", type=int, default=32, help='admm_unet_feats')
parser.add_argument("--cstfeats", type=int, default=48, help='')
parser.add_argument("--mirnetfeats", type=int, default=48, help='')
parser.add_argument("--unetfeats", type=int, default=8, help='unet_feature')
parser.add_argument("--hrnetchannels", type=int, default=24, help='hrnet_feature')
parser.add_argument("--hdnetfeats", type=int, default=24, help='hdnet_feature')
opt = parser.parse_args()
if opt.aperture_mode == 'CCFA' :
    opt.frame = 1
elif opt.aperture_mode == 'DO' :
    opt.frame = opt.frame
elif opt.aperture_mode == 'REAL' :
    opt.frame = opt.frame

opt.data_root = os.path.join(opt.root, opt.data_name)
last_folder = os.path.basename(os.path.normpath(opt.data_root))
opt.outf = os.path.join('../../train_record/', last_folder, 'train/', opt.method, str(opt.frame), str(opt.aperture_mode))

if opt.data_name == 'dataset_qiao_crop/' :
    opt.num_pixels = 25
    opt.num_wavelengths = 36
    opt.sizes = 400
elif opt.data_name == 'dataset_qiao_crop_9/' :
    opt.num_pixels = 9
    opt.num_wavelengths = 36
    opt.sizes = 400
elif opt.data_name == 'dataset_qiao_crop/' :
    opt.num_pixels = 25
    opt.num_wavelengths = 36
    opt.sizes = 200
    opt.optics_d = [672.9, 784.5, 591.2, 823.7, 654.8, 735.6, 589.1, 892.3, 765.4, 612.7, 748.9, 693.2, 856.5, 578.3, 721.6, 884.7, 639.8,
                    702.5, 567.9, 843.1, 759.2, 685.3, 812.4, 796.8, 924.5],
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
elif opt.data_name == 'dataset_9_25/' :
    opt.num_pixels = 9
    opt.num_wavelengths = 25
    opt.sizes = [400,400]
elif opt.data_name ==  'dataset_sy_9_25/':
    opt.num_pixels = 9
    opt.num_wavelengths = 25
    opt.sizes = [600,800]
elif opt.data_name == 'dataset_9_31/' :
    opt.num_pixels = 9
    opt.num_wavelengths = 31
    opt.sizes = [400,400]
elif opt.data_name == 'dataset_9_36/' :
    opt.num_pixels = 9
    opt.num_wavelengths = 36
    opt.sizes = [400,400]
elif opt.data_name == 'dataset_lv_9_plus/' or 'dataset_9_31_200/' or 'dataset_9_31_200_re/':
    opt.num_pixels = 9
    opt.num_wavelengths = 31
    opt.sizes = 200
    opt.optics_d = [644.0,671.0,692.0,587.0,683.0,645.0,905.0,854.0,597.0]
