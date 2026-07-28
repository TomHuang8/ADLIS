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
parser.add_argument('--pretrained_model_path', type=str, default=None)
parser.add_argument("--batch_size", type=int, default=8, help="batch size")
parser.add_argument("--end_epoch", type=int, default=3000, help="number of epochs")
parser.add_argument("--DO_start_iter", type=int, default=300, help="start of do-iter")
parser.add_argument("--DO_end_iter", type=int, default=50000, help="end of do-iter")
parser.add_argument("--init_lr", type=float, default=0.0001, help="initial recon-learning rate")
parser.add_argument("--do_lr", type=float, default=0.002, help="initial do-learning rate")
parser.add_argument("--k_loss_e", type=float, default=0.001, help="loss weight for edge loss")
parser.add_argument("--k_loss_d", type=float, default=10, help="loss weight for d loss")
parser.add_argument("--k_loss_g", type=float, default=0, help="loss weight for gradient loss")
parser.add_argument("--outf", type=str, default='../../exp_400_400_36/train/unet_2f/Test0', help='path log files')
parser.add_argument("--root", type=str, default="../../slf_dataset/")
parser.add_argument("--data_name", type=str, default="dataset_9_36/", help='dataset name')
parser.add_argument("--patch_size", type=int, default=100, help="patch size")
parser.add_argument("--stride", type=int, default=4, help="stride")
parser.add_argument("--gpu_id", type=str, default='1', help='gpu id')
parser.add_argument("--alpha", type=float, default=7.1 * torch.pi / 4, help='angle between optical axes')
parser.add_argument("--frame", type=int, default=1, help='number of measurement frames')
parser.add_argument("--thita", type=float, default=1 * torch.pi / 4, help='first frame analyzer angle')
parser.add_argument("--thita2", type=float, default=3 * torch.pi / 4, help='second frame analyzer angle')
parser.add_argument("--thita3", type=float, default=0.6 * torch.pi, help='third frame analyzer angle')
parser.add_argument("--num_pixels", type=int, default=0, help='number of views')
parser.add_argument("--num_wavelengths", type=int, default=0, help='number of wavelengths')
parser.add_argument("--n_o", type=float, default=1.5440, help='ordinary refractive index')
parser.add_argument("--n_e", type=float, default=1.5519, help='extraordinary refractive index')
parser.add_argument("--sizes", type=int, default=0, help='ground truth size')
parser.add_argument("--seed", type=int, default=1, help='random seed')
parser.add_argument("--canny_mode", type=str, default='REGION', help='canny mode: REGION or EDGE')
parser.add_argument("--aperture_mode", type=str, default='DO', help='aperture mode: CCFA, DO, REAL')
parser.add_argument("--canny_size", type=int, default=15, help='canny kernel size')
parser.add_argument("--canny_threshold", type=int, default=[5, 15], help='canny thresholds')
parser.add_argument("--salt_noise", type=float, default=0.0, help='salt and pepper noise intensity')
parser.add_argument("--gaussian_std", type=float, default=0.01, help='gaussian noise standard deviation')
parser.add_argument("--optics_d", type=int, default=[644.0, 671.0, 692.0, 587.0, 683.0, 645.0, 905.0, 854.0, 597.0], help='initial thickness values')
parser.add_argument("--norm", type=int, default=1, help='normalization factor')

parser.add_argument("--admm_unet_feats", type=int, default=32, help='admm unet features')
parser.add_argument("--cstfeats", type=int, default=48, help='cst features')
parser.add_argument("--mirnetfeats", type=int, default=48, help='mirnet features')
parser.add_argument("--unetfeats", type=int, default=8, help='unet features')
parser.add_argument("--hrnetchannels", type=int, default=24, help='hrnet channels')
parser.add_argument("--hdnetfeats", type=int, default=24, help='hdnet features')


opt = parser.parse_args()

if opt.aperture_mode == 'CCFA':
    opt.frame = 1
elif opt.aperture_mode == 'DO':
    opt.frame = opt.frame


opt.data_root = os.path.join(opt.root, opt.data_name)
last_folder = os.path.basename(os.path.normpath(opt.data_root))
opt.outf = os.path.join('../../train_record/', last_folder, 'train/', opt.method, str(opt.frame), str(opt.aperture_mode))

if opt.data_name == 'dataset_9_36/':
    opt.num_pixels = 9
    opt.num_wavelengths = 36
    opt.sizes = 400
