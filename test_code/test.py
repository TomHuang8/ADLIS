import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os
import torch.backends.cudnn as cudnn
from architecture import *
from utils import AverageMeter, save_matv73, Loss_MRAE, Loss_RMSE, Loss_PSNR
from hsi_dataset import TrainDataset, ValidDataset
from torch.utils.data import DataLoader
from utils import AverageMeter, save_matv73, Loss_MRAE, Loss_RMSE, Loss_PSNR, SSIM, Loss_SAM
import matplotlib.pyplot as plt
import datetime
from test_option import opt, save_options, save_model_params
from test_forward_process import FunModule
from ccfa_forward_process import CCFAModule
from real_forward_process import REALModule

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

# load dataset
val_data = ValidDataset(data_root=opt.data_root, bgr2rgb=True)
val_loader = DataLoader(dataset=val_data, batch_size=opt.batch_size, shuffle=False, num_workers=0, pin_memory=True)


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# 参数设置
A1 = 1
wavelengths = torch.linspace(450, 750, opt.num_wavelengths)  # Visible spectrum in nm
base_depth = 500  # um
max_depth = 1000  # um

num_pixels = opt.num_pixels
num_wavelengths = opt.num_wavelengths
n_o = opt.n_o  # Ordinary refractive index of quartz
n_e = opt.n_e  # Extraordinary refractive index of quartz
alpha = torch.tensor(opt.alpha)
thita = torch.tensor(opt.thita)
thita2 = torch.tensor(opt.thita2)
if alpha > torch.pi / 4 and alpha < 5 * torch.pi / 4:
    alpha = alpha + torch.pi

# loss function
criterion_mrae = Loss_MRAE()
criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()
criterion_sam = Loss_SAM()
criterion_ssim = SSIM()
if torch.cuda.is_available():
    criterion_mrae.cuda()
    criterion_rmse.cuda()
    criterion_psnr.cuda()
    criterion_ssim.cuda()
    criterion_sam.cuda()
# Validate
with open(f'{opt.data_root}/split_txt/valid_list.txt', 'r') as fin:
    hyper_list = [line.replace('\n', '.mat') for line in fin]
hyper_list.sort()
var_name = 'cube'
def validate(val_loader, model, fun_module, ccfa_module, real_module):
    fun_module.is_main = False
    ccfa_module.is_main = False
    real_module.is_main = False
    model.eval()
    fun_module.eval()
    ccfa_module.eval()
    real_module.eval()
    losses_mrae = AverageMeter()
    losses_rmse = AverageMeter()
    losses_psnr = AverageMeter()
    losses_ssim = AverageMeter()
    losses_sam = AverageMeter()
    for i, (input, target) in enumerate(val_loader):
        gt = input.cuda()
        target = target.cuda()
        with torch.no_grad():
        # if opt.data_root =='../../SLF-recon/dataset_hu/' or '../../SLF-recon/dataset_lv/' or '../../SLF-recon/dataset_lv_25/' or '../../SLF-recon/dataset_qiao_crop_9_plus/'  or '../../SLF-recon/dataset_9_31_200/' or '../../SLF-recon/dataset_qiao_crop_plus/':
            if opt.aperature_mode == 'DO' :
                input_meas, intensity_values, intensity_values_2 = fun_module(gt)
            elif opt.aperature_mode == 'CCFA' :
                input_meas = ccfa_module(gt)
            elif opt.aperature_mode == 'REAL' :
                input_meas = real_module(gt)
            output = model(input_meas)
            meas = input_meas[0, 0:3, :, :]
            meas = meas.cpu().detach().numpy()
            meas = np.transpose(meas, (1, 2, 0))
            plt.imshow(meas)
            plt.show()
            loss_mrae = criterion_mrae(output[:, :, :, :], target[:, :, :, :])
            loss_rmse = criterion_rmse(output[:, :, :, :], target[:, :, :, :])
            loss_psnr = criterion_psnr(output[:, :, :, :], target[:, :, :, :])
            loss_ssim = criterion_ssim(output[:, :, :, :], target[:, :, :, :])
            loss_sam = criterion_sam(output[:, :, :, :], target[:, :, :, :])
            # if method=='awan':   # To avoid out of memory, we crop the center region as input for AWAN.
            #     output = model(input[:, :, 118:-118, 118:-118])
            #     loss_mrae = criterion_mrae(output[:, :, 10:-10, 10:-10], target[:, :, 128:-128, 128:-128])
            #     loss_rmse = criterion_rmse(output[:, :, 10:-10, 10:-10], target[:, :, 128:-128, 128:-128])
            #     loss_psnr = criterion_psnr(output[:, :, 10:-10, 10:-10], target[:, :, 128:-128, 128:-128])

        # record loss
        losses_mrae.update(loss_mrae.data)
        losses_rmse.update(loss_rmse.data)
        losses_psnr.update(loss_psnr.data)
        losses_ssim.update(loss_ssim.data)
        losses_sam.update(loss_sam.data)

        result = output.cpu().numpy() * 1.0
        result = np.transpose(np.squeeze(result), [1, 2, 0])
        result = np.minimum(result, 1.0)
        result = np.maximum(result, 0)
        current_psnr = loss_psnr.item()
        current_ssim = loss_ssim.item()
        current_sam = loss_sam.item()
        # 修改文件名以包含 PSNR 和 SSIM 值
        mat_name = f"{os.path.splitext(hyper_list[i])[0]}_psnr_{current_psnr:.4f}_ssim_{current_ssim:.4f}.mat"
        mat_dir = os.path.join(opt.outf, mat_name)
        save_matv73(mat_dir, var_name, result)
    # line_1 = intensity_values[:, :, 0, 0]
    # line_2 = intensity_values_2[:, :, 0, 0]
    # line_1 = line_1.cpu().detach().numpy()
    # line_2 = line_2.cpu().detach().numpy()
    # ch = np.linspace(450, 750, opt.num_wavelengths)
    # fig, axs = plt.subplots(int(np.sqrt(opt.num_pixels)), int(np.sqrt(opt.num_pixels)), figsize=(15, 15))
    # colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # 颜色列表

    # for i in range(opt.num_pixels):
    #     row = i // int(np.sqrt(opt.num_pixels))
    #     col = i % int(np.sqrt(opt.num_pixels))
    #     axs[row, col].plot(ch, line_1[i], color='b', label='Frame 1')  # 绘制line_1，使用蓝色
    #     axs[row, col].plot(ch, line_2[i], color='r', linestyle='--', label='Frame 2')  # 绘制line_2，使用红色虚线
    #     axs[row, col].set_title(f'Line {i + 1}', fontsize=24)
    #     axs[row, col].set_xlabel('X', fontsize=24)
    #     axs[row, col].set_ylabel('Y', fontsize=24)
    #     axs[row, col].tick_params(axis='both', labelsize=20)
    #     axs[row, col].legend()  # 添加图例
    #
    # plt.tight_layout()
    # plt.show()
    return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg, losses_ssim.avg, losses_sam.avg


if __name__ == '__main__':
    cudnn.benchmark = True
    fun_module = FunModule(is_main=False).cuda()
    ccfa_module = CCFAModule(is_main=False).cuda()
    real_module = REALModule(is_main=False).cuda()
    pretrained_model_path = opt.pretrained_model_path
    method = opt.method
    model = model_generator(method, pretrained_model_path).cuda()
    mrae, rmse, psnr, ssim, sam = validate(val_loader, model, fun_module, ccfa_module, real_module)
    save_model_params(fun_module, opt.outf, 'fun_module_params.txt')
    save_model_params(model, opt.outf, 'model_params.txt')
    save_options(opt, opt.outf, 'test_options.txt')
    print(f'method:{method}, mrae:{mrae}, rmse:{rmse}, psnr:{psnr}, ssim:{ssim}, sam:{sam}')