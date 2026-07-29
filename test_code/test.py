import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os
import torch.backends.cudnn as cudnn
from architecture import *
from utils import AverageMeter, save_matv73, Loss_MRAE, Loss_RMSE, Loss_PSNR, SSIM, Loss_SAM
from hsi_dataset import ValidDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import datetime
from test_option import opt, save_options, save_model_params
from test_forward_process import FunModule


os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

val_data = ValidDataset(data_root=opt.data_root)
val_loader = DataLoader(dataset=val_data, batch_size=opt.batch_size, shuffle=False, num_workers=0, pin_memory=True)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

num_pixels = opt.num_pixels
num_wavelengths = opt.num_wavelengths

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

with open(f'{opt.data_root}/split_txt/valid_list.txt', 'r') as fin:
    hyper_list = [line.replace('\n', '.mat') for line in fin]
hyper_list.sort()
var_name = 'cube'


def validate(val_loader, model):
    fun_module.is_main = False
    model.eval()
    fun_module.eval()
    losses_mrae = AverageMeter()
    losses_rmse = AverageMeter()
    losses_psnr = AverageMeter()
    losses_ssim = AverageMeter()
    losses_sam = AverageMeter()
    for i, (input, target) in enumerate(val_loader):
        gt = input.cuda()
        target = target.cuda()
        with torch.no_grad():
            if opt.aperture_mode == 'DO':
                input_meas = fun_module(gt)
            output = model(input_meas)
            loss_mrae = criterion_mrae(output, target)
            loss_rmse = criterion_rmse(output, target)
            loss_psnr = criterion_psnr(output, target)
            loss_ssim = criterion_ssim(output, target)
            loss_sam = criterion_sam(output, target)
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
        mat_name = f"{os.path.splitext(hyper_list[i])[0]}_psnr_{current_psnr:.4f}_ssim_{current_ssim:.4f}.mat"
        mat_dir = os.path.join(opt.outf, mat_name)
        save_matv73(mat_dir, var_name, result)

    return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg, losses_ssim.avg, losses_sam.avg


if __name__ == '__main__':
    cudnn.benchmark = True
    fun_module = FunModule(is_main=False).cuda()
    pretrained_model_path = opt.pretrained_model_path
    method = opt.method
    model = model_generator(method, pretrained_model_path).cuda()

    mrae, rmse, psnr, ssim, sam = validate(val_loader, model, fun_module)

    save_model_params(fun_module, opt.outf, 'fun_module_params.txt')
    save_model_params(model, opt.outf, 'model_params.txt')
    save_options(opt, opt.outf, 'test_options.txt')

    print(f'method:{method}, mrae:{mrae:.6f}, rmse:{rmse:.6f}, psnr:{psnr:.6f}, ssim:{ssim:.6f}, sam:{sam:.6f}')
