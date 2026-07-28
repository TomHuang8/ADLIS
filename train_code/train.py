import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from hsi_dataset import TrainDataset, ValidDataset
from architecture import *
from utils import AverageMeter, initialize_logger, save_checkpoint, record_loss, \
    time2file_name, Loss_MRAE, Loss_RMSE, Loss_PSNR, SSIM
import matplotlib.pyplot as plt
import datetime
from train_forward_process import phase_delay, intensity, intensity_2, FunModule, process_images, slf_to_lf, process_images_plus
import numpy as np
from ccfa_forward_process import CCFAModule, generate_spectral_filters
from real_forward_process import REALModule
from train_option import opt, save_model_params, save_options
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import MSELoss
import torch.nn.functional as F
from collections import defaultdict
from thop import profile  # 需安装thop: pip install thop
opt = opt
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

seed = opt.seed  # fixed as 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
k = int(np.sqrt(opt.num_pixels))
num_wavelengths = opt.num_wavelengths

# load dataset
print("\nloading dataset ...")
train_data = TrainDataset(data_root=opt.data_root, crop_size=opt.patch_size, arg=True, stride=opt.stride, device='cuda')
print(f"Iteration per epoch: {len(train_data)}")
val_data = ValidDataset(data_root=opt.data_root)
print("Validation set samples: ", len(val_data))

# iterations
per_epoch_iteration = 1000
total_iteration = per_epoch_iteration*opt.end_epoch

# loss function
criterion_mrae = Loss_MRAE()
criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()
criterion_ssim = SSIM()

# model
pretrained_model_path = opt.pretrained_model_path
method = opt.method
fun_module = FunModule(is_main=True).cuda()
model = model_generator(method, pretrained_model_path).cuda()
print('Parameters number is ', sum(param.numel() for param in model.parameters()))
# # 计算GFLOPs（需要指定输入尺寸，这里假设输入为3通道224x224图像）
# # 根据你的实际输入尺寸修改下面的shape
# input_tensor = torch.randn(4, 3*opt.frame, 100, 100).cuda()
# # 统计计算量
# flops, params_thop = profile(model, inputs=(input_tensor,))
#
# # 转换为GFLOPs（1 GFLOPs = 10^9 FLOPs）
# gflops = flops / 1e9
# print(f'GFLOPs: {gflops:.2f}')

# output path
date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
opt.outf = opt.outf + '_' + date_time
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

save_options(opt, opt.outf, 'train_options.txt')

if torch.cuda.is_available():
    model.cuda()
    criterion_mrae.cuda()
    criterion_rmse.cuda()
    criterion_psnr.cuda()
    criterion_ssim.cuda()

optimizer = optim.Adam(model.parameters(), lr=opt.init_lr, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iteration, eta_min=1e-6)
optimizer_fun =  optim.Adam(fun_module.parameters(), lr=opt.do_lr, betas=(0.9, 0.999))
scheduler_fun = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_fun, total_iteration, eta_min=1e-2)
optimizer_fun_all =  optim.Adam(fun_module.parameters(), lr=opt.do_lr, betas=(0.9, 0.999))
scheduler_fun_all = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_fun, total_iteration, eta_min=1e-2)
# logging
log_dir = os.path.join(opt.outf, 'train.log')
logger = initialize_logger(log_dir)
file_path = os.path.join(opt.outf, 'd_value.txt')
# Resume
resume_file = opt.pretrained_model_path
if resume_file is not None:
    if os.path.isfile(resume_file):
        print("=> loading checkpoint '{}'".format(resume_file))
        checkpoint = torch.load(resume_file)
        start_epoch = checkpoint['epoch']
        iteration = checkpoint['iter']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

def main():

    # cudnn.benchmark = True
    # 在训练循环前定义Sobel滤波器
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                           dtype=torch.float32, device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                           dtype=torch.float32, device=device).view(1, 1, 3, 3)
    iteration = 0
    record_mrae_loss = 1000
    fun_module = FunModule(is_main=True).cuda()
    ccfa_module = CCFAModule(is_main=True).cuda()
    real_module = REALModule(is_main=True).cuda()
    optimizer_fun = optim.Adam(fun_module.parameters(), lr=opt.do_lr, betas=(0.9, 0.999))
    criterion_fun = MSELoss()
    d_values = [[] for _ in range(opt.num_pixels)]
    psnr_values = []
    loss_values = []
    ssim_values = []
    train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True, num_workers=8,pin_memory=True, drop_last=True, prefetch_factor=4)
    val_loader = DataLoader(dataset=val_data, batch_size=opt.batch_size, shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=4)
    lr = optimizer.param_groups[0]['lr']
    lr_fun = optimizer_fun.param_groups[0]['lr']
    flag_1 = iteration <= 150
    flag_2 = False
    flag_3 = False
    while iteration < total_iteration:
        model.train()
        losses = AverageMeter()
        losses_edge = AverageMeter()
        losses_d = AverageMeter()
        losses_grad = AverageMeter()
        for i, (images, labels) in enumerate(train_loader):
            labels = labels.cuda()
            images = images.cuda()
            images = Variable(images)
            labels = Variable(labels)
            optimizer.zero_grad()
            if opt.aperture_mode == 'WD':
                images_meas = fun_module(images)
            if opt.aperture_mode == 'DO':
                optimizer_fun.zero_grad()
                images_meas = fun_module(images)
            elif opt.aperture_mode == 'CCFA':
                images_meas = ccfa_module(images)
            elif opt.aperture_mode == 'REAL':
                images_meas = real_module(images)

            output = model(images_meas)
            output_space = slf_to_lf(output).mean(dim=1)
            labels_space = slf_to_lf(labels).mean(dim=1)
            # 预先计算边缘掩码
            with torch.no_grad():
                edge_mask = process_images_plus(labels_space, opt.batch_size).to(device)

            # 对gt应用边缘检测
            if opt.canny_mode == 'EDGE':
                labels_edges = process_images(labels_space, opt.batch_size).to(device) * labels_space
                output_edges = process_images(labels_space, opt.batch_size).to(device) * output_space
            if opt.canny_mode == 'REGION':
                labels_edges = edge_mask * labels_space
                output_edges = edge_mask * output_space
            # 计算边缘点上的RMSE
            edge_mask = edge_mask.unsqueeze(1)  # 适配卷积后的维度 [batch, 1, H, W]
            # 确保output_space和labels_space有通道维度
            output_space_ = output_space.unsqueeze(1)  # [batch, 1, H, W]
            labels_space_ = labels_space.unsqueeze(1)
            # 计算梯度
            grad_output_x = F.conv2d(output_space_, sobel_x, padding=1)
            grad_output_y = F.conv2d(output_space_, sobel_y, padding=1)
            grad_labels_x = F.conv2d(labels_space_, sobel_x, padding=1)
            grad_labels_y = F.conv2d(labels_space_, sobel_y, padding=1)

            # 应用边缘掩码并计算梯度损失
            grad_output_x_masked = grad_output_x * edge_mask
            grad_output_y_masked = grad_output_y * edge_mask
            grad_labels_x_masked = grad_labels_x * edge_mask
            grad_labels_y_masked = grad_labels_y * edge_mask

            loss_g_x = criterion_rmse(grad_output_x_masked, grad_labels_x_masked)
            loss_g_y = criterion_rmse(grad_output_y_masked, grad_labels_y_masked)
            loss_g = loss_g_x + loss_g_y

            loss_e = criterion_rmse(output_edges, labels_edges)
            loss = criterion_rmse(output, labels)
            loss_d = criterion_fun(output, labels)

            if flag_1:
                total_loss = loss + opt.k_loss_d * loss_d
            else:
                total_loss = loss + opt.k_loss_d * loss_d + opt.k_loss_e * loss_e
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            if iteration >= opt.DO_start_iter and iteration <= opt.DO_end_iter and opt.aperture_mode == 'DO':
                if not flag_2:
                    flag_2 = True
                optimizer_fun.step()
                scheduler_fun.step()
            losses.update(total_loss.data)
            losses_edge.update(loss_e.data)
            losses_d.update(loss_d.data)
            losses_grad.update(loss_g.data)

            # 在每次迭代后，将d的值添加到对应的列表中
            for i in range(opt.num_pixels):
                d_values[i].append(fun_module.d[i].item())

            if iteration % 100 == 0:
                mrae_loss, rmse_loss, psnr_loss, ssim_loss = validate(val_loader, model, fun_module, ccfa_module, real_module)
                psnr_values.append(psnr_loss)
                loss_values.append(losses.avg)
                ssim_values.append(ssim_loss)

            if iteration == 150:
                flag_1 = False
            if iteration % 20 == 0:
                print('[iter:%d/%d],lr=%.9f, lr_fun=%.9f, train_losses.avg=%.9f, train_losses.d=%.9f, train_losses.edge=%.9f, train_losses.grad=%.9f'
                      % (iteration, total_iteration, lr, lr_fun, losses.avg, losses_d.avg, losses_edge.avg, losses_grad.avg))
            if iteration % 100 == 0:
                print(f'RMSE: {rmse_loss}, PNSR:{psnr_loss}, SSIM:{ssim_loss}')
                # Save model
                if torch.abs(mrae_loss - record_mrae_loss) < 0.01 or mrae_loss < record_mrae_loss or iteration % 5000 == 0:
                    print(f'Saving to {opt.outf}')
                    save_checkpoint(opt.outf, (iteration // 1000), iteration, model, optimizer)
                if rmse_loss < record_mrae_loss:
                    record_mrae_loss = rmse_loss
                # Prepare the base message
                base_msg = " Iter[%06d], Epoch[%06d], learning rate : %.9f, %.9f, Train MRAE: %.9f, Test MRAE: %.9f, Test RMSE: %.9f, Test PSNR: %.9f, Test SSIM: %.9f,"
                base_msg = base_msg % (
                iteration, iteration // 1000, lr, lr_fun, losses.avg, mrae_loss, rmse_loss, psnr_loss,ssim_loss)
                print(base_msg)
                logger.info(base_msg)
                # Prepare the d values message
                d_values_msg = ""
                for i in range(opt.num_pixels):
                    d_values_msg += "%.1f," % (fun_module.d[i].item())
                    if (i + 1) % opt.num_pixels == 0:  # print and log every 5 d values
                        print(d_values_msg)
                        logger.info(d_values_msg)
                        d_values_msg = ""  # reset the d values message

                # 每150个迭代，绘制一次d的更新情况，并绘制当前的PSNR和损失
            if iteration % 20000 == 0:
                    plt.figure(figsize=(28, 28))
                    plt.subplot(k+1, k, 1)
                    plt.plot(psnr_values)
                    plt.title('DO_MF_PSNR over iters', fontsize=30)
                    plt.tick_params(axis='both', which='major', labelsize=24)

                    # 绘制当前的PSNR和损失
                    plt.subplot(k+1, k, 2)
                    plt.plot(ssim_values)
                    plt.title('DO_MF_SSIM over iters', fontsize=30)
                    plt.tick_params(axis='both', which='major', labelsize=24)

                    plt.subplot(k+1, k, 3)
                    plt.plot(loss_values)
                    plt.title('DO_MF_Loss over iters', fontsize=30)
                    plt.tick_params(axis='both', which='major', labelsize=24)

                    # 绘制d的更新情况
                    for i in range(opt.num_pixels):
                        plt.subplot(k+1, k, i + k+1)  # 创建一个4x3的子图，当前绘制第i+4个子图
                        plt.plot(d_values[i])  # 绘制第i个d的值
                        plt.title(f'd{i + 1}', fontsize=30)  # 设置子图的标题
                        plt.tick_params(axis='both', which='major', labelsize=24)
                    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
                    plt.savefig(os.path.join(opt.outf, f'training_plot_iter_{iteration}.png'))
                    plt.close()
            iteration = iteration + 1
    # 在训练结束后，保存模型参数
    save_model_params(fun_module, opt.outf, 'fun_module_params.txt')
    save_model_params(model, opt.outf, 'model_params.txt')
    save_options(opt, opt.outf, 'train_options.txt')
    return 0

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
    for i, (input, target) in enumerate(val_loader):
        gt= input.cuda()
        target = target.cuda()
        with torch.no_grad():
            # compute output
            if opt.aperture_mode == 'WD':
                input_meas = fun_module(gt)
            if opt.aperture_mode == 'DO':
                input_meas = fun_module(gt)
            elif opt.aperture_mode == 'CCFA':
                input_meas = ccfa_module(gt)
            elif opt.aperture_mode == 'REAL':
                input_meas = real_module(gt)
            output = model(input_meas)

            # 定义感兴趣区域
            roi = slice(opt.sizes[0] // 2 - 250, opt.sizes[0] // 2 + 250)
            output_roi = output[:, :, roi, roi]
            target_roi = target[:, :, roi, roi]

            # 计算各项损失指标
            loss_mrae = criterion_mrae(output_roi, target_roi)
            loss_rmse = criterion_rmse(output_roi, target_roi)
            loss_psnr = criterion_psnr(output_roi, target_roi)
            loss_ssim = criterion_ssim(output_roi, target_roi)
            # record loss
            losses_mrae.update(loss_mrae.data)
            losses_rmse.update(loss_rmse.data)
            losses_psnr.update(loss_psnr.data)
            losses_ssim.update(loss_ssim.data)
    fun_module.is_main = True
    return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg, losses_ssim.avg

def print_gradients(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name} gradients: {param.grad}")

if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main()
    print(torch.__version__)