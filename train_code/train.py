import os
import random
import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from hsi_dataset import TrainDataset, ValidDataset
from architecture import model_generator
from utils import (
    AverageMeter,
    initialize_logger,
    save_checkpoint,
    record_loss,
    time2file_name,
    Loss_MRAE,
    Loss_RMSE,
    Loss_PSNR,
    SSIM,
)
from train_option import opt, save_model_params, save_options
from train_forward_process import FunModule, process_images, slf_to_lf, process_images_plus
from ccfa_forward_process import CCFAModule

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seed = opt.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

print("\nLoading dataset ...")
train_data = TrainDataset(
    data_root=opt.data_root,
    crop_size=opt.patch_size,
    arg=True,
    stride=opt.stride,
)
print(f"Iterations per epoch: {len(train_data)}")
val_data = ValidDataset(data_root=opt.data_root)
print(f"Validation set samples: {len(val_data)}")

per_epoch_iteration = 1000
total_iteration = per_epoch_iteration * opt.end_epoch

criterion_mrae = Loss_MRAE()
criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()
criterion_ssim = SSIM()

pretrained_model_path = opt.pretrained_model_path
method = opt.method
fun_module = FunModule(is_main=True).cuda()
model = model_generator(method, pretrained_model_path).cuda()
print(f"Parameters number: {sum(param.numel() for param in model.parameters())}")

date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
opt.outf = opt.outf + "_" + date_time
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

save_options(opt, opt.outf, "train_options.txt")

if torch.cuda.is_available():
    model.cuda()
    criterion_mrae.cuda()
    criterion_rmse.cuda()
    criterion_psnr.cuda()
    criterion_ssim.cuda()

optimizer = optim.Adam(model.parameters(), lr=opt.init_lr, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, total_iteration, eta_min=1e-6
)

optimizer_fun = optim.Adam(fun_module.parameters(), lr=opt.do_lr, betas=(0.9, 0.999))
scheduler_fun = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_fun, total_iteration, eta_min=1e-2
)

log_dir = os.path.join(opt.outf, "train.log")
logger = initialize_logger(log_dir)

resume_file = opt.pretrained_model_path
if resume_file is not None and os.path.isfile(resume_file):
    print(f"=> loading checkpoint '{resume_file}'")
    checkpoint = torch.load(resume_file)
    start_epoch = checkpoint["epoch"]
    iteration = checkpoint["iter"]
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

sobel_x = torch.tensor(
    [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
    dtype=torch.float32,
    device=device,
).view(1, 1, 3, 3)

sobel_y = torch.tensor(
    [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
    dtype=torch.float32,
    device=device,
).view(1, 1, 3, 3)


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

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            gt = input.cuda()
            target = target.cuda()

            if opt.aperture_mode == "WD" or opt.aperture_mode == "DO":
                input_meas = fun_module(gt)
            elif opt.aperture_mode == "CCFA":
                input_meas = ccfa_module(gt)
            elif opt.aperture_mode == "REAL":
                input_meas = real_module(gt)

            output = model(input_meas)

            roi = slice(opt.sizes[0] // 2 - 250, opt.sizes[0] // 2 + 250)
            output_roi = output[:, :, roi, roi]
            target_roi = target[:, :, roi, roi]

            loss_mrae = criterion_mrae(output_roi, target_roi)
            loss_rmse = criterion_rmse(output_roi, target_roi)
            loss_psnr = criterion_psnr(output_roi, target_roi)
            loss_ssim = criterion_ssim(output_roi, target_roi)

            losses_mrae.update(loss_mrae.data)
            losses_rmse.update(loss_rmse.data)
            losses_psnr.update(loss_psnr.data)
            losses_ssim.update(loss_ssim.data)

    fun_module.is_main = True
    ccfa_module.is_main = True
    real_module.is_main = True

    return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg, losses_ssim.avg


def main():
    iteration = 0
    record_rmse = 1000

    ccfa_module = CCFAModule(is_main=True).cuda()
    real_module = REALModule(is_main=True).cuda()

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=4,
    )
    val_loader = DataLoader(
        dataset=val_data,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4,
    )

    flag_phase1 = True

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

            if opt.aperture_mode == "WD" or opt.aperture_mode == "DO":
                images_meas = fun_module(images)
            elif opt.aperture_mode == "CCFA":
                images_meas = ccfa_module(images)
            elif opt.aperture_mode == "REAL":
                images_meas = real_module(images)

            output = model(images_meas)

            output_space = slf_to_lf(output).mean(dim=1)
            labels_space = slf_to_lf(labels).mean(dim=1)

            with torch.no_grad():
                edge_mask = process_images_plus(labels_space, opt.batch_size).to(device)

            if opt.canny_mode == "EDGE":
                labels_edges = process_images(labels_space, opt.batch_size).to(device) * labels_space
                output_edges = process_images(labels_space, opt.batch_size).to(device) * output_space
            elif opt.canny_mode == "REGION":
                labels_edges = edge_mask * labels_space
                output_edges = edge_mask * output_space

            edge_mask = edge_mask.unsqueeze(1)

            output_space_ = output_space.unsqueeze(1)
            labels_space_ = labels_space.unsqueeze(1)

            grad_output_x = F.conv2d(output_space_, sobel_x, padding=1)
            grad_output_y = F.conv2d(output_space_, sobel_y, padding=1)
            grad_labels_x = F.conv2d(labels_space_, sobel_x, padding=1)
            grad_labels_y = F.conv2d(labels_space_, sobel_y, padding=1)

            grad_output_x_masked = grad_output_x * edge_mask
            grad_output_y_masked = grad_output_y * edge_mask
            grad_labels_x_masked = grad_labels_x * edge_mask
            grad_labels_y_masked = grad_labels_y * edge_mask

            loss_g_x = criterion_rmse(grad_output_x_masked, grad_labels_x_masked)
            loss_g_y = criterion_rmse(grad_output_y_masked, grad_labels_y_masked)
            loss_g = loss_g_x + loss_g_y

            loss_e = criterion_rmse(output_edges, labels_edges)
            loss = criterion_rmse(output, labels)
            loss_d = criterion_rmse(output, labels)

            if flag_phase1:
                total_loss = loss + opt.k_loss_d * loss_d
            else:
                total_loss = loss + opt.k_loss_d * loss_d + opt.k_loss_e * loss_e

            total_loss.backward()
            optimizer.step()
            scheduler.step()

            if (
                iteration >= opt.DO_start_iter
                and iteration <= opt.DO_end_iter
                and opt.aperture_mode == "DO"
            ):
                optimizer_fun.step()
                scheduler_fun.step()

            losses.update(total_loss.data)
            losses_edge.update(loss_e.data)
            losses_d.update(loss_d.data)
            losses_grad.update(loss_g.data)

            lr = optimizer.param_groups[0]["lr"]
            lr_fun = optimizer_fun.param_groups[0]["lr"]

            if iteration % 20 == 0:
                print(
                    f"[iter:{iteration}/{total_iteration}], lr={lr:.9f}, "
                    f"lr_fun={lr_fun:.9f}, train_losses.avg={losses.avg:.9f}, "
                    f"train_losses.d={losses_d.avg:.9f}, train_losses.edge={losses_edge.avg:.9f}, "
                    f"train_losses.grad={losses_grad.avg:.9f}"
                )

            if iteration % 100 == 0:
                mrae_loss, rmse_loss, psnr_loss, ssim_loss = validate(
                    val_loader, model, fun_module, ccfa_module, real_module
                )
                print(f"RMSE: {rmse_loss}, PSNR: {psnr_loss}, SSIM: {ssim_loss}")

                if torch.abs(rmse_loss - record_rmse) < 0.01 or rmse_loss < record_rmse or iteration % 5000 == 0:
                    print(f"Saving to {opt.outf}")
                    save_checkpoint(opt.outf, iteration // 1000, iteration, model, optimizer)

                if rmse_loss < record_rmse:
                    record_rmse = rmse_loss

                log_msg = (
                    f"Iter[{iteration:06d}], Epoch[{iteration // 1000:06d}], "
                    f"lr={lr:.9f}, lr_fun={lr_fun:.9f}, Train Loss={losses.avg:.9f}, "
                    f"Test MRAE={mrae_loss:.9f}, Test RMSE={rmse_loss:.9f}, "
                    f"Test PSNR={psnr_loss:.9f}, Test SSIM={ssim_loss:.9f}"
                )
                print(log_msg)
                logger.info(log_msg)

                d_msg = ""
                for j in range(opt.num_pixels):
                    d_msg += f"{fun_module.d[j].item():.1f},"
                    if (j + 1) % opt.num_pixels == 0:
                        print(d_msg)
                        logger.info(d_msg)
                        d_msg = ""

            if iteration == 150:
                flag_phase1 = False

            iteration += 1

    save_model_params(fun_module, opt.outf, "fun_module_params.txt")
    save_model_params(model, opt.outf, "model_params.txt")
    save_options(opt, opt.outf, "train_options.txt")

    return 0


if __name__ == "__main__":
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main()
