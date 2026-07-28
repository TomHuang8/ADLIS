import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import os
from architecture import *
from test_option import opt
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

A1 = 1
wavelengths = torch.linspace(450, 750, opt.num_wavelengths)

num_pixels = opt.num_pixels
num_wavelengths = opt.num_wavelengths
n_o = opt.n_o
n_e = opt.n_e
alpha = torch.tensor(opt.alpha)
thita = torch.tensor(opt.thita)
thita2 = torch.tensor(opt.thita2)
if alpha > torch.pi / 4 and alpha < 5 * torch.pi / 4:
    alpha = alpha + torch.pi


def phase_delay(d, lambda_, n_o, n_e):
    delta_n = n_e - n_o
    return (2 * torch.pi * d * 1e-6 * delta_n) / (lambda_ * 1e-9)


def intensity(d):
    I = torch.zeros_like(wavelengths)
    for i, lambda_ in enumerate(wavelengths):
        delta_fi = phase_delay(d, lambda_, n_o, n_e)
        I[i] = A1 * A1 * ((torch.cos(alpha - thita)) ** 2 - torch.sin(2 * thita) * torch.sin(2 * alpha) * (torch.sin(delta_fi / 2)) ** 2)
    return I


def intensity_2(d):
    I = torch.zeros_like(wavelengths)
    for i, lambda_ in enumerate(wavelengths):
        delta_fi = phase_delay(d, lambda_, n_o, n_e)
        I[i] = A1 * A1 * ((torch.cos(alpha - thita2)) ** 2 - torch.sin(2 * thita2) * torch.sin(2 * alpha) * (torch.sin(delta_fi / 2)) ** 2)
    return I


class FunModule(nn.Module):
    def __init__(self, is_main=True):
        super(FunModule, self).__init__()
        self.is_main = is_main
        d_values = opt.optics_d
        d_tensor = torch.round(torch.tensor(d_values, dtype=torch.float32))
        self.d = nn.Parameter(d_tensor)

    def process_images(self, images, d):
        device = d.device
        center_wavelengths = [480, 550, 680]
        std_dev = 80

        intensity_values = torch.zeros(num_pixels, num_wavelengths, device=device)
        intensity_values_2 = intensity_values.clone()
        for i in range(num_pixels):
            intensity_values[i, :] = intensity(d[i]).to(device)
            intensity_values_2[i, :] = intensity_2(d[i]).to(device)

        wavelengths_t = torch.linspace(450, 750, opt.num_wavelengths, device=device)
        R_response = torch.exp(-((wavelengths_t - center_wavelengths[0]) ** 2) / (2 * std_dev ** 2))
        G_response = torch.exp(-((wavelengths_t - center_wavelengths[1]) ** 2) / (2 * std_dev ** 2))
        B_response = torch.exp(-((wavelengths_t - center_wavelengths[2]) ** 2) / (2 * std_dev ** 2))

        R_response = R_response.float()
        G_response = G_response.float()
        B_response = B_response.float()

        intensity_values = intensity_values.view(num_pixels, num_wavelengths, 1, 1)
        intensity_values_2 = intensity_values_2.view(num_pixels, num_wavelengths, 1, 1)

        if self.is_main:
            images = images.view(opt.batch_size, num_pixels, num_wavelengths, opt.patch_size, opt.patch_size)
        else:
            images = images.view(-1, num_pixels, num_wavelengths, images.shape[2], images.shape[3])

        images = images * intensity_values
        images_2 = images * intensity_values_2

        images = images.to(device)
        images_2 = images_2.to(device)
        R_response = R_response.to(device)
        G_response = G_response.to(device)
        B_response = B_response.to(device)

        R_response = R_response.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        G_response = G_response.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        B_response = B_response.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        img_r = (images * R_response).sum(dim=2)
        img_g = (images * G_response).sum(dim=2)
        img_b = (images * B_response).sum(dim=2)

        img_r_2 = (images_2 * R_response).sum(dim=2)
        img_g_2 = (images_2 * G_response).sum(dim=2)
        img_b_2 = (images_2 * B_response).sum(dim=2)

        img = torch.stack([img_r, img_g, img_b], dim=2)
        img = torch.sum(img, dim=1)
        img = img / img.max().item()

        img_2 = torch.stack([img_r_2, img_g_2, img_b_2], dim=2)
        img_2 = torch.sum(img_2, dim=1)
        img_2 = img_2 / img_2.max().item()

        if opt.frame == 2:
            IMG = torch.cat((img, img_2), dim=1)
        else:
            IMG = img

        return IMG, intensity_values, intensity_values_2

    def forward(self, images):
        return self.process_images(images, self.d)
