import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from architecture import *
from train_option import opt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

A1 = 1
wavelengths = torch.linspace(450, 750, opt.num_wavelengths)
base_depth = 500
max_depth = 1000

n_o = opt.n_o
n_e = opt.n_e
alpha = torch.tensor(opt.alpha)
thita = torch.tensor(opt.thita)
thita2 = torch.tensor(opt.thita2)
thita3 = torch.tensor(opt.thita3)

if alpha > torch.pi / 4 and alpha < 5 * torch.pi / 4:
    alpha = alpha + torch.pi

num_pixels = opt.num_pixels
num_wavelengths = opt.num_wavelengths


def phase_delay(d, lambda_, n_o, n_e):
    delta_n = n_e - n_o
    return (2 * torch.pi * d * 1e-6 * delta_n) / (lambda_ * 1e-9)


def intensity(d):
    I = torch.zeros_like(wavelengths)
    for i, lambda_ in enumerate(wavelengths):
        delta_fi = phase_delay(d, lambda_, n_o, n_e)
        I[i] = (
            A1 * A1
            * (
                (torch.cos(alpha - thita)) ** 2
                - torch.sin(2 * thita)
                * torch.sin(2 * alpha)
                * (torch.sin(delta_fi / 2)) ** 2
            )
        )
    return I


def intensity_2(d):
    I = torch.zeros_like(wavelengths)
    for i, lambda_ in enumerate(wavelengths):
        delta_fi = phase_delay(d, lambda_, n_o, n_e)
        I[i] = (
            A1 * A1
            * (
                (torch.cos(alpha - thita2)) ** 2
                - torch.sin(2 * thita2)
                * torch.sin(2 * alpha)
                * (torch.sin(delta_fi / 2)) ** 2
            )
        )
    return I


def intensity_3(d):
    I = torch.zeros_like(wavelengths)
    for i, lambda_ in enumerate(wavelengths):
        delta_fi = phase_delay(d, lambda_, n_o, n_e)
        I[i] = (
            A1 * A1
            * (
                (torch.cos(alpha - thita3)) ** 2
                - torch.sin(2 * thita3)
                * torch.sin(2 * alpha)
                * (torch.sin(delta_fi / 2)) ** 2
            )
        )
    return I


class FunModule(nn.Module):
    def __init__(self, is_main=True):
        super(FunModule, self).__init__()
        self.is_main = is_main
        if self.training:
            d_values = opt.optics_d
            d_tensor = torch.round(torch.tensor(d_values, dtype=torch.float32))
            self.d = nn.Parameter(d_tensor)

    def process_images(self, images, d):
        noise_config = {
            "enabled": True,
            "gaussian_mean": 0.0,
            "gaussian_std": opt.gaussian_std,
            "salt_pepper_prob": 0.02,
            "salt_pepper_intensity": opt.salt_noise,
            "channel_scale": {"r": 1.0, "g": 1.0, "b": 1.0},
        }

        device = d.device
        center_wavelengths = torch.tensor([480, 550, 680], device=device)
        std_dev = 80

        intensity_values = torch.zeros(num_pixels, num_wavelengths, device=device)
        intensity_values_2 = intensity_values.clone()
        intensity_values_3 = intensity_values.clone()

        for i in range(num_pixels):
            intensity_values[i, :] = intensity(d[i]).to(device)
            intensity_values_2[i, :] = intensity_2(d[i])
            intensity_values_3[i, :] = intensity_3(d[i])

        wavelengths_t = torch.linspace(450, 750, opt.num_wavelengths, device=device)
        R_response = torch.exp(-((wavelengths_t - center_wavelengths[0]) ** 2) / (2 * std_dev ** 2))
        G_response = torch.exp(-((wavelengths_t - center_wavelengths[1]) ** 2) / (2 * std_dev ** 2))
        B_response = torch.exp(-((wavelengths_t - center_wavelengths[2]) ** 2) / (2 * std_dev ** 2))

        R_response = R_response.float()
        G_response = G_response.float()
        B_response = B_response.float()

        intensity_values = intensity_values.view(num_pixels, num_wavelengths, 1, 1)
        intensity_values_2 = intensity_values_2.view(num_pixels, num_wavelengths, 1, 1)
        intensity_values_3 = intensity_values_3.view(num_pixels, num_wavelengths, 1, 1)

        if self.is_main:
            images = images.view(opt.batch_size, num_pixels, num_wavelengths, opt.patch_size, opt.patch_size)
        else:
            images = images.view(-1, num_pixels, num_wavelengths, images.shape[2], images.shape[3])

        images = images * intensity_values
        images_2 = images * intensity_values_2
        images_3 = images * intensity_values_3

        images = images.to(device)
        images_2 = images_2.to(device)
        images_3 = images_3.to(device)

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

        img_r_3 = (images_3 * R_response).sum(dim=2)
        img_g_3 = (images_3 * G_response).sum(dim=2)
        img_b_3 = (images_3 * B_response).sum(dim=2)

        def add_noise(img, channel="r"):
            if not noise_config["enabled"]:
                return img
            channel_scale = noise_config["channel_scale"].get(channel, 1.0)
            gaussian_std = noise_config["gaussian_std"] * channel_scale
            gaussian_noise = torch.randn_like(img) * gaussian_std
            salt_pepper_prob = noise_config["salt_pepper_prob"] * channel_scale
            salt_pepper = torch.rand_like(img)
            salt = (salt_pepper > (1 - salt_pepper_prob / 2)).float() * noise_config["salt_pepper_intensity"]
            pepper = (salt_pepper < salt_pepper_prob / 2).float() * -noise_config["salt_pepper_intensity"]
            return img + gaussian_noise + salt + pepper

        img_r = add_noise(img_r, "r")
        img_g = add_noise(img_g, "g")
        img_b = add_noise(img_b, "b")

        img_r_2 = add_noise(img_r_2, "r")
        img_g_2 = add_noise(img_g_2, "g")
        img_b_2 = add_noise(img_b_2, "b")

        img_r_3 = add_noise(img_r_3, "r")
        img_g_3 = add_noise(img_g_3, "g")
        img_b_3 = add_noise(img_b_3, "b")

        img = torch.stack([img_r, img_g, img_b], dim=2)
        img = torch.sum(img, dim=1)
        img = opt.norm * img / img.max()

        img_2 = torch.stack([img_r_2, img_g_2, img_b_2], dim=2)
        img_2 = torch.sum(img_2, dim=1)
        img_2 = opt.norm * img_2 / img_2.max()

        img_3 = torch.stack([img_r_3, img_g_3, img_b_3], dim=2)
        img_3 = torch.sum(img_3, dim=1)
        img_3 = opt.norm * img_3 / img_3.max()

        if opt.frame == 2:
            IMG = torch.cat((img, img_2), dim=1)
        elif opt.frame == 1:
            IMG = img
        else:
            IMG = torch.cat((img, img_2, img_3), dim=1)

        return IMG

    def forward(self, images):
        return self.process_images(images, self.d)


def slf_to_lf(cube, num_slices=opt.num_pixels, channels_per_slice=opt.num_wavelengths):
    batch_size, total_channels, height, width = cube.shape
    assert total_channels == num_slices * channels_per_slice, (
        "cube channel dimension must equal num_slices * channels_per_slice"
    )

    v = torch.zeros((batch_size, channels_per_slice, num_slices, height, width), device=cube.device)

    for i in range(num_slices):
        start_idx = i * channels_per_slice
        end_idx = (i + 1) * channels_per_slice
        v[:, :, i, :, :] = cube[:, start_idx:end_idx, :, :]

    return v.sum(dim=1)


def process_images(tensor, batch_size):
    assert tensor.shape[0] == batch_size, "First dimension must match batch_size"

    x_np = tensor.cpu().detach().numpy()
    processed = np.empty_like(x_np)

    for i in range(batch_size):
        edges = cv2.Canny(image=np.uint8(x_np[i] * 255), threshold1=10, threshold2=50)
        processed[i] = np.where(edges > 0, 1, 0)

    return torch.from_numpy(processed)


def process_images_plus(
    tensor,
    batch_size,
    kernel_size=opt.canny_size,
    threshold1=opt.canny_threshold[0],
    threshold2=opt.canny_threshold[1],
):
    assert tensor.shape[0] == batch_size, "First dimension must match batch_size"

    x_np = tensor.cpu().detach().numpy()
    processed = np.empty_like(x_np)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    for i in range(batch_size):
        img = np.uint8(x_np[i] * 255)
        edges = cv2.Canny(image=img, threshold1=threshold1, threshold2=threshold2)
        processed_image = np.where(edges > 0, 1, 0)
        expanded = cv2.dilate(processed_image.astype(np.uint8), kernel)
        processed[i] = expanded

    return torch.from_numpy(processed).float()


def generate_spectral_filters(num_pixels, num_wavelengths):
    wavelengths = np.linspace(400, 800, num=num_wavelengths)
    mu_list = np.linspace(500, 700, num_pixels) if num_pixels > 1 else np.array([600.0])
    delta_mu = (700 - 500) / (num_pixels - 1) if num_pixels > 1 else 200
    sigma_max = max(10, delta_mu / 4)
    np.random.seed(42)
    sigma_list = np.random.uniform(10, sigma_max, num_pixels)

    response = np.zeros((num_pixels, num_wavelengths))
    for i, (mu, sigma) in enumerate(zip(mu_list, sigma_list)):
        response[i] = np.exp(-(wavelengths - mu) ** 2 / (2 * sigma ** 2))

    return torch.tensor(response, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
