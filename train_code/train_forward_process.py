import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import os
from architecture import *
from train_option import opt
import cv2
import numpy as np
import torch

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# 参数设置
A1 = 1
wavelengths = torch.linspace(450, 750, opt.num_wavelengths)  # Visible spectrum in nm
base_depth = 500  # um
max_depth = 1000  # um


n_o = opt.n_o  # Ordinary refractive index of quartz
n_e = opt.n_e  # Extraordinary refractive index of quartz
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
    return (2 * torch.pi * d * 1e-6 * delta_n) / (lambda_ * 1e-9)  # lambda in nm to m


def intensity(d):
    I = torch.zeros_like(wavelengths)
    for i, lambda_ in enumerate(wavelengths):
        delta_fi = phase_delay(d, lambda_, n_o, n_e)
        # I[i] = 0.5 * (1 + np.cos(delta_fi))  # Assuming initial light is at 45 degrees to the optical axis
        I[i] = A1 * A1 * ((torch.cos(alpha - thita)) ** 2 - torch.sin(2 * thita) * torch.sin(2 * alpha) * (
            torch.sin(delta_fi / 2)) ** 2)
    return I


def intensity_2(d):
    I = torch.zeros_like(wavelengths)
    for i, lambda_ in enumerate(wavelengths):
        delta_fi = phase_delay(d, lambda_, n_o, n_e)
        # I[i] = 0.5 * (1 + np.cos(delta_fi))  # Assuming initial light is at 45 degrees to the optical axis
        I[i] = A1 * A1 * ((torch.cos(alpha - thita2)) ** 2 - torch.sin(2 * thita2) * torch.sin(2 * alpha) * (
            torch.sin(delta_fi / 2)) ** 2)
    return I


def intensity_3(d):
    I = torch.zeros_like(wavelengths)
    for i, lambda_ in enumerate(wavelengths):
        delta_fi = phase_delay(d, lambda_, n_o, n_e)
        # I[i] = 0.5 * (1 + np.cos(delta_fi))  # Assuming initial light is at 45 degrees to the optical axis
        I[i] = A1 * A1 * ((torch.cos(alpha - thita3)) ** 2 - torch.sin(2 * thita2) * torch.sin(2 * alpha) * (
            torch.sin(delta_fi / 2)) ** 2)
    return I


class FunModule(nn.Module):
    def __init__(self, is_main=True):
        super(FunModule, self).__init__()
        # self.d = nn.Parameter(torch.round(base_depth + (max_depth - base_depth) * torch.rand(9)))
        self.is_main = is_main
        # self.d = nn.Parameter(750 * torch.ones(num_pixels, device='cuda'))
        if self.training:
            # self.d = nn.Parameter(torch.round(base_depth + (max_depth - base_depth) * torch.rand(num_pixels)))
            d_values = opt.optics_d
            d_tensor = torch.round(torch.tensor(d_values, dtype=torch.float32))
            self.d = nn.Parameter(d_tensor)

    def process_images(self, images, d):
        # 噪声配置字典 - 可根据需要修改这些值
        noise_config = {
            'enabled': True,  # 是否启用噪声
            'gaussian_mean': 0.0,  # 高斯噪声均值
            'gaussian_std': opt.gaussian_std,  # 高斯噪声标准差
            'salt_pepper_prob': 0.02,  # 椒盐噪声概率
            'salt_pepper_intensity': opt.salt_noise,  # 椒盐噪声强度
            'channel_scale': {  # 各通道噪声缩放因子
                'r': 1.0,
                'g': 1.0,
                'b': 1.0
            }
        }
        device = d.device  # 获取设备信息
        center_wavelengths = torch.tensor([480, 550, 680], device=device)
        std_dev = 80
        intensity_values = torch.zeros(num_pixels, num_wavelengths, device=device)  # 在 GPU 上创建 tensor
        intensity_values_2 = intensity_values.clone()
        intensity_values_3 = intensity_values.clone()
        for i in range(num_pixels):
            # intensity_values[i, :] = intensity(d[i]).clone().detach().to('cuda:0')
            intensity_values[i, :] = intensity(d[i]).to('cuda:0')
            intensity_values_2[i, :] = intensity_2(d[i])
            intensity_values_3[i, :] = intensity_3(d[i])
        # Generate RGB response curves
        wavelengths = torch.linspace(450, 750, opt.num_wavelengths, device=device)
        R_response = torch.exp(-((wavelengths - center_wavelengths[0]) ** 2) / (2 * std_dev ** 2))
        G_response = torch.exp(-((wavelengths - center_wavelengths[1]) ** 2) / (2 * std_dev ** 2))
        B_response = torch.exp(-((wavelengths - center_wavelengths[2]) ** 2) / (2 * std_dev ** 2))

        # Convert to PyTorch tensors
        R_response = R_response.float()
        G_response = G_response.float()
        B_response = B_response.float()

        # Reshape intensity_values to match the dimensions of images
        intensity_values = intensity_values.view(num_pixels, num_wavelengths, 1, 1)
        intensity_values_2 = intensity_values_2.view(num_pixels, num_wavelengths, 1, 1)
        intensity_values_3 = intensity_values_3.view(num_pixels, num_wavelengths, 1, 1)
        # Apply intensity mask and convert to RGB color space
        if self.is_main:
            images = images.view(opt.batch_size, num_pixels, num_wavelengths, opt.patch_size, opt.patch_size)
        else:
            images = images.view(-1, num_pixels, num_wavelengths, images.shape[2], images.shape[3])
        images = images * intensity_values
        images_2 = images * intensity_values_2
        images_3 = images * intensity_values_3
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        img_b_2 = (images_2 * B_response).sum(dim=2) # images_2这里一直都是写的3，逆天（2025.01.13）

        img_r_3 = (images_3 * R_response).sum(dim=2)
        img_g_3 = (images_3 * G_response).sum(dim=2)
        img_b_3 = (images_3 * B_response).sum(dim=2)

        # 添加随机噪声
        def add_noise(img, channel='r'):
            """根据配置为图像添加随机噪声"""
            if not noise_config['enabled']:
                return img

            # 获取通道特定的缩放因子
            channel_scale = noise_config['channel_scale'].get(channel, 1.0)

            # 高斯噪声
            gaussian_std = noise_config['gaussian_std'] * channel_scale
            gaussian_noise = torch.randn_like(img) * gaussian_std

            # 椒盐噪声
            salt_pepper_prob = noise_config['salt_pepper_prob'] * channel_scale
            salt_pepper = torch.rand_like(img)
            salt = (salt_pepper > (1 - salt_pepper_prob / 2)).float() * noise_config['salt_pepper_intensity']
            pepper = (salt_pepper < salt_pepper_prob / 2).float() * -noise_config['salt_pepper_intensity']

            return img + gaussian_noise + salt + pepper

        # 为RGB通道添加噪声，可通过channel_scale设置不同强度
        img_r = add_noise(img_r, 'r')
        img_g = add_noise(img_g, 'g')
        img_b = add_noise(img_b, 'b')

        img_r_2 = add_noise(img_r_2, 'r')
        img_g_2 = add_noise(img_g_2, 'g')
        img_b_2 = add_noise(img_b_2, 'b')

        img_r_3 = add_noise(img_r_3, 'r')
        img_g_3 = add_noise(img_g_3, 'g')
        img_b_3 = add_noise(img_b_3, 'b')

        # Combine channels into a single image
        img = torch.stack([img_r, img_g, img_b], dim=2)
        img = torch.sum(img, dim=1)
        # img = opt.norm * (img-img.min())+1e-5 / (img.max().item()-img.min())
        img = opt.norm * img / img.max()

        img_2 = torch.stack([img_r_2, img_g_2, img_b_2], dim=2)
        img_2 = torch.sum(img_2, dim=1)
        # img_2 = opt.norm * (img_2-img_2.min())+1e-5 / (img_2.max().item()-img_2.min())
        img_2 = opt.norm * img_2 / img_2.max()

        img_3 = torch.stack([img_r_3, img_g_3, img_b_3], dim=2)
        img_3 = torch.sum(img_3, dim=1)
        # img_3 = opt.norm * (img_3-img_3.min())+1e-5 / (img_3.max().item()-img_3.min())
        img_3 = opt.norm * img_3 / img_3.max()

        if opt.frame == 2:
            IMG = torch.cat((img, img_2), dim=1)
        elif opt.frame == 1:
            IMG = img
        else:
            IMG = torch.cat((img, img_2, img_3), dim=1)

        # # 确保最终图像值在[0,1]范围内
        # IMG = torch.clamp(IMG, 0, 1)

        return IMG

    def forward(self, images):
        # This is a simple linear transformation, you can modify it according to your needs
        return self.process_images(images, self.d)


def slf_to_lf(cube,num_slices=opt.num_pixels, channels_per_slice=opt.num_wavelengths):
    """
    将输入张量 cube 切分并计算每个切片组的和。

    参数:
    - cube: PyTorch 张量，形状为 (batch_size, total_channels, height, width)，
            其中 total_channels = num_slices * channels_per_slice
    - num_slices: 切片数量（默认值为 9）
    - channels_per_slice: 每个切片组的通道数（默认值为 36）

    返回:
    - v_integrated: 经过通道聚合后的 PyTorch 张量，形状为 (batch_size, num_slices, height, width)
    """
    # 获取输入的维度
    batch_size, total_channels, height, width = cube.shape

    # 检查 cube 的通道数是否符合预期
    assert total_channels == num_slices * channels_per_slice, \
        "cube 的第二个维度大小必须等于 num_slices * channels_per_slice"

    # 初始化切分后的五维张量 v
    v = torch.zeros((batch_size, channels_per_slice, num_slices, height, width), device=cube.device)

    # 填充 v 的数据
    for i in range(num_slices):
        start_index = i * channels_per_slice
        end_index = (i + 1) * channels_per_slice
        v[:, :, i, :, :] = cube[:, start_index:end_index, :, :]

    # 对切片组聚合，计算 v_integrated
    v_integrated = v.sum(dim=1)

    return v_integrated


def process_images(tensor, batch_size):
    # 确保输入tensor的第一个维度与batch_size相同
    assert tensor.shape[0] == batch_size, "The first dimension of the tensor must be the same as batch_size"

    # 将tensor转换为numpy数组
    x_np = tensor.cpu().detach().numpy()

    # 创建一个新的numpy数组来存储处理后的图像
    processed_images = np.empty_like(x_np)

    for i in range(batch_size):
        # 使用Canny函数进行边缘检测
        edges = cv2.Canny(image=np.uint8(x_np[i]*255), threshold1=10, threshold2=50)  # 这里的阈值可能需要根据你的图像进行调整
        # 将边缘点赋值为1，非边缘点赋值为0
        processed_images[i] = np.where(edges > 0, 1, 0)

    # 将处理后的numpy数组转换回tensor
    processed_images_tensor = torch.from_numpy(processed_images)

    return processed_images_tensor


def process_images_plus(tensor, batch_size, kernel_size=opt.canny_size, threshold1=opt.canny_threshold[0], threshold2=opt.canny_threshold[1]):
    """
    处理批量图像，进行边缘检测，并扩展边缘区域。

    Parameters:
    - tensor (torch.Tensor): 输入的图像张量，形状为 [batch_size, H, W]
    - batch_size (int): 批次大小
    - kernel_size (int): 用于膨胀操作的邻域大小，默认是 15
    - threshold1 (int): Canny 边缘检测的第一个阈值
    - threshold2 (int): Canny 边缘检测的第二个阈值

    Returns:
    - torch.Tensor: 处理后的图像张量，形状与输入相同
    """
    # 确保输入 tensor 的第一个维度与 batch_size 相同
    assert tensor.shape[0] == batch_size, "The first dimension of the tensor must be the same as batch_size"

    # 将 tensor 转换为 numpy 数组
    x_np = tensor.cpu().detach().numpy()
    # 创建一个新的 numpy 数组来存储处理后的图像
    processed_images = np.empty_like(x_np)
    # 创建一个结构元素（膨胀的窗口）
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # 遍历批次中的每个图像
    for i in range(batch_size):
        # 将图像从 [0, 1] 范围转换为 [0, 255]，以便进行 Canny 边缘检测
        img = np.uint8(x_np[i] * 255)
        # 使用 Canny 边缘检测
        edges = cv2.Canny(image=img, threshold1=threshold1, threshold2=threshold2)
        # 将边缘点赋值为 1，非边缘点赋值为 0
        processed_image = np.where(edges > 0, 1, 0)
        # 使用膨胀操作扩大边缘区域
        expanded_edges = cv2.dilate(processed_image.astype(np.uint8), kernel)
        # 存储处理后的图像
        processed_images[i] = expanded_edges

    # 将处理后的 numpy 数组转换回 tensor
    processed_images_tensor = torch.from_numpy(processed_images).float()
    return processed_images_tensor


def generate_spectral_filters(num_pixels, num_wavelengths):
    """
    生成具有物理合理性的光谱滤波张量

    参数:
        num_pixels (int): 滤波片数量
        num_wavelengths (int): 每个滤波片的波长采样点数

    返回:
        torch.Tensor: 形状为(num_pixels, num_wavelengths, 1, 1)的张量
    """
    # 波长范围设置（400-800nm覆盖常见光学范围）
    wavelengths = np.linspace(400, 800, num=num_wavelengths)
    # 中心波长均匀分布（500-700nm）
    mu_list = np.linspace(500, 700, num_pixels) if num_pixels > 1 else np.array([600.0])
    # 动态计算带宽限制
    delta_mu = (700 - 500) / (num_pixels - 1) if num_pixels > 1 else 200
    sigma_max = max(10, delta_mu / 4)  # 保证最小10nm带宽
    # 生成随机带宽参数
    np.random.seed(42)  # 保证可重复性
    sigma_list = np.random.uniform(10, sigma_max, num_pixels)
    # 生成高斯响应曲线
    response = np.zeros((num_pixels, num_wavelengths))
    for i, (mu, sigma) in enumerate(zip(mu_list, sigma_list)):
        response[i] = np.exp(-(wavelengths - mu) ** 2 / (2 * sigma ** 2))
    # 转换为PyTorch张量并调整维度
    return torch.tensor(response, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)