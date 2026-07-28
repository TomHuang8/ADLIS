from torch.utils.data import Dataset
import numpy as np
import random
import cv2
from scipy.io import loadmat
import torch
from train_option import opt


class TrainDataset(Dataset):
    def __init__(self, data_root, crop_size, arg=True, bgr2rgb=True, stride=opt.stride, device='cuda'):
        self.crop_size = crop_size
        self.hypers = []
        self.bgrs = []
        self.arg = arg
        h, w = opt.sizes[0], opt.sizes[1]  # img shape
        self.stride = stride
        self.patch_per_line = (w-crop_size)//stride+1
        self.patch_per_colum = (h-crop_size)//stride+1
        self.patch_per_img = self.patch_per_line*self.patch_per_colum

        hyper_data_path = f'{data_root}/Train_Spec/'
        bgr_data_path = f'{data_root}/Train_RGB/'

        with open(f'{data_root}/split_txt/train_list.txt', 'r') as fin:
            hyper_list = [line.replace('\n','.mat') for line in fin]
            bgr_list = [line.replace('mat','jpg') for line in hyper_list]
        hyper_list.sort()
        bgr_list.sort()
        print(f'len(gt) of  dataset:{len(hyper_list)}')
        # print(f'len(bgr) of  dataset:{len(bgr_list)}')
        for i in range(len(hyper_list)):
            hyper_path = hyper_data_path + hyper_list[i]
            if 'mat' not in hyper_path:
                continue
            # with h5py.File(hyper_path, 'r') as mat:
            #     hyper =np.float32(np.array(mat['cube']))
            mat = loadmat(hyper_path)
            hyper = np.float32(mat['cube'])
            # hyper = hyper[300:900, 400:1200, :]
            hyper = np.transpose(hyper, [2, 0, 1])
            bgr = hyper
            self.hypers.append(hyper)
            self.bgrs.append(bgr)
            # mat.close()
            print(f'train scene {i} is loaded.')
        self.img_num = len(self.hypers)
        self.length = self.patch_per_img * self.img_num

    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, idx):
        stride = self.stride
        crop_size = self.crop_size
        img_idx, patch_idx = idx//self.patch_per_img, idx%self.patch_per_img
        h_idx, w_idx = patch_idx//self.patch_per_line, patch_idx%self.patch_per_line
        bgr = self.bgrs[img_idx]
        hyper = self.hypers[img_idx]
        bgr = bgr[:,h_idx*stride:h_idx*stride+crop_size, w_idx*stride:w_idx*stride+crop_size]
        hyper = hyper[:, h_idx * stride:h_idx * stride + crop_size,w_idx * stride:w_idx * stride + crop_size]
        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        if self.arg:
            bgr = self.arguement(bgr, rotTimes, vFlip, hFlip)
            hyper = self.arguement(hyper, rotTimes, vFlip, hFlip)
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)
        #     bgr = torch.FloatTensor(bgr.copy())
        #     hyper = torch.FloatTensor(hyper.copy())
        # return bgr, hyper



    def __len__(self):
        return self.patch_per_img*self.img_num


class ValidDataset(Dataset):
    def __init__(self, data_root, bgr2rgb=True):
        self.hypers = []
        self.bgrs = []
        hyper_data_path = f'{data_root}/Train_Spec/'
        bgr_data_path = f'{data_root}/Train_RGB/'
        with open(f'{data_root}/split_txt/valid_list.txt', 'r') as fin:
            hyper_list = [line.replace('\n', '.mat') for line in fin]
            bgr_list = [line.replace('mat','jpg') for line in hyper_list]
        hyper_list.sort()
        bgr_list.sort()
        print(f'len(hyper_valid) of SLF dataset:{len(hyper_list)}')
        print(f'len(bgr_valid) of SLF dataset:{len(bgr_list)}')
        for i in range(len(hyper_list)):
            hyper_path = hyper_data_path + hyper_list[i]
            if 'mat' not in hyper_path:
                continue
            # with h5py.File(hyper_path, 'r') as mat:
            #     hyper = np.float32(np.array(mat['cube']))
            mat = loadmat(hyper_path)
            hyper = np.float32(mat['cube'])
            # hyper = hyper[300:900, 400:1200, :]
            hyper = np.transpose(hyper, [2, 0, 1])
            # h, w = hyper.shape[1], hyper.shape[2]  # img shape
            # hyper = hyper[:, h//2-200:h//2+200, 0:400]
            bgr = hyper
            self.hypers.append(hyper)
            self.bgrs.append(bgr)
            # mat.close()
            print(f'val scene {i} is loaded.')

    def __getitem__(self, idx):
        hyper = self.hypers[idx]
        bgr = self.bgrs[idx]
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)
        # bgr = torch.FloatTensor(bgr.copy())
        # hyper = torch.FloatTensor(hyper.copy())
        # return bgr, hyper

    def __len__(self):
        return len(self.hypers)