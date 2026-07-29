
<div align="center">
  
# ADLIS

*[Aperture-awared Dispersion 5D Light-field Imaging Spectrometer](https://arxiv.org/pdf/2607.04635)* 

<p align="middle">
  <a href="https://opensource.org/licenses/MIT">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="Visitors">
  <a href='https://arxiv.org/pdf/2607.04635'>
  <img src='https://img.shields.io/badge/Arxiv-2607.04635-A42C25?style=flat&logo=arXiv&logoColor=A42C25'></a> 
  <a href='https://github.com/lishiqiao/FlexiDim_demo'>
  <img src='https://img.shields.io/badge/DATA-RealSLF_FlexiDim-00008B?style=flat&logo=github&logoColor=FFFFFF'></a>
  <a href="https://visitor-badge.laobi.icu/badge?page_id=TomHuang8.ADLIS">
  <img src="https://visitor-badge.laobi.icu/badge?page_id=TomHuang8.ADLIS&left_text=VISITORS&left_color=gray&right_color=%2342b983" alt="Visitors">
   </a>
  <img src="https://img.shields.io/github/last-commit/TomHuang8/ADLIS">

</p>

</div>

<br>

&emsp;This repository contains the official PyTorch implementation of ADLIS (Aperture-aware Dispersion Light-field Imaging Spectrometer), the end-to-end ADLI reconstruction framework, and benchmark for the paper. The primary objective of this work is to break the inherent dimension trade-off in single-detector high-dimensional imaging systems, advancing 5D spectral light-field (5D-SLF) acquisition from the conventional spatial-division paradigm to a novel encoding-integration paradigm.

In the paper, the proposed ADLIS adopts a compact birefringent coding model with a manufacturing-friendly, cost-effective birefringent quartz phase plate mounted on the aperture plane to realize angular-spectral-aware encoding. Different from microlens arrays that map separate viewpoints onto discrete sensor pixels, ADLIS superimposes light rays from all viewpoints onto each sensor pixel through aperture multiplexing, which maximizes spatial information throughput and enables full-resolution light field reconstruction. We further develop an end-to-end ADLI framework that jointly optimizes the differentiable phase plate thickness design and the 5D-SLF reconstruction decoder in a unified pipeline.

<!-- 进阶版：适配不同屏幕，精准居中+控大小 -->
<div style="display: flex; justify-content: center; align-items: center; margin: 10px 0;">
  <video 
    style="
      max-width: 90%;  /* 最大宽度占屏幕90%，适配小屏幕 */
      width: 600px;    /* 电脑端固定宽度，超过90%时自动缩小 */
      height: auto;    /* 保持宽高比 */
      border-radius: 8px; /* 可选：给视频加圆角，更美观 */
    "
    controls 
    muted 
    loop 
    autoplay
    src="https://github.com/user-attachments/assets/65d8ac00-41ce-4100-a61f-6441833d28f8.mp4"
  >
    Your browser does not support video playback.
  </video>
</div>

&emsp;If you find this repo or dataset useful, please give it a star ⭐ and consider citing our paper in your research.
THANK YOU!!

---

## 💾 Dataset
![image](https://github.com/TomHuang8/ADLIS/blob/main/Figures/dataset.png)

&emsp;Beyond validating the proposed ADLIS framework, the hyperspectral light-field dataset from [RealSLF, Optics Express 2025](https://opg.optica.org/abstract.cfm?uri=oe-33-21-45049) [https://github.com/lishiqiao/FlexiDim_demo] offers broad applicability. It can facilitate research on diverse 5D spectral light-field reconstruction tasks for snapshot computational imaging systems with different encoding strategies. Additionally, the dataset can be adopted as clean ground-truth data to support downstream tasks including light-field denoising and light field super-resolution.

 We use the 5D spectral light-field dataset from [FlexiDim_demo](https://github.com/lishiqiao/FlexiDim_demo). The raw data has dimensions $H \times W \times 36 \times 5 \times 5$, where $5 \times 5$ denotes the angular views and $36$ is the number of spectral bands. To adapt it for our ADLIS framework, we preprocess the data as follows:
* Spatial resolution cropped to $400 \times 400$ pixels,
* Angular views subsampled to a $3 \times 3$ grid,
* All 36 spectral channels retained.
After preprocessing, each scene is stored as a single MATLAB file with dimensions $400 \times 400 \times 324$, where $324 = 36 \text{ (bands)} \times 9 \text{ (views)}$.

### Download Links (Pre-processed Data)
* [Baidu Netdisk] https://pan.baidu.com/s/1gOKF_OkGgzSnxWzeebCavA?pwd=7ygn

The dataset contains 27 hyperspectral light-field scenes. We partition all scenes into 21 training scenes and 6 validation scenes for model training and quantitative evaluation. The raw 5D-SLF dataset contains full-resolution hyperspectral light-field images captured by GaiaField push-broom camera. Each `.mat` file contains a hyperspectral light-field image variable named `cube` with dimensions 400×400×324. The download package contains `Train_RGB` and `Test_RGB` folders with pseudo-color RGB renderings of the central view for training and test sets, to help users intuitively grasp the scene layouts.

### Dataset Structure
All processed hyperspectral `.mat` files are stored under `dataset/Train_Spec/`. The training/validation partition is defined by two text files placed within `dataset/split_txt/`:
- `train_list.txt`: List of 21 scene names for the training set
- `valid_list.txt`: List of 6 scene names for the validation set

```text
dataset_9_36/
├── Train_Spec/
│   ├── Tao_1_25real_9_36.mat
│   ├── Tao_2_25real_9_36.mat
│   ├── ...
│   ├── near_9_25real_9_36.mat
│   └── far_4_25real_9_36.mat
└── split_txt/
    ├── train_list.txt
    └── valid_list.txt
```

### Dataset Access
* The raw hyperspectral light-field data originates from [FlexiDim_demo](https://github.com/lishiqiao/FlexiDim_demo).
* The processed data cube can be downloaded here: [Baidu Netdisk](https://pan.baidu.com/s/1gOKF_OkGgzSnxWzeebCavA?pwd=7ygn).
#### 👉 Raw Dataset Download: Google Drive
To run the extended experiments:
Download the raw data from the official Google Drive link (https://drive.google.com/drive/folders/15bdm__k6pzH18y-VnNB9X6QTfNbZETrT?usp=drive_link) provided by the authors of the [RealSLF dataset](https://opg.optica.org/abstract.cfm?uri=oe-33-21-45049) linked with [FlexiDim_demo](https://github.com/lishiqiao/FlexiDim_demo).
Place the downloaded data under the root directory of this repository.
Perform the preprocessing (spatial cropping & viewpoint subsampling) described above, or modify the dataloader to adapt to our input format.
#### ⚠️ Important Notice:
This repository does not host or redistribute the complete raw dataset. All dataset resources are maintained by the original [RealSLF](https://opg.optica.org/abstract.cfm?uri=oe-33-21-45049) authors.
Please check the [FlexiDim_demo](https://github.com/lishiqiao/FlexiDim_demo) repository for data license, usage terms, future full dataset release updates, and the internal variable structure of each `.mat` file.

## 🚀 Getting Started

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/TomHuang8/ADLIS.git
cd ADLIS

# Create a conda environment (Python 3.6, tested)
conda create -n adlis python=3.6.13
conda activate adlis

# Install PyTorch 1.10.2 with CUDA 11.3 (tested)
pip install torch==1.10.2+cu113 torchvision==0.11.3 --index-url https://download.pytorch.org/whl/cu113
# For other CUDA versions of PyTorch 1.10, refer to https://pytorch.org/get-started/previous-versions/

# Install other dependencies
cd train_code
pip install -r requirements.txt
```

### 2. Data Preparation

For training and evaluation, we provide pre-processed data derived from [RealSLF](https://opg.optica.org/abstract.cfm?uri=oe-33-21-45049). 

Download the pre-processed data from [Baidu Netdisk](https://pan.baidu.com/s/1gOKF_OkGgzSnxWzeebCavA?pwd=7ygn) and organize as follows:

```text
<data_root>/
├── dataset_9_36/                              #  (training & testing)
```

### 3. Pre-trained Models

Download the pretrained model zoo from [Baidu Netdisk](https://pan.baidu.com/s/1bKeZNBdK3KtNdNo9QrVH4Q?pwd=wzak) and place them to `./model_zoo/`:

```text
model_zoo/
└── net_300epoch.pth   # Decoder: Restormer, Frame num: 2
```

### 4. Training

All training is launched from the `train_code/` directory. The training jointly optimizes the birefringent phase plate parameters (thickness map) and the reconstruction network in an end-to-end manner. The key argument `--frame` defines the number of measurement frames used for encoding:

| System Mode | `--frame` | Description |
| :---: | :---: | :--- |
| 1-frame | `1` | Single measurement with `theta = thita` |
| 2-frame | `2` | Two measurements with `theta = thita, thita2` |
| 3-frame | `3` | Three measurements with `theta = thita, thita2, thita3` |

**Default training** (2-frame, Restormer decoder):

```bash
cd train_code
python train.py --root /path/to/your/root --data_name dataset_9_36/
```

#### Train with different numbers of frames:
```bash
# 1-frame
python train.py --frame 1 --root /path/to/your/root --data_name dataset_9_36/

# 3-frame
python train.py --frame 3 --root /path/to/your/root --data_name dataset_9_36/
```
#### Resume training from a checkpoint:
```bash
python train.py --root /path/to/your/root --data_name dataset_9_36/ \
    --pretrained_model_path ./train_record/dataset_9_36/train/restormer/2_DO/net_90epoch.pth
```

**Key arguments**:
| Argument | Default | Description |
| :--- | :---: | :--- |
| `--method` | `restormer` | Reconstruction network (`restormer`, `miret`, `mprnet`, `edsr`) |
| `--frame` | `1` | Number of measurement frames (1, 2, or 3) |
| `--aperture_mode` | `DO` | Encoding mode (`DO` for proposed) |
| `--patch_size` | `400` | Training patch size |
| `--batch_size` | `1` | Batch size per GPU |
| `--gpu_id` | `'1'` | CUDA visible device(s) |
| `--end_epoch` | `3000` | Total training epochs |

After training, all outputs are saved under'./train_record/dataset/train/method/frame/aperture_mode/':
```text
train_record/dataset_9_36/train/restormer/2_DO/
├── train.log                          # Full training log
├── train_options.txt                  # Training configuration
├── model_params.txt                   # Model parameters
├── fun_module_params.txt              # Phase plate parameters
├── net_0epoch.pth                     # Checkpoint (saved at each epoch)
├── net_1epoch.pth
├── ...
├── net_90epoch.pth                    # Last epoch checkpoint
└── net_best.pth                       # Best checkpoint (saved when RMSE improves)
```
Key training options (see train_option.py for full list):
| Argument | Default | Description |
| :--- | :---: | :--- |
| `--end_epoch` | 3000 | Total training epochs |
| `--init_lr` | 0.0001 | Initial learning rate for reconstruction network |
| `--do_lr` | 0.002 | Initial learning rate for phase plate parameters |
| `--batch_size` | 1 | Batch size per GPU |
| `--patch_size` | 400 | Input patch size |
| `--stride` | 2 | Stride for patch extraction |
| `--gpu_id` | `'1'` | CUDA visible device(s) |
| `--DO_start_iter` | 300 | Iteration to start optimizing phase plate |
| `--DO_end_iter` | 50000 | Iteration to stop optimizing phase plate |
| `--k_loss_d` | 10 | Loss weight for the reconstruction loss |
| `--k_loss_e` | 0.001 | Loss weight for edge-aware loss |
| `--salt_noise` | 0.0 | Salt-and-pepper noise intensity |
| `--gaussian_std` | 0.01 | Gaussian noise standard deviation |

### 5. Testing
```bash
cd train_code
python test.py --root /path/to/your/root --data_name dataset_9_36/
```
This loads the default pretrained model from ../../model_zoo/net_90epoch.pth. To test a specific checkpoint:

```bash
python test.py --root /path/to/your/root --data_name dataset_9_36/ \
    --pretrained_model_path /path/to/your/checkpoint.pth
```

**Key arguments**:

| Argument | Default | Description |
| :--- | :---: | :--- |
| `--method` | `restormer` | Reconstruction network (`restormer`, `miret`, `mprnet`, `edsr`) |
| `--frame` | `1` | Number of measurement frames (1, 2 or 3) |
| `--aperture_mode` | `DO` | Encoding mode (`DO` for proposed) |
| `--pretrained_model_path` | `../../model_zoo/net_90epoch.pth` | Path to the pretrained checkpoint |

After testing, results are saved under ./test_record/dataset/test/method/timestamp/:
```text
test_record/dataset_9_36/test/restormer/2025-01-15_10-30-45/  #as a demo
├── test_options.txt                  # Testing configuration
├── model_params.txt                  # Model parameters
├── fun_module_params.txt             # Phase plate parameters
├── scene1_psnr_35.1234_ssim_0.9654.mat   # Reconstruction results (.mat)
├── scene2_psnr_34.5678_ssim_0.9587.mat
└── ...
```
The printed output includes average metrics across all test scenes:
```text
method:restormer, mrae:0.023456, rmse:0.034567, psnr:35.123456, ssim:0.965432, sam:12.345678  # as a demo
```

## ✒️ Citation
If this repo helps you, please consider citing our works:


```bibtex
@article{huang2026aperture,
  title={Aperture-aware Dispersion 5-D Light-field Imaging Spectrometer},
  author={Huang, Chenglong and Lv, Tao and Yang, Jianing and Zi, Chongde and Chen, Linsen and Cao, Xun},
  journal={arXiv preprint arXiv:2607.04635},
  year={2026}
}
```

If you use the 5D-SLF dataset [RealSLF](https://opg.optica.org/abstract.cfm?uri=oe-33-21-45049), please consider citing the works below:
```bibtex
@article{li2025realslf,
  title={RealSLF and FlexiDim: towards practical spectral light field imaging},
  author={Li, Shiqiao and Lv, Tao and Huang, Chenglong and Ye, Hao and Deng, Zhiwei and Hu, Lihao and Li, Qiping and Zi, Chongde and Chen, Linsen and Cao, Xun},
  journal={Optics Express},
  volume={33},
  number={21},
  pages={45049--45065},
  year={2025},
  publisher={Optica Publishing Group}
}
```

## 🙏 Acknowledgments
This code is built on [MST](https://github.com/caiyuanhao1998/MST) & [DynaSpec](https://github.com/nju-cite/DynaSpec). We sincerely thank the authors for sharing their codes.
We thank the authors of [FlexiDim_demo](https://github.com/lishiqiao/FlexiDim_demo) for publicly releasing the hyperspectral light-field dataset, which supports the experimental validation of this work.

## 📧 Contact
If you have any questions, please feel free to contact `Chenglong-Huang@smail.nju.edu.cn` or open an issue.
