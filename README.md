
<div align="center">
  
# ADLIS

*[Aperture-awared Dispersion 5D Light-field Imaging Spectrometer](https://arxiv.org/pdf/2607.04635)* 

<p align="middle">
  <a href="https://opensource.org/licenses/MIT">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="Visitors">
  <a href='https://arxiv.org/pdf/2607.04635'>
  <img src='https://img.shields.io/badge/Arxiv-2607.04635-A42C25?style=flat&logo=arXiv&logoColor=A42C25'></a> 
  <a href='https://github.com/lishiqiao/FlexiDim_demo'> 
  <img src='https://img.shields.io/badge/GitHub-Dataset%20(FlexiDim_demo)-blue'></a>
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
<div style="display: flex; justify-content: center; align-items: center; margin: 20px 0;">
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
    src="[https://github.com/user-attachments/assets/65d8ac00-41ce-4100-a61f-6441833d28f8]"
  >
    你的浏览器不支持视频播放
  </video>
</div>

&emsp;However, beyond our specific baseline, the open-sourced high-quality dynamic hyperspectral images dataset (DynaSpec) is highly versatile. For example, it can be readily adapted to advance research in a variety of other video-level hyperspectral tasks, such as reconstruction in various snapshot hyperspectral imaging systems with either adaptive or fixed modulation. It can also serve as approximately clean data for hyperspectral video denoising tasks.

If you find this repo or dataset useful, please give it a star ⭐ and consider citing our paper in your research. Thank you!

---

## 💾 Dataset
The example hyperspectral light-field dataset adopted in this work originates from **FlexiDim_demo** [(https://github.com/lishiqiao/FlexiDim_demo)].
The demo provides a single scene with 25-view hyperspectral data for method validation.

To run the ADLIS test pipeline:
1. Download the `data/` folder from the official Google Drive link provided by FlexiDim_demo:
👉 https://drive.google.com/drive/folders/15bdm__k6pzH18y-VnNB9X6QTfNbZETrT?usp=drive_link
2. Move the downloaded `data/` folder into the root directory of this repository.

> ⚠️ Important Notice:
> We do not redistribute or host the original dataset. All dataset resources are maintained by the original authors.
> Please check the FlexiDim_demo repository for data license, usage terms and future full dataset release updates.

 
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

If you use the 5D-SLF dataset, please consider citing the works below:
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
