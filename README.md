
<div align="center">
  
# ADLIS

*[Aperture-awared Dispersion 5D Light-field Imaging Spectrometer](https://arxiv.org/pdf/2607.04635)* 

<p align="middle">
  <a href="https://opensource.org/licenses/MIT">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="Visitors">
  <a href='https://arxiv.org/pdf/2607.04635'>
  <img src='https://img.shields.io/badge/Arxiv-2607.04635-A42C25?style=flat&logo=arXiv&logoColor=A42C25'></a> 
  <a href='https://huggingface.co/datasets/Flipped99/DynaSpec'> 
  <img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow'></a>
  <a href="https://visitor-badge.laobi.icu/badge?page_id=TomHuang8.ADLIS">
  <img src="https://visitor-badge.laobi.icu/badge?page_id=TomHuang8.ADLIS&left_text=VISITORS&left_color=gray&right_color=%2342b983" alt="Visitors">
   </a>
  <img src="https://img.shields.io/github/last-commit/TomHuang8/ADLIS">

</p>

</div>

<br>

&emsp;This repository contains the official PyTorch implementation of ADLIS (Aperture-aware Dispersion Light-field Imaging Spectrometer), the end-to-end ADLI reconstruction framework, and benchmark for the paper. The primary objective of this work is to break the inherent dimension trade-off in single-detector high-dimensional imaging systems, advancing 5D spectral light-field (5D-SLF) acquisition from the conventional spatial-division paradigm to a novel encoding-integration paradigm.

In the paper, the proposed ADLIS adopts a compact birefringent coding model with a manufacturing-friendly, cost-effective birefringent quartz phase plate mounted on the aperture plane to realize angular-spectral-aware encoding. Different from microlens arrays that map separate viewpoints onto discrete sensor pixels, ADLIS superimposes light rays from all viewpoints onto each sensor pixel through aperture multiplexing, which maximizes spatial information throughput and enables full-resolution light field reconstruction. We further develop an end-to-end ADLI framework that jointly optimizes the differentiable phase plate thickness design and the 5D-SLF reconstruction decoder in a unified pipeline.

 
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

## 🙏 Acknowledgments
This code is built on [MST](https://github.com/caiyuanhao1998/MST) & [DynaSpec](https://github.com/nju-cite/DynaSpec). We sincerely thank the authors for sharing their codes.

## 📧 Contact
If you have any questions, please feel free to contact `Chenglong-Huang@smail.nju.edu.cn` or open an issue.
