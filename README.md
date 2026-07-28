
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

&emsp;This repository contains the official PyTorch implementation (**PG-SVRT**), dataset (**DynaSpec**), and benchmark for the paper. The primary objective of this work is to advance compressive spectral imaging from traditional image-level reconstruction (i.e., **reconstructing HSIs from a single-frame measurement**) to video-level reconstruction (i.e., **reconstructing HSIs by fusing multi-frame measurements across the temporal domain**). In the paper, the proposed baseline (PG-SVRT) primarily evaluates the reconstruction of multi-frame measurements within a CASSI system utilizing a **fixed mask**. As shown in the video, video-level reconstruction can effectively enhance completeness, improve reconstruction accuracy and temporal consistency, and reduce flickering.

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
