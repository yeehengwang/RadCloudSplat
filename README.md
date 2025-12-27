# RadCloudSplat
<h3><a href="">RadCloudSplat: Scatterer-Driven 3D Gaussian Splatting with Point-Cloud Priors for Radiomap Extrapolation</a></h3>

[Yiheng Wang](https://yeehengwang.github.io/), [Ye Xue](https://yokoxue.github.io/), Shutao Zhang, Hongmiao Fan and [Tsung-Hui Chang](https://myweb.cuhk.edu.cn/changtsunghui/Home) 

 <a href='https://arxiv.org/abs/2502.12686'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>

Thanks for your interest in our work. This repository contains code and links to the **RadCloudSplat** method for radiomap extrapolation. In this work, we first extended 3DGS to the radio frequency domain, leveraging **camera-free RadCloudSplat** to extrapolate RSSs with high accuracy from sparse measurements in an outdoor environment. By efficiently selecting the means of key virtual scatterers from dense point clouds aided by the **relaxed-mean (RM) scheme**, the model captured **intricate multi-path propagation characteristics**. Experiments and analysis validated the effectiveness of these scatterers, advancing the state-of-the-art in wireless network modeling and extrapolation performance and highlighting the transformative potential of integrating advanced 3D modeling techniques with wireless propagation analysis for next-generation applications in the radio domain.


## Introduction

Schematic illustration of **RadCloudSplat**, comprising three major parts: 1) Relaxed-Mean Reparameterization for Key Virtual Scatters Positions Extraction. 2) Camera-Free RadCloudSplat Model for RSS Synthesis. 3) Optimizing RadCloudSplat Scheme

![](assets/arch.png)

## News
Due to copyright issues regarding the measurement real data from wireless network, we are unable to provide the data used in the paper.
## Release
- [x] Release the demo dataset revised from NeRF2 datasets. 
- [x] Release the training code. 
- [x] Release the inference code. 
- [x] Release the paper of RadCloudSplat on [arXiv](https://arxiv.org/abs/2502.12686). 

## Training & Evaluation
A small demo dataset in ./demo_data is included to help quickly verify the code, which can be executed using the following command:
```python
python train_radcloudsplat.py
```
More datasets can be found [here](https://github.com/XPengZhao/NeRF2?tab=readme-ov-file).<be>

When you finish the training, you can inference the trained model by using the following command:
```python
python train_radcloudsplat.py --mode test
```




## Citation

If you find our work useful in your research, please consider citing RadCloudSplat:

```tex
@article{wang2025radsplatter,
  title={RadSplatter: Extending 3D Gaussian Splatting to Radio Frequencies for Wireless Radiomap Extrapolation},
  author={Wang, Yiheng and Xue, Ye and Zhang, Shutao and Chang, Tsung-Hui},
  journal={arXiv preprint arXiv:2502.12686},
  year={2025}
}
```

## Acknowledgement

We thank Dr. Xiaopeng Zhao, the authors of [NeRF2](https://github.com/XPengZhao/NeRF2), for making their code and dataset available. 
