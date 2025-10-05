<div align="center">
<h1>Mixed-Bit Sampling Marking: Towards Unifying Document Authentication in Copy-Sensitive Graphical Codes</h1>

Jiale Chen<sup>1,3</sup>, Li Dong<sup>1</sup>, and Wei Wang<sup>2,3</sup>, Weiwei Sun<sup>4</sup>, Yushu Zhang<sup>5</sup>, Jiantao Zhou<sup>6</sup>

<sup>1</sup>Ningbo University,
<sup>2</sup>Shenzhen MSU-BIT University,
<sup>3</sup>Beijing Institute of Technology,
<sup>4</sup>Alibaba Group,
<sup>5</sup>Jiangxi University of Finance and Economics,
<sup>6</sup>University of Macau
</div>

# Introduction
1. This is the official implementation of the paper titled *"Mixed-Bit Sampling Marking: Towards Unifying Document Authentication in Copy-Sensitive Graphical Codes"*
2. The work is based on our preliminary research [MSG](https://ieeexplore.ieee.org/document/10376267), and more details of the preliminary work can be found at [GitHub Repository](https://github.com/chenoly/MSG).
3. The dataset used in this work is publicly available at [Kaggle](https://www.kaggle.com/datasets/chenoly/msw-dataset).

## Overview
Combating counterfeit products is crucial for maintaining a healthy market. Recently, Copy Sensitive Graphical Codes (CSGC) have garnered significant attention due to their high sensitivity to illegal physical copying. Copy Detection Patterns (CDP) and Two-Level QR Codes (2LQR code) are two representative methods. CDP offers high efficiency and low cost, enabling use in document authentication and product anti-counterfeiting, and has achieved broad commercial adoption. In contrast, 2LQR code, as a consumer-grade document authentication solution, provides additional private message sharing functionalities.
We observe that both the CDP and 2LQR code can be synthesized using textured patterns. To this end, we propose a flexible framework that integrates the stochastic anti-counterfeiting properties of CDP with the private message sharing of 2LQR code. Specifically, we model CDP as a random noise image composed of multiple textured patterns similar to those in 2LQR code, where each pattern represents an informative digit. Thus, both codes can be generated through textured pattern design. We formulate this as a constrained optimization framework called Mixed-Bit Sampling Marking (MSM). The objective incorporates white pixel ratio and spatial randomness, with constraints defined by a flexible modulation function (e.g., DCT or Pearson similarity), customizable to user needs. A two-step sampling algorithm solves the optimization.
We demonstrate CDP and 2LQR codes generated via MSM and validate their ability to inherit advantages from both approaches. Experiments show that MSM-generated texture patterns effectively synthesize both CDPs and 2LQR codes, preserving their advantages while offering a novel, flexible solution for document authentication.

## Requirements
- Python
- bchlib==0.14.0
- opencv-python==3.4.2.16
- numpy
- scikit-image
- scipy
- hashlib
- pytorch

```bibtex
@ARTICLE{chen2025mixed,
  author={Chen, Jiale and Dong, Li and Wang, Wei and Wang, Rangding and Sun, Weiwei and Zhang, Yushu and Zhou, Jiantao},
  journal={IEEE Transactions on Information Forensics and Security}, 
  title={Mixed-Bit Sampling Marking: Towards Unifying Document Authentication in Copy-Sensitive Graphical Codes}, 
  year={2025},
  doi={10.1109/TIFS.2025.3616619}
}
