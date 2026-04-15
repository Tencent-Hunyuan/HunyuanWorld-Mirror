[中文阅读](README_zh.md)
# **HunyuanWorld-Mirror**

<p align="center">
  <img src="assets/teaser.jpg" width="95%" alt="HunyuanWorld-Mirror Teaser">
</p>

<p align="center">
<a href='https://3d-models.hunyuan.tencent.com/world/'><img src='https://img.shields.io/badge/Project-Page-green'></a>
<a href='https://3d-models.hunyuan.tencent.com/world/worldMirror1_0/HYWorld_Mirror_Tech_Report.pdf'><img src='https://img.shields.io/badge/Technique-Report-red'></a>
<a href='https://huggingface.co/tencent/HunyuanWorld-Mirror'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>
<a href='https://huggingface.co/spaces/tencent/HunyuanWorld-Mirror'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-orange'></a>
<a href=https://discord.gg/dNBrdrGGMa target="_blank"><img src= https://img.shields.io/badge/Discord-white.svg?logo=discord height=22px></a>
  <a href=https://x.com/TencentHunyuan target="_blank"><img src=https://img.shields.io/badge/Hunyuan-black.svg?logo=x height=22px></a>
<p align="center">



HunyuanWorld-Mirror is a versatile feed-forward model for comprehensive 3D geometric prediction. It integrates diverse geometric priors (**camera poses**, **calibrated intrinsics**, **depth maps**) and simultaneously generates various 3D representations (**point clouds**, **multi-view depths**, **camera parameters**, **surface normals**, **3D Gaussians**) in a single forward pass.



https://github.com/user-attachments/assets/146a9a25-5eb7-4400-aa09-5b58e1d10a5e




## 🔥🔥🔥 Updates
* **[Apr 16, 2026]**: 🤗 We release [HY-World-2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0), state-of-the-art 3D world model! 
* **[Dec 17, 2025]**:  🤗 We release [HunyuanWorld-1.5](https://github.com/Tencent-Hunyuan/HY-WorldPlay), enabling real-time world creation and play! 
* **[Nov 7, 2025]**: 🚀🚀🚀 We release the training and evaluation code. See [Training Instructions](#🤖-training) and [Evaluation Instructions](#📊-evaluation).
* **[Oct 22, 2025]**: We release the inference code and model weights. [Download Pretrained Model](https://huggingface.co/tencent/HunyuanWorld-Mirror).

> Join our **[Wechat](#)** and **[Discord](https://discord.gg/dNBrdrGGMa)** group to discuss and find help from us.

| Wechat Group                                     | Xiaohongshu                                           | X                                           | Discord                                           |
|--------------------------------------------------|-------------------------------------------------------|---------------------------------------------|---------------------------------------------------|
| <img src="assets/qrcode/wechat.png"  height=140> | <img src="assets/qrcode/xiaohongshu.png"  height=140> | <img src="assets/qrcode/x.png"  height=140> | <img src="assets/qrcode/discord.png"  height=140> | 


## Table of Contents

- [Introduction](#☯️-hunyuanworld-mirror-introduction)
  - [Architecture](#architecture)
- [Installation](#🛠️-dependencies-and-installation)
- [Quick Start](#🎮-quick-start)
  - [Online Demo](#online-demo)
  - [Local Demo](#local-demo)
- [Download Pretrained Models](#📦-download-pretrained-models)
- [Inference with Images & Priors](#🚀-inference-with-images--priors)
  - [Example Code Snippet](#example-code-snippet)
  - [Output Format](#output-format)
  - [Inference with More Functions](#inference-with-more-functions)
- [Post 3DGS Optimization (Optional)](#🎯-post-3dgs-optimization-optional)
  - [Install Dependencies](#install-dependencies)
  - [Optimization](#optimization)
- [Performance](#🔮-performance)
  - [Point Cloud Reconstruction](#point-cloud-reconstruction)
  - [Novel View Synthesis](#novel-view-synthesis)
  - [Boost of Geometric Priors](#boost-of-geometric-priors)
- [Training](#🤖-training)
  - [Training Data Preparation](#training-data-preparation)
  - [Install Dependencies](#install-dependencies)
  - [Training Commands](#training-commands)
  - [Fine-tuning Commands](#fine-tuning-commands)
- [Evaluation](#📊-evaluation)
  - [Evaluation Data Preparation](#evaluation-data-preparation)
  - [Install Dependencies](#install-dependencies)
  - [Evaluation Commands](#evaluation-commands)
    - [1. Point Map Reconstruction](#1-point-map-reconstruction)
    - [2. Surface Normal Estimation](#2-surface-normal-estimation)
    - [3. Novel View Synthesis](#3-novel-view-synthesis)
    - [4. Depth Estimation](#4-depth-estimation)
    - [5. Camera Pose Estimation](#5-camera-pose-estimation)
- [Open-Source Plan](#📑-open-source-plan)
- [BibTeX](#🔗-bibtex)
- [Contact](#📧-contact)
- [Acknowledgments](#acknowledgements)


## ☯️ **HunyuanWorld-Mirror Introduction**

### Architecture
HunyuanWorld-Mirror consists of two key components:

**(1) Multi-Modal Prior Prompting**: A mechanism that embeds diverse prior modalities,
including calibrated intrinsics, camera pose, and depth, into the feed-forward model. Given any subset of the available priors, we utilize several lightweight encoding layers to convert each modality into structured tokens.

**(2) Universal Geometric Prediction**: A unified architecture capable of handling
the full spectrum of 3D reconstruction tasks from camera and depth estimation to point map regression, surface normal estimation, and novel view synthesis. 

<p align="left">
  <img src="assets/arch.png">
</p>


## 🛠️ Dependencies and Installation
We recommend using CUDA version 12.4 for the manual installation.
```shell
# 1. Clone the repository
git clone https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror
cd HunyuanWorld-Mirror

# 2. Create conda environment
conda create -n hunyuanworld-mirror python=3.10 cmake=3.14.0 -y
conda activate hunyuanworld-mirror

# 3. Install PyTorch and other dependencies using pip
# For CUDA 12.4
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124

# 4. Install pip dependencies
pip install -r requirements.txt

# 5. Install gsplat for 3D Gaussian Splatting rendering 
# For CUDA 12.4
pip install gsplat --index-url https://docs.gsplat.studio/whl/pt24cu124
```

## 🎮 Quick Start
We provide a Gradio demo for the HunyuanWorld-Mirror model for quick start.

<p align="center">
  <img src="assets/gradio_demo.gif" width="95%" alt="HunyuanWorld-Mirror Gradio Demo">
</p>


### Online Demo
Try our online demo without installation: [🤗 Hugging Face Demo](https://huggingface.co/spaces/tencent/HunyuanWorld-Mirror)

### Local Demo
```shell
# 1. Install requirements for gradio demo
pip install -r requirements_demo.txt 
# For Windows, please replace onnxruntime and gsplat with Windows wheels (comments in requirements_demo.txt)
# 2. Launch gradio demo locally
python app.py
```

## 📦 Download Pretrained Models
To download the HunyuanWorld-Mirror model, first install the huggingface-cli:
```
python -m pip install "huggingface_hub[cli]"
```
Then download the model using the following commands:
```
huggingface-cli download tencent/HunyuanWorld-Mirror --local-dir ./ckpts
```
> **Note**: For inference, the model weights will be automatically downloaded from Hugging Face when running the inference scripts, so you can skip this manual download step if preferred.

## 🚀 Inference with Images & Priors
### Example Code Snippet
```python
from pathlib import Path
import torch
from src.models.models.worldmirror import WorldMirror
from src.utils.inference_utils import extract_load_and_preprocess_images

# --- Setup ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = WorldMirror.from_pretrained("tencent/HunyuanWorld-Mirror").to(device)

# --- Load Data ---
# Load a sequence of N images into a tensor
inputs = {}
inputs['img'] = extract_load_and_preprocess_images(
    Path("path/to/your/data"), # video or directory containing images 
    fps=1, # fps for extracing frames from video
    target_size=518
).to(device)  # [1,N,3,H,W], in [0,1]
# -- Load Priors (Optional) --
# Configure conditioning flags and prior paths
cond_flags = [0, 0, 0]  # [camera_pose, depth, intrinsics]
prior_data = {
    'camera_poses': None,      # Camera pose tensor [1, N, 4, 4]
    'depthmap': None,         # Depth map tensor [1, N, H, W]
    'camera_intrs': None # Camera intrinsics tensor [1, N, 3, 3]
}
for idx, (key, data) in enumerate(prior_data.items()):
    if data is not None:
        cond_flags[idx] = 1
        inputs[key] = data

# --- Inference ---
with torch.no_grad():
    predictions = model(views=inputs, cond_flags=cond_flags)
```

### Output Format

```python
# Geometry outputs
pts3d_preds, pts3d_conf = predictions["pts3d"][0], predictions["pts3d_conf"][0]       # 3D point cloud in world coordinate: [S, H, W, 3] and point confidence: [S, H, W]
depth_preds, depth_conf = predictions["depth"][0], predictions["depth_conf"][0]       # Z-depth in camera frame: [S, H, W, 1] and depth confidence: [S, H, W]
normal_preds, normal_conf = predictions["normals"][0], predictions["normals_conf"][0] # Surface normal in camera coordinate: [S, H, W, 3] and normal confidence: [S, H, W]

# Camera outputs
camera_poses = predictions["camera_poses"][0]  # Camera-to-world poses (OpenCV convention): [S, 4, 4]
camera_intrs = predictions["camera_intrs"][0]  # Camera intrinsic matrices: [S, 3, 3]
camera_params = predictions["camera_params"][0]   # Camera vector: [S, 9] (translation, quaternion rotation, fov_v, fov_u)

# 3D Gaussian Splatting outputs
splats = predictions["splats"]
means = splats["means"][0].reshape(-1, 3)      # Gaussian means: [N, 3]
opacities = splats["opacities"][0].reshape(-1) # Gaussian opacities: [N]
scales = splats["scales"][0].reshape(-1, 3)    # Gaussian scales: [N, 3]
quats = splats["quats"][0].reshape(-1, 4)      # Gaussian quaternions: [N, 4]
sh = splats["sh"][0].reshape(-1, 1, 3)         # Gaussian spherical harmonics: [N, 1, 3]
```

Where:
- `S` is the number of input views
- `H, W` are the height and width of input images
- `N` is the number of 3D Gaussians

</details>


### Inference with More Functions

For advanced usage, see `infer.py` which provides additional features:
- Save predictions: point clouds, depth maps, normals, camera parameters, and 3D Gaussian Splatting
- Visualize outputs: depth maps, surface normals, and 3D point clouds
- Render novel views using 3D Gaussians
- Export 3D Gaussian Splatting results and camera parameters to COLMAP format

## 🎯 Post 3DGS Optimization (Optional)

### Install dependencies
```shell
cd submodules/gsplat/examples
# install example requirements
pip install -r requirements.txt
# install pycolmap2 by rmbrualla
git clone https://github.com/rmbrualla/pycolmap.git
cd pycolmap
# in pyproject.toml, rename name = "pycolmap" to name = "pycolmap2"
vim pyproject.toml
# rename folder pycolmap to pycolmap2
mv pycolmap/ pycolmap2/
python3 -m pip install -e .
```
### Optimization
First, run infer.py with `--save_colmap` and `--save_gs` flags to generate COLMAP format initialization:
```shell
python infer.py --input_path /path/to/your/input --output_path /path/to/your/output --save_colmap --save_gs
```
The reconstruction result (camera parameters, 3D points, and 3D Gaussians) will be saved under `/path/to/your/output`, such as:
``` 
output/
├── images/                 # Input images
├── sparse/
│   └── 0/
│       ├── cameras.bin     # Camera intrinsics
│       ├── images.bin      # Camera poses
│       └── points3D.bin    # 3D points
└── gaussians.ply           # 3D Gaussian Splatting initialization
```
Then, run the optimization script:
```shell
python submodules/gsplat/examples/simple_trainer_worldmirror.py default --data_factor 1 --data_dir /path/to/your/inference_output --result_dir /path/to/your/gs_optimization_output
```

## 🔮 **Performance**
HunyuanWorld-Mirror achieves state-of-the-art performance across multiple 3D perception tasks, surpassing feed-forward 3D reconstruction methods. It demonstrates superior performance in **point cloud reconstruction, camera pose estimation, surface normal prediction, novel view rendering and depth estimation**. Incorporating 3D priors, such as **camera poses, depth, or intrinsics**, plays a crucial role in enhancing performance across these tasks. For point cloud reconstruction and novel view synthesis tasks, the performance is as follows:

### Point cloud reconstruction

| Method                        | 7-Scenes            |           | NRGBD             |           | DTU               |           |
|------------------------------|---------------------|-----------|-------------------|-----------|-------------------|-----------|
|                              | Acc. ⬇             | Comp. ⬇  | Acc. ⬇          | Comp. ⬇   | Acc. ⬇            | Comp. ⬇   |
| Fast3R                       | 0.096               | 0.145     | 0.135             | 0.163     | 3.340             | 2.929     |
| CUT3R                        | 0.094               | 0.101     | 0.104             | 0.079     | 4.742             | 3.400     |
| VGGT                         | 0.046               | 0.057     | 0.051             | 0.066     | 1.338             | 1.896     |
| π³                           | 0.048               | 0.072     | 0.026             | 0.028     | 1.198             | 1.849     |
| **HunyuanWorld-Mirror**      | 0.043           | 0.049 | 0.041         | 0.045 | 1.017        | 1.780 |
| **+ Intrinsics** | 0.042    | 0.048 | 0.041         | 0.045 | 0.977         | 1.762 |
| **+ Depths**     | 0.038    | 0.039 | 0.032         | 0.031 | 0.831         | 1.022 |
| **+ Camera Poses** | 0.023  | 0.036 | 0.029         | 0.032 | 0.990         | 1.847 |
| **+ All Priors** | **0.018**    | **0.023** | **0.016**         | **0.014** | **0.735**         | **0.935** |

### Novel view synthesis
| Method                          | Re10K |           |           | DL3DV    |           |           |
|--------------------------------|-------------------------|-----------|-----------|-------------------|-----------|-----------|
|                                | PSNR ⬆                 | SSIM ⬆   | LPIPS ⬇  | PSNR ⬆           | SSIM ⬆   | LPIPS ⬇  |
| FLARE                          | 16.33                  | 0.574     | 0.410     | 15.35            | 0.516     | 0.591     |
| AnySplat                       | 17.62                  | 0.616     | 0.242     | 18.31            | 0.569     | 0.258     |
| **HunyuanWorld-Mirror**                | 20.62                  | 0.706     | 0.187     | 20.92            | 0.667     | 0.203     |
| **+ Intrinsics**   | 22.03                  | 0.765     | 0.165     | 22.08            | 0.723     | 0.175     |
| **+ Camera Poses** | 20.84                  | 0.713     | 0.182     | 21.18            | 0.674     | 0.197     |
| **+ Intrinsics + Camera Poses**   | **22.30**              | **0.774** | **0.155** | **22.15**        | **0.726** | **0.174** |

### Boost of Geometric Priors
<p align="left">
  <img src="assets/num-prior.png">
</p>

For the other tasks, refer to the [technique report](https://3d-models.hunyuan.tencent.com/world/worldMirror1_0/HYWorld_Mirror_Tech_Report.pdf) for detailed performance comparisons.

## 🤖 Training

### Training Data Preparation
Please follow the [CUT3R data preparation instructions](https://github.com/CUT3R/CUT3R/blob/main/docs/preprocess.md) to download and prepare the training datasets. Currently, we provide an example dataset of Hypersim. 

### Install Dependencies
Refer to [Installation](#🛠️-dependencies-and-installation).

### Training Commands
Our model training consists of two stages. The quick start commands are:

```bash
# stage1 for prior, pointmap, camera, depth, and normal
python training/launch.py train=stage1.yaml 
# stage2 for 3dgs
python training/lanuch.py train=stage2.yaml
```

**Notes:** 
- This will automatically detect all available GPUs. To specify GPUs, use `CUDA_VISIBLE_DEVICES=<CUDA_ID>`
- If you want to resume training from a checkpoint, you can set the `ckpt_path` flag to the path of the checkpoint, such as `python training/launch.py train=stage1.yaml ckpt_path=path/to/your/checkpoint.ckpt`.
- You can comment some validation datasets in `data.validation_datasets` and adjust `wrapper.eval_modalities` in the configuration file to reduce the evaluation time.


### Customized Training Commands
We have provided a customized training config in `training/configs/train/custom.yaml`, where you can customize training parameters and model architecture. For example, you can disable certain prediction heads in the configuration file:
```yaml
wrapper:
  model:
    enable_cam: true      # Camera prediction head
    enable_pts: true      # Point cloud prediction head
    enable_depth: true    # Depth prediction head
    enable_norm: false    # Normal prediction head
    enable_gs: false      # Gaussian Splatting head
```
And run the following script:
```bash
python training/lanuch.py train=custom.yaml
```
If you want to train the model with all heads open in a single stage, you can run the following command:
```bash
python training/launch.py train=all.yaml # adjust max_images_per_gpu in train/all.yaml to avoid OOM
```

### Evaluate Training Checkpoints
After training, you can evaluate the trained checkpoints in point reconstruction by running the following script:
```bash
python training/launch.py --config-name=eval.yaml eval=pointmap.yaml wrapper.pretrained=path/to/your/checkpoint.ckpt
```

## 📊 Evaluation
### Evaluation Data Preparation
Please place all evaluation datasets in the `data` folder, or modify the configuration file at `configs/paths/default.yaml` accordingly. For data preprocessing:

- **Point map reconstruction**: We follow [Spann3r's data processing instructions](https://github.com/HengyiWang/spann3r/blob/main/docs/data_preprocess.md) to prepare DTU, 7-Scene, and NRGBD.

- **Surface normal estimation**: We follow [DSINE](https://github.com/baegwangbin/DSINE) to prepare Ibims, NYU-v2, and Scannet-Normal.

- **Novel view synthesis**: For RealEstate10K, we follow [pixelSplat's detailed instructions](https://github.com/dcharatan/pixelsplat?tab=readme-ov-file#acquiring-datasets). For DL3DV, we follow [official DL3DV repo](https://github.com/DL3DV-10K/Dataset?tab=readme-ov-file#dataset-download) and download the 480P 10K subset.

- **Depth estimation**: We follow [MonST3R's data instructions](https://github.com/Junyi42/monst3r/blob/main/data/evaluation_script.md) to prepare NYU-v2, Sintel, and KITTI.


- **Camera pose estimation**: We download RealEstate10K from [Huggingface](https://huggingface.co/datasets/mutou0308/RE10K).

### Evaluation Commands
#### 1. Point Map Reconstruction
See the config in `configs/eval/pointmap.yaml`.
```shell
python training/launch.py --config-name eval.yaml eval=pointmap.yaml
```
#### 2. Surface Normal Estimation
See the config in `configs/eval/normal.yaml`.
```shell
python training/launch.py --config-name eval.yaml eval=normal.yaml
```
#### 3. Novel View Synthesis
See the config in `configs/eval/nvs.yaml`.
```shell
python training/launch.py --config-name eval.yaml eval=nvs.yaml
```
#### 4. Depth Estimation
See the config in `configs/eval/depthmap.yaml`.
```shell
python training/launch.py --config-name eval.yaml eval=depthmap.yaml
```
#### 5. Camera Pose Estimation
See the config in `configs/eval/pose.yaml`.
```shell
python training/launch.py --config-name eval.yaml eval=pose.yaml
```

## 📑 Open-Source Plan

- [x] Inference Code
- [x] Model Checkpoints
- [x] Technical Report
- [x] Gradio Demo
- [x] Evaluation Code
- [x] Training Code


## 🔗 BibTeX

If you find HunyuanWorld-Mirror useful for your research and applications, please cite using this BibTeX:

```BibTeX
@article{liu2025worldmirror,
  title={WorldMirror: Universal 3D World Reconstruction with Any-Prior Prompting},
  author={Liu, Yifan and Min, Zhiyuan and Wang, Zhenwei and Wu, Junta and Wang, Tengfei and Yuan, Yixuan and Luo, Yawei and Guo, Chunchao},
  journal={arXiv preprint arXiv:2510.10726},
  year={2025}
}
```

## 📧 Contact
Please send emails to tengfeiwang12@gmail.com if there is any question.

## Acknowledgements
We would like to thank [HunyuanWorld](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0). We also sincerely thank the authors and contributors of [VGGT](https://github.com/facebookresearch/vggt), [Fast3R](https://github.com/facebookresearch/fast3r), [CUT3R](https://github.com/CUT3R/CUT3R), and [DUSt3R](https://github.com/naver/dust3r) for their outstanding open-source work and pioneering research.
