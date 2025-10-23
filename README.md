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
* **[Oct 22, 2025]**: We release the inference code and model weights. [Download](https://huggingface.co/tencent/HunyuanWorld-Mirror).

> Join our **[Wechat](#)** and **[Discord](https://discord.gg/dNBrdrGGMa)** group to discuss and find help from us.

| Wechat Group                                     | Xiaohongshu                                           | X                                           | Discord                                           |
|--------------------------------------------------|-------------------------------------------------------|---------------------------------------------|---------------------------------------------------|
| <img src="assets/qrcode/wechat.png"  height=140> | <img src="assets/qrcode/xiaohongshu.png"  height=140> | <img src="assets/qrcode/x.png"  height=140> | <img src="assets/qrcode/discord.png"  height=140> | 


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

# 3. Install PyTorch and other dependencies using conda
# For CUDA 12.4
conda install pytorch=2.4.0 torchvision pytorch-cuda=12.4 nvidia/label/cuda-12.4.0::cuda-toolkit -c pytorch -c nvidia -y

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
pip install -r requiremens_demo.txt
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
import numpy as np
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
    'camera_pose': None,      # Camera pose tensor [1, N, 4, 4]
    'depthmap': None,         # Depth map tensor [1, N, H, W]
    'camera_intrinsics': None # Camera intrinsics tensor [1, N, 3, 3]
}
for idx, (key, data) in enumerate(prior_data.items()):
    if data is not None:
        cond_flags[idx] = 1

# --- Inference ---
with torch.no_grad():
    predictions = model(views=inputs, cond_flags=cond_flags)
```

<details>
<summary>Click to view output format</summary>

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
colors = (splats["sh"][0] if "sh" in splats else splats["colors"][0]).reshape(-1, 3)  # Gaussian colors: [N, 3]
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

## 📑 Open-Source Plan

- [x] Inference Code
- [x] Model Checkpoints
- [x] Technical Report
- [x] Gradio Demo
- [ ] Evaluation Code
- [ ] Training Code


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
