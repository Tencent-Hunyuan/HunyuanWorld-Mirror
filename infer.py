import argparse
import glob
from pathlib import Path
import os
import time

import numpy as np
import torch
from PIL import Image

from src.models.models.worldmirror import WorldMirror
from src.utils.inference_utils import prepare_images_to_tensor
from src.utils.video_utils import video_to_image_frames
from src.models.utils.geometry import depth_to_world_coords_points
from src.models.utils.geometry import create_pixel_coordinate_grid

from src.utils.save_utils import save_depth_png, save_depth_npy, save_normal_png
from src.utils.save_utils import save_scene_ply, save_gs_ply, save_points_ply
from src.utils.render_utils import render_interpolated_video

from src.utils.build_pycolmap_recon import build_pycolmap_reconstruction
from src.models.utils.camera_utils import vector_to_camera_matrices


def create_confidence_mask(confidence: torch.Tensor,
                          conf_threshold_percent: float = 30.0) -> torch.Tensor:
    """
    Create confidence mask for filtering points based on confidence threshold.
    Discard bottom p% confidence points, keep top (100-p)%.
    
    Args:
        confidence: Confidence scores (any shape)
        conf_threshold_percent: Percentage of low-confidence points to filter out (0-100)
    
    Returns:
        Boolean mask for filtering points
    """
    # Flatten confidence scores
    conf_flat = confidence.flatten()
    # Mask out extremely small/invalid values
    conf_flat = conf_flat.masked_fill(conf_flat <= 1e-5, float("-inf"))
    
    N = conf_flat.numel()
    
    # Discard bottom p%, keep top (100-p)%
    if conf_threshold_percent > 0:
        keep_from_percent = int(np.ceil(N * (100.0 - conf_threshold_percent) / 100.0))
    else:
        keep_from_percent = N
    K = max(1, keep_from_percent)
    
    # Select top-K indices (deterministic, no randomness)
    topk_idx = torch.topk(conf_flat, K, largest=True, sorted=False).indices
    
    # Create mask
    conf_mask = torch.zeros_like(conf_flat, dtype=torch.bool)
    conf_mask[topk_idx] = True
    
    return conf_mask


def main():
    parser = argparse.ArgumentParser(description="HunyuanWorld-Mirror inference")
    parser.add_argument("--input_path", type=str, default="examples/realistic/Ireland_Landscape", help="Input can be: a directory of images; a single video file; or a directory containing multiple video files (.mp4/.avi/.mov/.webm/.gif). If directory has multiple videos, frames from all clips are extracted (using --fps) and merged in filename order.")
    parser.add_argument("--output_path", type=str, default="inference_output")
    parser.add_argument("--conf_threshold", type=float, default=0.0, help="Confidence threshold percentage for filtering points (0-100)")
    parser.add_argument("--fps", type=int, default=1, help="Frames per second for video extraction")
    parser.add_argument("--target_size", type=int, default=518, help="Target size for image resizing")
    parser.add_argument("--write_txt", action="store_true", help="Also write human-readable COLMAP txt (slow, huge)")
    # Save flags
    parser.add_argument("--save_pointmap", action="store_true", default=True, help="Save points PLY")
    parser.add_argument("--save_depth", action="store_true", default=True, help="Save depth PNG")
    parser.add_argument("--save_normal", action="store_true", default=True, help="Save normal PNG")
    parser.add_argument("--save_gs", action="store_true", default=True, help="Save Gaussians PLY")
    parser.add_argument("--save_rendered", action="store_true", default=True, help="Save rendered video")
    parser.add_argument("--save_colmap", action="store_true", default=True, help="Save COLMAP sparse")
    # Conditioning flags
    parser.add_argument("--cond_pose", action="store_true", help="Use camera pose conditioning if available")
    parser.add_argument("--cond_intrinsics", action="store_true", help="Use intrinsics conditioning if available")
    parser.add_argument("--cond_depth", action="store_true", help="Use depth conditioning if available")
    args = parser.parse_args()

    # Print inference parameters
    print(f"🔧 Configuration:")
    print(f"  - Confidence threshold: {args.conf_threshold}%")
    print(f"  - FPS: {args.fps}")
    print(f"  - Target size: {args.target_size}px")
    print(f"  - Conditioning:")
    print(f"    - Pose: {'✅' if args.cond_pose else '❌'}")
    print(f"    - Intrinsics: {'✅' if args.cond_intrinsics else '❌'}")
    print(f"    - Depth: {'✅' if args.cond_depth else '❌'}")

    # 1) Init model - This requires internet access or the huggingface hub cache to be pre-downloaded
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WorldMirror.from_pretrained("tencent/HunyuanWorld-Mirror").to(device)
    model.eval()
    
    input_path = Path(args.input_path)
    
    # Create output directory with filename-based subdirectory
    if input_path.is_file():
        subdir_name = input_path.stem
    elif input_path.is_dir():
        subdir_name = input_path.name
    else:
        raise ValueError(f"❌ Invalid input path: {input_path} (must be directory or video file)")
    
    outdir = Path(args.output_path) / subdir_name
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Determine input type and get image paths
    video_exts = ['.mp4', '.avi', '.mov', '.webm', '.gif']
    
    if input_path.is_file() and input_path.suffix.lower() in video_exts:
        # Case 1: Single video file - extract frames
        print(f"📹 Processing video: {input_path}")
        input_frames_dir = outdir / "input_frames"
        input_frames_dir.mkdir(exist_ok=True)
        
        img_paths = video_to_image_frames(str(input_path), str(input_frames_dir), fps=args.fps)
        if not img_paths:
            raise RuntimeError("❌ Failed to extract frames from video")

        img_paths = sorted(img_paths)
        print(f"✅ Extracted {len(img_paths)} frames to {input_frames_dir}")
    
    elif input_path.is_dir():
        # Case 2: Directory of images
        print(f"📁 Processing directory: {input_path}")
        img_paths = []
        for ext in ["*.jpeg", "*.jpg", "*.png", "*.webp"]:
            img_paths.extend(glob.glob(os.path.join(str(input_path), ext)))
        if len(img_paths) == 0:
            raise FileNotFoundError(f"❌ No image files found in {input_path}")
        img_paths = sorted(img_paths)
        print(f"✅ Loaded {len(img_paths)} images from {input_path}")

    else:
        raise ValueError(f"❌ Invalid input path: {input_path}")

    # 3) Load and preprocess images
    views = {}
    imgs = prepare_images_to_tensor(img_paths, target_size=args.target_size, resize_strategy="crop").to(device)  # [1,S,3,H,W], in [0,1]
    views["img"] = imgs
    B, S, C, H, W = imgs.shape
    cond_flags = [0, 0, 0]
    print(f"📸 Loaded {S} images with shape {imgs.shape}")

    # 4) Inference
    print("\n🚀 Starting inference pipeline...")
    start_time = time.time()
    use_amp = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if use_amp:
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float32
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=bool(use_amp), dtype=amp_dtype):
            predictions = model(views=views, cond_flags=cond_flags)  # Multi-modal inference with priors
    print(f"🕒 Inference time: {time.time() - start_time:.3f} seconds")
    
    # 5) Save results
    print("\n📤 Saving results...")
    images_dir = outdir / "images" # original resolution images
    images_dir.mkdir(exist_ok=True)
    images_resized_dir = outdir / "images_resized" # resized images
    images_resized_dir.mkdir(exist_ok=True)
    if args.save_depth:
        depth_dir = outdir / "depth"
        depth_dir.mkdir(exist_ok=True)
    if args.save_normal:
        normal_dir = outdir / "normal"
        normal_dir.mkdir(exist_ok=True)
    if args.save_colmap:
        sparse_dir = outdir / "sparse" / "0"
        sparse_dir.mkdir(parents=True, exist_ok=True)
    
    # save images
    processed_image_names = []
    for i in range(S):
        im = (imgs[0, i].permute(1, 2, 0).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        fname = f"image_{i+1:04d}.png"
        Image.fromarray(im).save(str(images_resized_dir / fname))
        pil_img = Image.open(img_paths[i]).convert("RGB")
        processed_height, processed_width = imgs[0, i].shape[1], imgs[0, i].shape[2]
        processed_aspect_ratio = processed_width / processed_height
        orig_width, orig_height = pil_img.size
        new_height = int(orig_width / processed_aspect_ratio)
        new_width = orig_width
        pil_img = pil_img.resize((orig_width, new_height), Image.Resampling.BICUBIC)
        pil_img.save(str(images_dir / fname))

        processed_image_names.append(fname)
        
    # save pointmap
    if "pts3d" in predictions and args.save_pointmap:
        pts_list = []
        pts_colors_list = []
        pts_conf_list = []
        
        for i in range(S):
            pts = predictions["pts3d"][0, i]  # [H,W,3]
            pts_conf = predictions["pts3d_conf"][0, i]  # [H,W]
            img_colors = imgs[0, i].permute(1, 2, 0)  # [H, W, 3]
            img_colors = (img_colors * 255).to(torch.uint8)
            
            pts_list.append(pts.reshape(-1, 3))
            pts_colors_list.append(img_colors.reshape(-1, 3))
            pts_conf_list.append(pts_conf.reshape(-1))

        all_pts = torch.cat(pts_list, dim=0)
        all_colors = torch.cat(pts_colors_list, dim=0)
        all_conf = torch.cat(pts_conf_list, dim=0)
        # Apply filtering using mask (optional)
        conf_mask = create_confidence_mask(
            all_conf,
            conf_threshold_percent=args.conf_threshold,
        )
        filtered_pts = all_pts[conf_mask]
        filtered_colors = all_colors[conf_mask]
        
        save_scene_ply(outdir / "pts.ply", filtered_pts, filtered_colors)
        print(f"  - Saved {len(filtered_pts)} points to {outdir / 'pts.ply'}")

    # save depthmap
    if "depth" in predictions and args.save_depth:
        for i in range(S):
            # Save both PNG (for visualization) and NPY (for actual depth values)
            save_depth_png(depth_dir / f"depth_{i:04d}.png", predictions["depth"][0, i, :, :, 0])
            save_depth_npy(depth_dir / f"depth_{i:04d}.npy", predictions["depth"][0, i, :, :, 0])
        print(f"  - Saved {S} depth maps to {depth_dir} (both PNG and NPY formats)")

    # save normalmap
    if "normals" in predictions and args.save_normal:
        for i in range(S):
            save_normal_png(normal_dir / f"normal_{i:04d}.png", predictions["normals"][0, i])
        print(f"  - Saved {S} normal maps to {normal_dir}")

    # Save Gaussians PLY and render video
    if "splats" in predictions and args.save_gs:
        # Get Gaussian parameters (already filtered by GaussianSplatRenderer)
        means = predictions["splats"]["means"][0].reshape(-1, 3)
        scales = predictions["splats"]["scales"][0].reshape(-1, 3)
        quats = predictions["splats"]["quats"][0].reshape(-1, 4)
        colors = (predictions["splats"]["sh"][0] if "sh" in predictions["splats"] else predictions["splats"]["colors"][0]).reshape(-1, 3)
        opacities = predictions["splats"]["opacities"][0].reshape(-1)
        
        # Save Gaussian PLY
        ply_path = outdir / "gaussians.ply"
        save_gs_ply(
            ply_path,
            means,
            scales,
            quats,
            colors,
            opacities,
        )

        # Render video using the same filtered splats from predictions
        num_views = S
        if args.save_rendered:
            e4x4 = predictions['camera_poses']
            k3x3 = predictions['camera_intrs']
            render_interpolated_video(model.gs_renderer, predictions["splats"], e4x4, k3x3, (H, W), outdir / "rendered", interp_per_pair=15, loop_reverse=num_views==1)
            print(f"  - Saved rendered.mp4 to {outdir}")
        else:
            print(f"⚠️  Not set --save_rendered flag, skipping video rendering")

    # Build and export COLMAP reconstruction (images + sparse)
    if args.save_colmap:
        final_width, final_height = new_width, new_height
        print(f"colmap_width: {final_width}, colmap_height: {final_height}")
        
        # Prepare extrinsics/intrinsics (camera-from-world) using resized image size
        e3x4, intr = vector_to_camera_matrices(predictions["camera_params"], image_hw=(final_height, final_width))
        _, intr_resize = vector_to_camera_matrices(predictions["camera_params"], image_hw=(H, W))
        extrinsics = e3x4[0] # [S,3,4]
        intrinsics = intr[0] # [S,3,3]
        intrinsics_resize = intr_resize[0] # [S,3,3]
                
        points_list = []
        colors_list = []
        conf_list = []
        xyf_list = []

        # Precompute pixel coordinate grid (XYF) like demo_colmap
        xyf_grid = create_pixel_coordinate_grid(num_frames=S, height=H, width=W)  # [S,H,W,3] float32
        xyf_grid = xyf_grid.astype(np.int32)
        
        # Calculate scaling factors to map from processed to resized coordinates
        scale_x = final_width / W
        scale_y = final_height / H

        # Use the SAME coordinate transformation as GaussianSplatRenderer.prepare_splats
        # to ensure consistency between Gaussian PLY and depth-based sparse points
        for i in range(S):
            d = predictions["depth"][0, i, :, :, 0]
            d_conf = predictions["depth_conf"][0, i, :, :]
            w2c = extrinsics[i][:3, :4]  # [3, 4] camera-to-world
            w2c = torch.cat([w2c, torch.tensor([[0, 0, 0, 1]], device=w2c.device)], dim=0)  # [4,4]
            c2w = torch.linalg.inv(w2c)[:3, :4]  # [4,4]
            K = intrinsics_resize[i]
            pts_i, _, mask = depth_to_world_coords_points(d[None], c2w[None], K[None])

            img_colors = (imgs[0, i].permute(1, 2, 0) * 255).to(torch.uint8)
            valid = mask[0]
            if valid.sum().item() == 0:
                continue
            xyf_np = xyf_grid[i][valid.cpu().numpy()]  # [N,3] int32
            xyf_list.append(torch.from_numpy(xyf_np).to(valid.device))
            points_list.append(pts_i[0][valid])
            colors_list.append(img_colors[valid])
            conf_list.append(d_conf[valid])

        all_pts = torch.cat(points_list, dim=0)
        all_cols = torch.cat(colors_list, dim=0)
        all_conf = torch.cat(conf_list, dim=0)
        all_xyf = torch.cat(xyf_list, dim=0)

        # Global confidence filtering
        conf_mask = create_confidence_mask(
            all_conf,
            conf_threshold_percent=args.conf_threshold,
        )

        # Convert to numpy
        extrinsics = extrinsics.detach().cpu().numpy()
        intrinsics = intrinsics.detach().cpu().numpy()
        f_pts = all_pts[conf_mask].detach().cpu().to(torch.float32).numpy()
        f_cols = all_cols[conf_mask].detach().cpu().to(torch.uint8).numpy()
        f_xyf = all_xyf[conf_mask].detach().cpu().to(torch.int32).numpy()
        
        # Scale 2D coordinates from processed image to resized image resolution (if still valid)
        f_xyf[:, 0] = (f_xyf[:, 0] * scale_x).astype(np.int32)  # x coordinates
        f_xyf[:, 1] = (f_xyf[:, 1] * scale_y).astype(np.int32)  # y coordinates

        # Build reconstruction using pycolmap (PINHOLE) with resized image size
        # Standard COLMAP reconstruction with 2D-3D correspondences
        image_size = np.array([final_width, final_height])
        reconstruction = build_pycolmap_reconstruction(
            points=f_pts,
            pixel_coords=f_xyf,
            point_colors=f_cols,
            poses=extrinsics,
            intrinsics=intrinsics,
            image_size=image_size,
            shared_camera_model=False,
            camera_model="SIMPLE_PINHOLE",
        )

        # Update image names to match saved files
        for pyimageid in reconstruction.images:
            reconstruction.images[pyimageid].name = processed_image_names[pyimageid - 1]

        # Write BIN
        reconstruction.write(str(sparse_dir))
        # Save points3D.ply
        save_points_ply(sparse_dir / "points3D.ply", f_pts, f_cols)
        
        print(f"  - Saved COLMAP BIN and points3D.ply to {sparse_dir}")
        
        
if __name__ == "__main__":
    main()


