"""Loading driving frames, masks, depth, rigs and point clouds; saving point clouds."""
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def preload_gt_frames(gt_frames_dir, start_frame, end_frame, device,
                      mask_dir=None):
    """
    Load GT rendered frames as GPU tensors.

    Mask loading priority:
      1. If mask_dir is given, load SAM2-extracted masks from there
         (same filenames as GT frames, single-channel PNG, 0/255).
      2. Else if the GT image is RGBA, use the alpha channel.
      3. Else default to all-ones (full foreground).

    Returns:
        gt_frames:  {frame: (H, W, 3) float32 in [0,1]}
        gt_masks:   {frame: (H, W, 1) float32, 1=foreground, 0=background}
        masks_from: str — source of masks ("sam2", "rgba", "ones")
    """
    gt_frames = {}
    gt_masks = {}
    masks_from = None
    for frame in range(start_frame, end_frame + 1):
        img_name = f"frame_{frame:04d}.png"
        img_path = os.path.join(gt_frames_dir, img_name)
        img_pil = Image.open(img_path)

        # Extract RGB — composite onto white background so that background
        # pixels match PAPR's rendered background color.  Without this,
        # RGBA images with black (0,0,0) background create a huge spurious
        # loss signal in unmasked / dilated mask modes.
        if img_pil.mode == "RGBA":
            rgba = np.array(img_pil, dtype=np.float32) / 255.0
            rgba_alpha = rgba[..., 3:4]     # (H, W, 1), 1=foreground
            rgb = rgba[..., :3] * rgba_alpha + (1.0 - rgba_alpha)  # white bg
        else:
            rgb = np.array(img_pil.convert("RGB"), dtype=np.float32) / 255.0
            rgba_alpha = None

        # Load mask: SAM2 dir > RGBA alpha > all-ones
        if mask_dir is not None:
            mask_path = os.path.join(mask_dir, img_name)
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask not found: {mask_path}")
            mask_img = np.array(Image.open(mask_path))
            if mask_img.ndim == 3:
                mask_img = mask_img[:, :, 0]
            alpha = (mask_img > 0).astype(np.float32)[..., np.newaxis]
            if masks_from is None:
                masks_from = "sam2"
        elif rgba_alpha is not None:
            alpha = rgba_alpha
            if masks_from is None:
                masks_from = "rgba"
        else:
            alpha = np.ones((*rgb.shape[:2], 1), dtype=np.float32)
            if masks_from is None:
                masks_from = "ones"

        gt_frames[frame] = torch.from_numpy(rgb).to(device)
        gt_masks[frame] = torch.from_numpy(alpha).to(device)

    # Report foreground coverage
    sample_mask = next(iter(gt_masks.values()))
    fg_pct = sample_mask.mean().item() * 100
    print(f"  GT foreground coverage: {fg_pct:.1f}% of pixels "
          f"(masks from: {masks_from})")

    return gt_frames, gt_masks, masks_from


def load_delta_depth_maps(depth_dir, start_frame, end_frame, H, W, device):
    """Load per-frame metric depth maps from a directory of .npz files.

    Expects files named frame_XXXX.npz with a 'depth' key containing (H', W')
    float32 arrays.  Resizes to (H, W) if needed.

    Returns: (T, H, W) tensor on device, where T = end_frame - start_frame + 1.
    """
    maps = []
    for fi in range(start_frame, end_frame + 1):
        path = os.path.join(depth_dir, f"frame_{fi:04d}.npz")
        d = np.load(path)["depth"].astype(np.float32)  # (H', W')
        t = torch.from_numpy(d).to(device)
        if t.shape[0] != H or t.shape[1] != W:
            t = F.interpolate(
                t.unsqueeze(0).unsqueeze(0), size=(H, W),
                mode="bilinear", align_corners=False
            ).squeeze(0).squeeze(0)
        maps.append(t)
    return torch.stack(maps, dim=0)  # (T, H, W)


def load_canonical_points(path):
    """Load canonical points from .npy or .ply file, returning (N, 3) float32."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.ply':
        from plyfile import PlyData
        ply = PlyData.read(path)
        v = ply['vertex']
        return np.stack([v['x'], v['y'], v['z']], axis=-1).astype(np.float32)
    else:
        return np.load(path).astype(np.float32)


def load_autorig_with_scale_check(autorig_path, model_coord_scale=None):
    """Load autorig_data.npz and validate coordinate_scale against model.

    If autorig_data contains a stored coordinate_scale, it means joint_positions
    are in normalized space and must be multiplied by coordinate_scale to return
    to PAPR's internal coordinate space.

    When canonical_points are also in normalized space, FK+LBS operates in that
    space and the result is multiplied by model.coord_scale for rendering. This
    works correctly as long as coordinate_scale == model.coord_scale.

    Returns:
        joint_positions_np: (J, 3) float32
        parent_indices: list of int
    """
    autorig = np.load(autorig_path, allow_pickle=True)
    joint_positions_np = autorig['joint_positions'].astype(np.float32)
    parent_indices = autorig['parent_indices'].astype(np.int64).tolist()

    if "coordinate_scale" in autorig:
        stored_cs = float(autorig["coordinate_scale"])
        print(f"  autorig coordinate_scale={stored_cs} "
              f"(joint positions in normalized space, "
              f"multiply by {stored_cs} to get PAPR scale)")
        if model_coord_scale is not None and abs(stored_cs - model_coord_scale) > 1e-6:
            print(f"  WARNING: autorig coordinate_scale ({stored_cs}) != "
                  f"model.coord_scale ({model_coord_scale}). "
                  f"Joint positions and canonical points may be in different spaces!")

    return joint_positions_np, parent_indices


def save_points_ply(points, out_path):
    """Save point cloud (N,3) as an ASCII PLY file."""
    points = np.asarray(points, dtype=np.float32)
    with open(out_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for x, y, z in points:
            f.write(f"{x:.8f} {y:.8f} {z:.8f}\n")
