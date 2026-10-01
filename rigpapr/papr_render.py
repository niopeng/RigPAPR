"""Load a trained PAPR model and render it with a given (deformed) point cloud."""
import copy
import json
import math
import os

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

from utils import DictAsMember, update_dict
from dataset import get_dataset
from models import get_model


def load_model_and_dataset(config_path, checkpoint_path):
    """Load PAPR model and dataset following test.py pattern."""
    # Load default config
    project_root = os.path.join(os.path.dirname(__file__), '..')
    default_config_path = os.path.join(project_root, "configs", "default_fast.yml")
    with open(default_config_path, 'r') as f:
        default_config = yaml.safe_load(f)

    # Load experiment config and merge
    with open(config_path, 'r') as f:
        exp_config = yaml.safe_load(f)

    # Handle bkg_sphere_radius backward compat (from test.py)
    if 'bkg_sphere_radius' not in default_config['geoms']['points']:
        default_config['geoms']['points']['bkg_sphere_radius'] = default_config['geoms']['points']['bkg_points_init_scale'][0]
    default_config['dataset']['bkg_sphere_radius'] = default_config['geoms']['points']['bkg_sphere_radius']

    config = copy.deepcopy(default_config)
    update_dict(config, exp_config)
    args = DictAsMember(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model
    model = get_model(args, device)

    # Load checkpoint (test.py lines 570-580)
    model_state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
    if len(list(model_state_dict.keys())) == 1:
        resume_step = int(list(model_state_dict.keys())[0])
        model.load_my_state_dict(model_state_dict[str(resume_step)])
    else:
        model.load_my_state_dict(model_state_dict)
        resume_step = 250000  # default

    model = model.to(device)
    model.eval()

    # Load dataset for camera info
    dataset = get_dataset(args.dataset, mode="test")

    # Set camera intrinsics on model (test.py lines 434-439)
    model.fx = dataset.focal_x
    model.fy = dataset.focal_y
    model.cx = dataset.cx
    model.cy = dataset.cy
    model.H = dataset.H
    model.W = dataset.W

    return model, dataset, args, device, resume_step


def render_single_frame(model, device, args, rayo, rayd, rayd_no_norm, c2w, pix_coords,
                        deformed_points, resume_step):
    """
    Render a single frame with deformed points, following finetune_with_animate.py pattern.

    Args:
        deformed_points: torch.Tensor (N, 3) in scaled NeRF space (already * coord_scale)
    """
    N = 1
    H = model.H
    W = model.W

    final_select_k = model.select_k
    if args.geoms.points.select_k_rnd:
        final_select_k = max(final_select_k, args.geoms.points.select_k_max)
    if args.models.attn.select_prop_scores:
        final_select_k = min(final_select_k, args.models.attn.select_prop_scores_topk)
    if args.models.attn.select_v_scores:
        final_select_k = min(final_select_k, args.models.attn.select_v_scores_topk)
        if args.models.attn.select_v_scores_rnd:
            final_select_k = max(final_select_k, args.models.attn.select_v_scores_topk_max)

    bkg_seq_len_attn = 0
    if hasattr(model, 'args') and model.args.geoms.no_additional_bkg:
        bkg_seq_len_attn = 0
    elif model.bkg_feats is not None and not model.args.geoms.points.use_bkg_points and not model.args.models.attn.append_bkg_points:
        bkg_seq_len_attn = model.bkg_feats.shape[0]

    append_bkg_points_size = 1 if (model.args.models.attn.append_bkg_points and not model.args.geoms.no_additional_bkg) else 0

    feat_map_dim = model.feat_map_dim
    feature_map = torch.zeros(N, H, W, 1, feat_map_dim).to(device)
    attn = torch.zeros(N, H, W, final_select_k + append_bkg_points_size + bkg_seq_len_attn, 1).to(device)
    hit_pred = torch.empty(N, H, W).to(device)

    max_height = args.test.max_height if hasattr(args.test, 'max_height') else 100
    max_width = args.test.max_width if hasattr(args.test, 'max_width') else 100

    with torch.no_grad():
        for height_start in range(0, H, max_height):
            for width_start in range(0, W, max_width):
                height_end = min(height_start + max_height, H)
                width_end = min(width_start + max_width, W)

                feature_map[:, height_start:height_end, width_start:width_end, :, :], \
                attn[:, height_start:height_end, width_start:width_end, :, :], \
                _, cur_hit_pred = model.evaluate(
                    rayo,
                    rayd[:, height_start:height_end, width_start:width_end],
                    c2w,
                    pix_coords[:, height_start:height_end, width_start:width_end],
                    step=resume_step,
                    deformed_points=deformed_points,
                    rays_d_no_norm=rayd_no_norm[:, height_start:height_end, width_start:width_end]
                )

                if cur_hit_pred is not None:
                    hit_pred[:, height_start:height_end, width_start:width_end] = cur_hit_pred

        # Apply UNet if used
        if args.models.unet.use:
            if model.args.models.unet.double_channel:
                feature_map = torch.cat([feature_map, feature_map], dim=-1)
            foreground_rgb = model.unet(feature_map.squeeze(-2).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).unsqueeze(-2)
        else:
            foreground_rgb = feature_map

        # Background compositing (following test.py lines 167-238)
        if model.args.geoms.no_additional_bkg:
            rgb = foreground_rgb.squeeze(-2)
        elif model.bkg_feats is not None and not model.args.geoms.points.use_bkg_points and not model.args.models.attn.append_bkg_points:
            bkg_attn = attn[..., -1:, :]
            if args.geoms.background.use_dumb_constant:
                bkg_attn = (bkg_attn * model.dumb_constant).clamp(0, 1)
            bkg_attn = model.bkg_attn_act((bkg_attn + args.models.bkg_attn_shift) * args.models.bkg_attn_scale)

            if args.mask_bkg_rays_thresh > 0:
                min_d2r = torch.zeros(N, H, W).to(device)
                for hs in range(0, H, max_height):
                    for ws in range(0, W, max_width):
                        he = min(hs + max_height, H)
                        we = min(ws + max_width, W)
                        min_d2r[:, hs:he, ws:we] = model.proximity_attn.d2r.min(dim=-2)[0].squeeze(-1)
                fg_mask = min_d2r[..., None] < args.mask_bkg_rays_thresh
                bkg_attn_masked = torch.ones_like(bkg_attn)
                bkg_attn_masked[fg_mask] = bkg_attn[fg_mask]
            else:
                bkg_attn_masked = bkg_attn

            if model.args.models.bkg_attn_deno_thresh > 0:
                bkg_deno = bkg_attn_masked.clone().detach()
                bkg_deno[bkg_deno < model.args.models.bkg_attn_deno_thresh] = 1
                bkg_attn_masked = bkg_attn_masked / bkg_deno
            if model.args.models.bkg_attn_sigmoid_scale > 0:
                bkg_attn_masked = torch.sigmoid(bkg_attn_masked * model.args.models.bkg_attn_sigmoid_scale)

            bkg_color = model.bkg_feats * model.bkg_scaler.squeeze(0)
            cubemap_bkg_color = model.get_cubemap_background_color(rayo, rayd, cur_step=-1)
            if cubemap_bkg_color is not None:
                rgb = foreground_rgb * (1 - bkg_attn_masked) + cubemap_bkg_color.unsqueeze(-2) * bkg_attn_masked
            else:
                rgb = foreground_rgb * (1 - bkg_attn_masked) + bkg_color.expand(N, H, W, -1, -1) * bkg_attn_masked
            rgb = rgb.squeeze(-2)
        else:
            rgb = foreground_rgb.squeeze(-2)
            if model.bkg_feats is not None:
                bkg_color = model.bkg_feats * model.bkg_scaler.squeeze(0)
                cubemap_bkg_color = model.get_cubemap_background_color(rayo, rayd, cur_step=-1)
                effective_bkg_color = cubemap_bkg_color if cubemap_bkg_color is not None else bkg_color.expand(N, H, W, -1)
                if model.args.geoms.points.use_bkg_points and model.args.models.attn.append_bkg_points:
                    bkg_attn = attn[..., -1, :]
                    if model.args.geoms.points.bkg_points_alpha_blend or model.args.models.attn.append_bkg_points_alpha_blend:
                        rgb = rgb * (1 - bkg_attn) + effective_bkg_color * bkg_attn
                elif model.args.models.attn.append_bkg_points:
                    bkg_attn = attn[..., -1, :]
                    if model.args.models.attn.append_bkg_points_alpha_blend:
                        rgb = rgb * (1 - bkg_attn) + effective_bkg_color * bkg_attn

        rgb = model.last_act(rgb)
        rgb = torch.clamp(rgb, 0, 1)

    return rgb


def setup_view_from_cameras_json(cameras_json_path, camera_idx, dataset,
                                 device, H, W, coord_scale):
    """Load a specific camera from cameras.json and build ray data.

    Returns:
        rayo, rayd, rayd_no_norm, c2w, pix_coords (all torch tensors on device)
    """
    from dataset.utils import get_rays

    with open(cameras_json_path, 'r') as f:
        cam_data = json.load(f)

    camera_angle_x = cam_data["camera_angle_x"]
    frames = cam_data["frames"]
    if camera_idx >= len(frames):
        raise ValueError(f"camera_idx={camera_idx} out of range "
                         f"(cameras.json has {len(frames)} frames)")

    frame = frames[camera_idx]
    c2w_np = np.array(frame["transform_matrix"], dtype=np.float64)

    scale_mat = np.diag([coord_scale, coord_scale, coord_scale, 1.0])
    c2w_scaled = (scale_mat @ c2w_np).astype(np.float32)
    c2w_t = torch.tensor(c2w_scaled).unsqueeze(0)

    focal = 0.5 * W / math.tan(0.5 * camera_angle_x)
    fx = fy = focal

    rayo, rayd, _, rayd_no_norm = get_rays(H, W, fx, fy, c2w_t)
    rayo = rayo.to(device)
    rayd = rayd.to(device)
    rayd_no_norm = rayd_no_norm.to(device)
    c2w_t = c2w_t.to(device)

    i_coords = torch.arange(W, dtype=torch.float32) + 0.5
    j_coords = torch.arange(H, dtype=torch.float32) + 0.5
    ii, jj = torch.meshgrid(i_coords, j_coords, indexing='xy')
    pix_coords = torch.stack([ii, jj], dim=-1).to(device).unsqueeze(0)

    return rayo, rayd, rayd_no_norm, c2w_t, pix_coords


def render_comparison(deformed_scaled, frame_idx, step, gt_frames, gt_masks,
                      model, args_model, device, resume_step,
                      full_rayo, full_rayd, full_rayd_no_norm,
                      full_pix_coords, c2w, eval_dir, diff_amplify=5.0):
    """Render full frame and save [GT | Rendered | Diff] comparison PNG."""
    with torch.no_grad():
        rgb = render_single_frame(
            model, device, args_model, full_rayo, full_rayd,
            full_rayd_no_norm, c2w, full_pix_coords,
            deformed_scaled, resume_step)

    rendered_np = rgb.squeeze(0).cpu().numpy()
    gt_np = gt_frames[frame_idx].cpu().numpy()
    gt_fg = gt_masks[frame_idx].cpu().numpy()  # (H, W, 1)
    rendered_fg = (rendered_np.mean(axis=-1, keepdims=True) < 0.98).astype(np.float32)
    either_fg = np.clip(gt_fg + rendered_fg, 0, 1)  # union of silhouettes
    diff = np.clip(np.abs(rendered_np - gt_np) * diff_amplify, 0, 1) * either_fg

    # Full-image PSNR
    mse = ((rendered_np - gt_np) ** 2).mean()
    psnr_val = -10 * np.log10(mse) if mse > 1e-10 else 100.0

    comparison = np.concatenate([gt_np, rendered_np, diff], axis=1)
    comparison_u8 = (comparison * 255).astype(np.uint8)
    comp_img = Image.fromarray(comparison_u8)

    # Draw PSNR label with dark background box for visibility on white BG
    H_img, W_panel = gt_np.shape[0], gt_np.shape[1]
    draw = ImageDraw.Draw(comp_img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except (IOError, OSError):
        font = ImageFont.load_default()
    labels = [
        (f"GT  |  Frame {frame_idx}", 0),
        (f"Rendered  PSNR={psnr_val:.2f}dB", W_panel),
        (f"|Diff| x{diff_amplify:.0f}", W_panel * 2),
    ]
    pad = 4
    for text, x_offset in labels:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = x_offset + (W_panel - tw) // 2
        y = 4
        draw.rectangle(
            [x - pad, y - pad, x + tw + pad, y + th + pad],
            fill=(0, 0, 0, 200))
        draw.text((x, y), text, fill=(255, 255, 255), font=font)

    out_path = os.path.join(eval_dir, f"frame_{frame_idx:04d}_step_{step:05d}.png")
    comp_img.save(out_path)
    print(f"      [eval] saved comparison: {out_path}  PSNR={psnr_val:.2f}dB")
