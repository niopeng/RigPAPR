"""
Fit a skeletal rig to a driving video through PAPR's differentiable renderer.

The canonical PAPR point cloud is deformed with linear blend skinning (LBS);
joint rotations, root translation and skinning weights are optimized so that
renders of the deformed points match the driving frames.

  Phase 1 - per-frame pose sweep:
    Frames are processed outward from the canonical frame, each warm-started
    from its neighbour. Only joint rotations + root translation are optimized;
    skinning weights stay at their initialization.

  Phase 2 - joint refinement over all frames:
    Phase 1 poses are the frozen base. A small time-conditioned MLP predicts
    per-joint pose corrections, optimized jointly with the skinning weights and
    root translations. With --refine_start S, at step S the corrections are baked
    into the base poses, the MLP and optimizer restart, root translation is
    frozen, and the tracking and delta-depth losses are switched off.

Usage (see scripts/run_trex.sh for the full T-Rex settings):
    python optimize.py --phase 1 \
      --config <config.yml> --checkpoint <model.pth> \
      --canonical_points <canonical.npy> \
      --autorig_data <autorig_data.npz> --init_weights <skinning_weights.npz> \
      --gt_frames_dir <frames/> --mask_dir <masks/> \
      --canonical_frame 40 --start_frame 40 --end_frame 0 \
      --output_dir output/<scene>/phase1

    python optimize.py --phase 2 <same inputs> \
      --phase1_dir output/<scene>/phase1 --total_steps 50000 --refine_start 30000 \
      --output_dir output/<scene>/phase2
"""
import os
import sys
import argparse
import json
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.utils import fused_projection
from models.loss import ssim as compute_ssim
from rigpapr.data import (
    preload_gt_frames, load_delta_depth_maps, load_canonical_points,
    load_autorig_with_scale_check, save_points_ply)
from rigpapr.lbs import (
    aa_to_matrix4, topological_sort, forward_kinematics,
    compute_joint_depth_and_descendants, apply_lbs_torch, build_knn)
from rigpapr.losses import (
    smoothness_loss, sparsity_loss, temporal_smoothness,
    compute_distance_preservation_loss, compute_delta_depth_loss)
from rigpapr.papr_render import (
    load_model_and_dataset, render_single_frame, setup_view_from_cameras_json,
    render_comparison)


def rotation_reg_weights(parent_indices, device):
    """Per-joint rotation-regularization weights: 1 + number of descendants,
    normalized to mean 1, so joints that move more of the body cost more."""
    _, n_desc = compute_joint_depth_and_descendants(parent_indices)
    w = torch.tensor([1.0 + n_desc[j] for j in range(len(parent_indices))],
                     dtype=torch.float32, device=device)
    return w / w.mean()


class CorrectionMLP(torch.nn.Module):
    """Small MLP that maps a normalized time step to per-joint axis-angle corrections.

    Architecture:
        PE(t) -> Linear(d_in, hidden) -> ReLU -> Linear(hidden, hidden) -> ReLU
              -> Linear(hidden, num_joints*3)

    The last layer is zero-initialized so the MLP outputs identity (zero
    correction) at the start of training.  Positional encoding uses sinusoidal
    features: [t, sin(2^0 pi t), cos(2^0 pi t), ..., sin(2^{L-1} pi t), cos(2^{L-1} pi t)].
    """

    def __init__(self, num_joints, num_freqs=10, hidden_dim=128):
        super().__init__()
        self.num_joints = num_joints
        self.num_freqs = num_freqs
        d_in = 1 + 2 * num_freqs  # raw t + sin/cos pairs

        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, num_joints * 3),
        )
        # Zero-init last layer so initial output is zero (identity rotation)
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

        # Precompute frequency bands: 2^0, 2^1, ..., 2^{L-1}
        freqs = torch.pow(2.0, torch.arange(num_freqs, dtype=torch.float32))
        self.register_buffer("freqs", freqs * torch.pi)

    def positional_encoding(self, t):
        """t: scalar or (B,) tensor in [0, 1].  Returns (B, 1+2L)."""
        if t.dim() == 0:
            t = t.unsqueeze(0)
        t = t.unsqueeze(-1)  # (B, 1)
        phase = t * self.freqs.unsqueeze(0)  # (B, L)
        return torch.cat([t, phase.sin(), phase.cos()], dim=-1)  # (B, 1+2L)

    def forward(self, t_normalized):
        """t_normalized: scalar or (B,) tensor in [0,1].
        Returns: (num_joints, 3) if scalar input, else (B, num_joints, 3)."""
        pe = self.positional_encoding(t_normalized)  # (B, d_in)
        out = self.net(pe)  # (B, num_joints*3)
        out = out.view(-1, self.num_joints, 3)  # (B, num_joints, 3)
        if t_normalized.dim() == 0:
            return out.squeeze(0)  # (num_joints, 3)
        return out


def bake_corrections(base_rot_mats, corrections):
    """Compose per-frame corrections into the base rotations, in place.

    Args:
        base_rot_mats: (T, J, 4, 4) base rotation matrices, updated in place.
        corrections:   (T, J, 3) axis-angle corrections.

    Returns:
        (T, J, 3) axis-angle of the baked base rotations (for checkpoints).
    """
    T, J = corrections.shape[:2]
    for t in range(T):
        base_rot_mats[t] = torch.einsum(
            'bij,bjk->bik', base_rot_mats[t], aa_to_matrix4(corrections[t]))
    baked_aa = torch.zeros(T, J, 3, device=corrections.device, dtype=torch.float32)
    for t in range(T):
        for j in range(J):
            R = base_rot_mats[t, j, :3, :3]
            trace = R[0, 0] + R[1, 1] + R[2, 2]
            cos_theta = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
            theta = torch.acos(cos_theta)
            if theta < 1e-6:
                continue
            axis = torch.stack([
                R[2, 1] - R[1, 2],
                R[0, 2] - R[2, 0],
                R[1, 0] - R[0, 1],
            ]) / (2.0 * torch.sin(theta))
            baked_aa[t, j] = axis * theta
    return baked_aa


# ═══════════════════════════════════════════════════════════════════════════════
# Args
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="LBS-parameterized optimization via differentiable rendering")

    # Phase
    p.add_argument("--phase", type=int, required=True, choices=[1, 2],
                   help="1 = per-frame pose sweep, 2 = joint refinement over all frames")

    # Model / data
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--canonical_points", type=str, required=True)
    p.add_argument("--autorig_data", type=str, required=True,
                   help="Path to autorig_data.npz (joint_positions, parent_indices)")
    p.add_argument("--init_weights", type=str, required=True,
                   help="Path to initial skinning weights (.npz with 'weights' key)")
    p.add_argument("--gt_frames_dir", type=str, required=True)
    p.add_argument("--mask_dir", type=str, default=None)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--view_idx", type=int, default=0,
                   help="Camera view index for rendering. "
                        "Ignored when --cameras_json is provided.")
    p.add_argument("--cameras_json", type=str, default=None,
                   help="Path to cameras.json for custom camera views. "
                        "Overrides --view_idx.")
    p.add_argument("--camera_idx", type=int, default=0,
                   help="Camera index within cameras.json (default: 0)")

    # Phase 2: load rotations from Phase 1
    p.add_argument("--phase1_dir", type=str, default=None,
                   help="Phase 1 output dir (required for phase 2)")

    # Frame range
    p.add_argument("--canonical_frame", type=int, required=True)
    p.add_argument("--start_frame", type=int, required=True)
    p.add_argument("--end_frame", type=int, required=True)
    p.add_argument("--frame_skip", type=int, default=1)

    # Phase 1 optimization
    p.add_argument("--steps_per_frame", type=int, default=500,
                   help="Optimization steps per frame (phase 1)")
    p.add_argument("--steps_first_frame", type=int, default=1000,
                   help="Steps for the first frame (phase 1)")
    p.add_argument("--min_steps", type=int, default=200)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--early_stop_thresh", type=float, default=1e-5,
                   help="Early stop when rotation velocity drops below this")
    p.add_argument("--early_stop_patience", type=int, default=3)
    p.add_argument("--freeze_root_trans", action="store_true",
                   help="Do not optimize root translation (Phase 1: all frames; "
                        "Phase 2: from the start rather than only in the refine stage)")

    # Phase 2 optimization
    p.add_argument("--total_steps", type=int, default=10000,
                   help="Total steps for phase 2 bundle adjustment")
    p.add_argument("--warmup_steps", type=int, default=2000,
                   help="Phase 2: rotation-only warmup before unlocking weights")
    p.add_argument("--refine_start", type=int, default=0,
                   help="Phase 2: step at which the refine stage begins (0 = off). "
                        "At this step the tracking and delta-depth weights drop to 0, "
                        "root translation is frozen, the pose corrections so far are "
                        "baked into the base rotations with a fresh correction MLP, and "
                        "the optimizer and cosine LR schedule restart for the remaining "
                        "total_steps - refine_start steps.")

    # Learning rates
    p.add_argument("--lr_root_trans", type=float, default=1e-3)
    p.add_argument("--lr_weights", type=float, default=1e-2,
                   help="Learning rate for weight logits (phase 2)")
    p.add_argument("--lr", type=float, default=5e-3,
                   help="LR for phase 1 (rotation + root_trans use this)")
    p.add_argument("--lr_end", type=float, default=1e-4)

    # Loss weights — rendering
    p.add_argument("--loss_type", type=str, default="l1_ssim",
                   choices=["mse", "l1", "l1_ssim"],
                   help="RGB loss type: 'l1_ssim' (default, 0.8*L1 + 0.2*(1-SSIM)), 'l1', or 'mse'")
    p.add_argument("--lambda_lpips", type=float, default=0.01)
    p.add_argument("--lambda_mask", type=float, default=0.0,
                   help="Explicit mask/silhouette loss weight (BCE or MSE on rendered alpha vs GT mask). "
                        "Applied on the rendered patch. Separate from model's alpha_loss_weight in config.")
    p.add_argument("--mask_loss_type", type=str, default="bce",
                   choices=["bce", "mse"],
                   help="Mask loss type: 'bce' = binary cross-entropy (Mani-GS style), "
                        "'mse' = mean squared error (default: bce)")
    p.add_argument("--lambda_delta_depth", type=float, default=0.0)
    p.add_argument("--delta_depth_dir", type=str, default=None)
    p.add_argument("--depth_subsample", type=int, default=4096,
                   help="Points sampled per step for the delta-depth loss")

    # Loss weights — LBS regularization
    p.add_argument("--lambda_rot_reg", type=float, default=0.1)
    p.add_argument("--lambda_temporal", type=float, default=0.1)
    p.add_argument("--lambda_smooth", type=float, default=0.1,
                   help="Skinning weight smoothness (phase 2)")
    p.add_argument("--lambda_sparse", type=float, default=0.01,
                   help="Skinning weight sparsity (phase 2)")
    p.add_argument("--lambda_correction", type=float, default=1e-3,
                   help="L2 regularization on MLP correction magnitudes (phase 2)")
    p.add_argument("--lambda_dist", type=float, default=0.0,
                   help="Distance-preservation (ARAP) loss weight (phase 2)")

    # CorrectionMLP (phase 2)
    p.add_argument("--mlp_hidden_dim", type=int, default=128,
                   help="Hidden dimension for CorrectionMLP")
    p.add_argument("--mlp_num_freqs", type=int, default=10,
                   help="Number of positional encoding frequencies for CorrectionMLP")
    p.add_argument("--lr_correction", type=float, default=1e-4,
                   help="Learning rate for CorrectionMLP (phase 2)")

    # 2D tracking loss
    p.add_argument("--track_data", type=str, default=None,
                   help="Path to tracks_2d_attn.npz (phase 2)")
    p.add_argument("--lambda_track", type=float, default=0.005,
                   help="Weight for 2D tracking loss (phase 2)")
    p.add_argument("--track_subsample", type=int, default=1000,
                   help="Number of tracked points to sample per step")

    # KNN
    p.add_argument("--knn_k", type=int, default=12,
                   help="KNN neighbors for weight smoothness (phase 2)")
    p.add_argument("--dist_k", type=int, default=100,
                   help="KNN neighbors for ARAP distance-preservation loss (phase 2)")

    # Patch rendering
    p.add_argument("--patch_h", type=int, default=200)
    p.add_argument("--patch_w", type=int, default=200)

    # Eval / save
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for python, numpy and torch (frame / patch / track sampling, MLP init)")

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Per-frame rotation sweep
# ═══════════════════════════════════════════════════════════════════════════════

def optimize_frame_rotations(rotations, root_trans, frame_idx,
                             canonical_points, skinning_weights,
                             joint_positions, parent_indices, topo_order,
                             coord_scale,
                             model, args_model, device, resume_step,
                             full_rayo, full_rayd, full_rayd_no_norm,
                             full_pix_coords, c2w,
                             gt_frames, gt_masks, loss_masks, bkg_color,
                             lpips_fn,
                             opt_args, H, W, is_first_frame,
                             rot_reg_weights=None,
                             delta_depth_maps=None,
                             eval_dir=None, pbar=None, log_file=None,
                             track_positions=None, track_visibility=None,
                             track_point_idx=None, track_start_frame=0,
                             track_T=0):
    """Optimize rotations + root_trans for a single frame (weights frozen).

    Args:
        rotations: (J, 3) axis-angle, will be optimized in-place
        root_trans: (3,) root translation, will be optimized in-place
        skinning_weights: (N, B) frozen weights (no grad)
    """
    J = rotations.shape[0]
    nominal_steps = opt_args.steps_first_frame if is_first_frame else opt_args.steps_per_frame
    max_steps = opt_args.max_steps
    min_steps = min(opt_args.min_steps, nominal_steps)
    check_interval = 100

    freeze_root = getattr(opt_args, 'freeze_root_trans', False)
    if freeze_root:
        root_trans.requires_grad_(False)
        param_groups = [{'params': [rotations], 'lr': opt_args.lr}]
    else:
        param_groups = [
            {'params': [rotations], 'lr': opt_args.lr},
            {'params': [root_trans], 'lr': opt_args.lr * 0.1},
        ]
    optimizer = torch.optim.Adam(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_steps, eta_min=opt_args.lr_end)

    alpha_loss_weight = getattr(args_model.training, 'alpha_loss_weight', 0.0)
    loss_history = []

    prev_rot_snapshot = rotations.data.clone()
    rot_velocity = float('inf')
    plateau_count = 0
    early_stopped = False

    step = 0
    while step < max_steps:
        optimizer.zero_grad()

        # FK → LBS → deformed points
        deform = forward_kinematics(
            rotations, joint_positions, parent_indices, topo_order,
            root_translation=root_trans)
        deformed = apply_lbs_torch(canonical_points, skinning_weights, deform)
        deformed_scaled = deformed * coord_scale

        # Full-image rendering when patch covers the entire image, else random patch
        full_image_mode = (opt_args.patch_h >= H and opt_args.patch_w >= W)
        if full_image_mode:
            patch_rayd = full_rayd
            patch_rayd_no_norm = full_rayd_no_norm
            patch_pix_coords = full_pix_coords
            patch_gt = gt_frames[frame_idx].unsqueeze(0)
            patch_fg_mask = gt_masks[frame_idx]
            patch_loss_mask = loss_masks[frame_idx]
        else:
            h_start = random.randint(0, H - opt_args.patch_h)
            w_start = random.randint(0, W - opt_args.patch_w)
            h_end = h_start + opt_args.patch_h
            w_end = w_start + opt_args.patch_w

            patch_rayd = full_rayd[:, h_start:h_end, w_start:w_end, :]
            patch_rayd_no_norm = full_rayd_no_norm[:, h_start:h_end, w_start:w_end, :]
            patch_pix_coords = full_pix_coords[:, h_start:h_end, w_start:w_end, :]
            patch_gt = gt_frames[frame_idx][h_start:h_end, w_start:w_end, :].unsqueeze(0)
            patch_fg_mask = gt_masks[frame_idx][h_start:h_end, w_start:w_end, :]
            patch_loss_mask = loss_masks[frame_idx][h_start:h_end, w_start:w_end, :]

        # Forward through PAPR
        out, int_out, grid, regularizers, hit_pred = model(
            full_rayo, patch_rayd, c2w, patch_pix_coords,
            patch_gt, bkg_color, patch_fg_mask.unsqueeze(0),
            cur_step=-1,
            deformed_points=deformed_scaled,
            rays_d_no_norm=patch_rayd_no_norm
        )
        torch.cuda.synchronize()
        rgb = torch.clamp(model.last_act(out), 0, 1)

        # RGB loss
        loss_mask = patch_loss_mask
        num_loss_px = loss_mask.sum().clamp(min=1.0)
        loss_type = getattr(opt_args, 'loss_type', 'l1_ssim')
        if loss_type == 'l1':
            rgb_loss = ((rgb - patch_gt).abs() * loss_mask).sum() / (num_loss_px * 3)
        elif loss_type == 'l1_ssim':
            l1_val = ((rgb - patch_gt).abs() * loss_mask).sum() / (num_loss_px * 3)
            # SSIM expects (B, C, H, W)
            rgb_bchw = (rgb * loss_mask).permute(0, 3, 1, 2)
            gt_bchw = (patch_gt * loss_mask).permute(0, 3, 1, 2)
            ssim_val = compute_ssim(rgb_bchw, gt_bchw)
            rgb_loss = 0.8 * l1_val + 0.2 * (1.0 - ssim_val)
        else:  # mse
            rgb_loss = ((rgb - patch_gt) ** 2 * loss_mask).sum() / (num_loss_px * 3)
        total_loss = rgb_loss

        # LPIPS
        lpips_loss_val = 0.0
        if lpips_fn is not None:
            rgb_masked = rgb * loss_mask + 0.5 * (1.0 - loss_mask)
            gt_masked = patch_gt * loss_mask + 0.5 * (1.0 - loss_mask)
            if rgb_masked.dim() == 3:
                rgb_masked = rgb_masked.unsqueeze(0)
            if gt_masked.dim() == 3:
                gt_masked = gt_masked.unsqueeze(0)
            lpips_loss = opt_args.lambda_lpips * lpips_fn(rgb_masked, gt_masked)
            lpips_loss_val = lpips_loss.item()
            total_loss = total_loss + lpips_loss

        # Alpha / mask loss (from model regularizers)
        alpha_loss_val = 0.0
        if alpha_loss_weight > 0 and 'alpha_loss' in regularizers:
            alpha_loss_tensor, alpha_w = regularizers['alpha_loss']
            total_loss = total_loss + alpha_loss_tensor * alpha_w
            alpha_loss_val = alpha_loss_tensor.item()

        # Explicit mask loss (BCE or MSE on rendered opacity vs GT mask)
        mask_loss_val = 0.0
        if getattr(opt_args, 'lambda_mask', 0.0) > 0:
            # Derive opacity from rendered RGB: bg pixels are bkg_color,
            # opacity ≈ 1 - closeness_to_background
            rgb_squeezed = rgb.squeeze(0)  # (H, W, 3) or (pH, pW, 3)
            rendered_opacity = 1.0 - rgb_squeezed.mean(dim=-1)  # (H, W)
            gt_mask_patch = patch_fg_mask.squeeze(-1)  # (H, W)
            if opt_args.mask_loss_type == 'bce':
                o = rendered_opacity.clamp(1e-6, 1 - 1e-6)
                mask_loss_tensor = -(gt_mask_patch * torch.log(o)
                                     + (1 - gt_mask_patch) * torch.log(1 - o)).mean()
            else:
                mask_loss_tensor = F.mse_loss(rendered_opacity, gt_mask_patch)
            total_loss = total_loss + opt_args.lambda_mask * mask_loss_tensor
            mask_loss_val = (opt_args.lambda_mask * mask_loss_tensor).item()

        # Delta-depth loss
        delta_depth_loss_val = 0.0
        delta_depth_alpha_val = 0.0
        if delta_depth_maps is not None and opt_args.lambda_delta_depth > 0:
            delta_depth_loss_raw, delta_depth_alpha_val = compute_delta_depth_loss(
                deformed, canonical_points, delta_depth_maps,
                c2w, model.fx, model.fy, model.cx, model.cy,
                H, W, coord_scale, loss_masks, frame_idx,
                opt_args.canonical_frame, opt_args.start_frame,
                opt_args.depth_subsample, device)
            delta_depth_loss = opt_args.lambda_delta_depth * delta_depth_loss_raw
            delta_depth_loss_val = delta_depth_loss.item()
            total_loss = total_loss + delta_depth_loss

        # Rotation regularization (single frame)
        rot_sq = (rotations * rotations).sum(dim=-1)  # (J,)
        if rot_reg_weights is not None:
            rot_sq = rot_sq * rot_reg_weights
        rot_reg_loss = opt_args.lambda_rot_reg * rot_sq.mean()
        total_loss = total_loss + rot_reg_loss

        # 2D tracking loss
        track_loss_val = 0.0
        frame_offset = frame_idx - track_start_frame if track_positions is not None else -1
        if track_positions is not None and opt_args.lambda_track > 0 and 0 <= frame_offset < track_T:
            n_sub = min(opt_args.track_subsample, len(track_point_idx))
            sub_idx = torch.randint(len(track_point_idx), (n_sub,), device=device)
            pt_idx = track_point_idx[sub_idx]
            deformed_sub = deformed[pt_idx]
            deformed_sub_scaled = deformed_sub * coord_scale
            pts_2d, _ = fused_projection(
                deformed_sub_scaled, c2w,
                model.fx, model.fy, model.cx, model.cy, H, W,
                vector=False)
            pts_2d = pts_2d.squeeze(0)
            target_2d = track_positions[sub_idx, frame_offset]
            vis = track_visibility[sub_idx, frame_offset]
            vis_count = vis.sum().clamp(min=1.0)
            track_loss = opt_args.lambda_track * (
                ((pts_2d - target_2d) ** 2 * vis.unsqueeze(-1)).sum()
                / (vis_count * 2))
            total_loss = total_loss + track_loss
            track_loss_val = track_loss.item()

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        loss_val = total_loss.item()
        loss_history.append(loss_val)

        # Check rotation velocity
        if step > 0 and step % check_interval == 0:
            rot_delta = (rotations.data - prev_rot_snapshot).norm()
            rot_velocity = rot_delta.item() / check_interval
            prev_rot_snapshot = rotations.data.clone()

            if step >= min_steps:
                if rot_velocity < opt_args.early_stop_thresh:
                    plateau_count += 1
                else:
                    plateau_count = 0
                if plateau_count >= opt_args.early_stop_patience:
                    early_stopped = True

        # Progress bar
        if pbar is not None:
            rot_mag = rotations.data.norm(dim=-1).mean().item()
            pbar.set_postfix_str(
                f"f{frame_idx} loss={loss_val:.4f} rgb={rgb_loss.item():.4f} "
                f"|rot|={rot_mag:.4f} vel={rot_velocity:.2e}")
            pbar.update(1)

        # Log
        if log_file is not None and (step % check_interval == 0 or early_stopped):
            rot_mag = rotations.data.norm(dim=-1).mean().item()
            log_file.write(
                f"frame={frame_idx} step={step:5d}  "
                f"loss={loss_val:.5f}  rgb={rgb_loss.item():.5f}  "
                f"ddepth={delta_depth_loss_val:.5f} dd_a={delta_depth_alpha_val:.3f}  "
                f"lpips={lpips_loss_val:.5f}  alpha={alpha_loss_val:.5f}  "
                f"mask={mask_loss_val:.5f}  "
                f"track={track_loss_val:.5f}  "
                f"rot_reg={rot_reg_loss.item():.5f}  "
                f"|rot|={rot_mag:.5f}  vel={rot_velocity:.2e}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}\n")
            log_file.flush()

        # Intermediate eval
        if (eval_dir is not None and opt_args.eval_every > 0
                and step % opt_args.eval_every == 0):
            with torch.no_grad():
                eval_deformed_scaled = deformed_scaled
            render_comparison(
                eval_deformed_scaled, frame_idx, step, gt_frames, gt_masks,
                model, args_model, device, resume_step,
                full_rayo, full_rayd, full_rayd_no_norm,
                full_pix_coords, c2w, eval_dir)

        if early_stopped:
            break
        step += 1

    actual_steps = step + 1

    # Final eval render
    if eval_dir is not None and opt_args.eval_every > 0:
        with torch.no_grad():
            deform = forward_kinematics(
                rotations, joint_positions, parent_indices, topo_order,
                root_translation=root_trans)
            deformed = apply_lbs_torch(canonical_points, skinning_weights, deform)
            eval_deformed_scaled = deformed * coord_scale
        render_comparison(
            eval_deformed_scaled, frame_idx, step, gt_frames, gt_masks,
            model, args_model, device, resume_step,
            full_rayo, full_rayd, full_rayd_no_norm,
            full_pix_coords, c2w, eval_dir)

    if log_file is not None:
        if early_stopped:
            log_file.write(
                f"  >> EARLY STOP at step {step} (vel={rot_velocity:.2e})\n")
        log_file.flush()

    return loss_history, actual_steps


def run_phase1(args):
    """Phase 1: per-frame rotation sweep with frozen skinning weights."""
    # Normalize frame range so start <= end
    args.start_frame, args.end_frame = (
        min(args.start_frame, args.end_frame),
        max(args.start_frame, args.end_frame))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 1. Load model ----
    print("Loading model and dataset...")
    model, dataset, args_model, device, resume_step = load_model_and_dataset(
        args.config, args.checkpoint)
    coord_scale = model.coord_scale
    H, W = model.H, model.W
    print(f"Model loaded. H={H}, W={W}, coord_scale={coord_scale}")

    for p in model.parameters():
        p.requires_grad_(False)

    # ---- 2. Camera rays ----
    if args.cameras_json:
        print(f"Using camera from {args.cameras_json} (idx={args.camera_idx})")
        full_rayo, full_rayd, full_rayd_no_norm, c2w, full_pix_coords = \
            setup_view_from_cameras_json(
                args.cameras_json, args.camera_idx, dataset,
                device, H, W, coord_scale)
    else:
        view_idx = args.view_idx
        print(f"Using PAPR test set view_idx={view_idx}")
        view_data = dataset.get_full_img(view_idx)
        c2w = dataset.get_c2w([view_idx]).to(device)
        full_rayo = view_data["rayo"].to(device)
        full_rayd = view_data["rayd"].to(device)
        full_rayd_no_norm = view_data["rays_d_no_norm"].to(device)
        full_pix_coords = view_data["pix_coords"].to(device)
        if full_pix_coords.dim() == 3:
            full_pix_coords = full_pix_coords.unsqueeze(0)

    # ---- 3. Canonical points ----
    canonical_points = torch.from_numpy(
        load_canonical_points(args.canonical_points)).to(device)
    N = canonical_points.shape[0]
    print(f"Canonical points: {N}")

    # ---- 4. Skeleton ----
    joint_positions_np, parent_indices = load_autorig_with_scale_check(
        args.autorig_data, model_coord_scale=coord_scale)
    B = len(parent_indices)
    joint_positions = torch.tensor(joint_positions_np, dtype=torch.float32, device=device)
    topo_order = topological_sort(parent_indices)

    roots = [i for i, p in enumerate(parent_indices) if p == -1]
    assert len(roots) == 1, f"Expected 1 root, got {len(roots)}: {roots}"
    root_idx = roots[0]
    print(f"Root joint: {root_idx}, Total joints: {B}")

    rot_reg_weights = rotation_reg_weights(parent_indices, device)

    # ---- 5. Skinning weights (frozen in Phase 1) ----
    init_w = np.load(args.init_weights)
    skinning_weights = torch.tensor(
        init_w['weights'].astype(np.float32), device=device)
    assert skinning_weights.shape == (N, B)
    # Ensure rows sum to 1
    skinning_weights = skinning_weights / skinning_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    print(f"Skinning weights: {skinning_weights.shape} (frozen)")

    # ---- 6. GT frames + masks ----
    load_start = min(args.start_frame, args.end_frame)
    load_end = max(args.start_frame, args.end_frame)
    gt_frames, gt_masks, masks_from = preload_gt_frames(
        args.gt_frames_dir, load_start, load_end, device,
        mask_dir=args.mask_dir)

    sample_gt = next(iter(gt_frames.values()))
    gt_H, gt_W = sample_gt.shape[0], sample_gt.shape[1]
    if gt_H != H or gt_W != W:
        print(f"Resizing GT frames from {gt_H}x{gt_W} to {H}x{W}")
        for frame in gt_frames:
            img = gt_frames[frame].permute(2, 0, 1).unsqueeze(0)
            img = F.interpolate(img, size=(H, W), mode='bilinear', align_corners=False)
            gt_frames[frame] = img.squeeze(0).permute(1, 2, 0)
            msk = gt_masks[frame].permute(2, 0, 1).unsqueeze(0)
            msk = F.interpolate(msk, size=(H, W), mode='nearest')
            gt_masks[frame] = msk.squeeze(0).permute(1, 2, 0)

    loss_masks = {f: torch.ones_like(gt_masks[f]) for f in gt_masks}

    # ---- 7. Delta-depth maps ----
    delta_depth_maps = None
    if args.delta_depth_dir is not None and args.lambda_delta_depth > 0:
        delta_depth_maps = load_delta_depth_maps(
            args.delta_depth_dir, args.start_frame, args.end_frame, H, W, device)
        print(f"Delta-depth maps: {delta_depth_maps.shape}")

    # ---- 8. LPIPS ----
    lpips_fn = None
    if args.lambda_lpips > 0:
        from models.lpips import LPIPS
        lpips_fn = LPIPS(net='vgg').to(device)
        lpips_fn.eval()
        for p in lpips_fn.parameters():
            p.requires_grad_(False)

    # ---- 9. 2D tracking data (optional) ----
    track_positions = track_visibility = track_point_idx = None
    track_start_frame = 0
    track_T = 0
    if args.track_data and os.path.exists(args.track_data):
        print(f"Loading 2D tracks from {args.track_data}...")
        tdata = np.load(args.track_data)
        track_positions = torch.from_numpy(
            tdata["track_positions"]).float().to(device)   # (N_tracked, T, 2)
        track_visibility = torch.from_numpy(
            tdata["track_visibility"]).float().to(device)  # (N_tracked, T)
        track_point_idx = torch.from_numpy(
            tdata["point_indices"]).long().to(device)       # (N_tracked,)
        track_start_frame = int(tdata["start_frame"])
        track_T = track_positions.shape[1]

        # Filter out-of-range point indices
        valid_mask = track_point_idx < N
        if valid_mask.sum() < len(valid_mask):
            track_point_idx = track_point_idx[valid_mask]
            track_positions = track_positions[valid_mask]
            track_visibility = track_visibility[valid_mask]

        # Rescale track positions if original GT resolution differs from model resolution
        if gt_H != H or gt_W != W:
            track_positions[:, :, 0] *= W / gt_W
            track_positions[:, :, 1] *= H / gt_H
            print(f"  Rescaled tracks from {gt_W}x{gt_H} to {W}x{H}")

        print(f"  Tracks: {len(track_point_idx)} points, {track_T} frames, "
              f"start_frame={track_start_frame}")

    # ---- 10. Background color ----
    bkg_color = model.bkg_feats
    if bkg_color is not None:
        bkg_color = bkg_color * model.bkg_scaler.squeeze(0)

    # ---- 11. Build frame order (outward from canonical) ----
    can = args.canonical_frame
    skip = args.frame_skip
    forward_frames = list(range(can + skip, args.end_frame + 1, skip))
    backward_frames = list(range(can - skip, args.start_frame - 1, -skip))
    frame_order = forward_frames + backward_frames
    print(f"Canonical frame: {can}")
    print(f"Frame order: {frame_order}")

    # ---- 12. Optimize ----
    eval_dir = os.path.join(args.output_dir, "eval")
    if args.eval_every > 0:
        os.makedirs(eval_dir, exist_ok=True)

    # Storage for all frame results
    all_rotations = {}
    all_root_trans = {}

    # Current rotation parameters (warm-started from previous frame)
    rotations = torch.zeros(B, 3, device=device, requires_grad=True)
    root_trans = torch.zeros(3, device=device, requires_grad=True)

    nominal_total = args.steps_first_frame + args.steps_per_frame * (len(frame_order) - 1)

    log_path = os.path.join(args.output_dir, "loss_log.txt")
    log_file = open(log_path, "w")
    log_file.write("Phase 1: per-frame rotation sweep\n")
    log_file.write(f"early_stop_thresh={args.early_stop_thresh:.2e}  "
                   f"early_stop_patience={args.early_stop_patience}\n")
    if args.freeze_root_trans:
        log_file.write(f"freeze_root_trans=True\n")
    log_file.flush()
    pbar = tqdm(total=nominal_total, desc="Phase 1", unit="step")

    for i, frame_idx in enumerate(frame_order):
        is_first = (i == 0)

        # When switching from forward to backward sweep, reset rotations
        is_sweep_start = backward_frames and frame_idx == backward_frames[0]
        if is_sweep_start:
            log_file.write(f"\n--- Switching to backward sweep ---\n")
            rotations = torch.zeros(B, 3, device=device, requires_grad=True)
            root_trans = torch.zeros(3, device=device, requires_grad=True)

        nominal_steps = args.steps_first_frame if is_first else args.steps_per_frame

        # ---- Pre-optimization diagnostic: foreground mask loss ----
        with torch.no_grad():
            diag_deform = forward_kinematics(
                rotations, joint_positions, parent_indices, topo_order,
                root_translation=root_trans)
            diag_deformed = apply_lbs_torch(
                canonical_points, skinning_weights, diag_deform)
            diag_deformed_scaled = diag_deformed * coord_scale

            diag_rgb = render_single_frame(
                model, device, args_model, full_rayo, full_rayd,
                full_rayd_no_norm, c2w, full_pix_coords,
                diag_deformed_scaled, resume_step)
            # Rendered foreground mask (threshold on mean RGB)
            rendered_fg = (diag_rgb.squeeze(0).mean(dim=-1, keepdim=True) < 0.98).float()
            gt_fg = gt_masks[frame_idx]  # (H, W, 1)
            # Mask L1: average absolute difference between rendered and GT silhouettes
            mask_l1 = (rendered_fg - gt_fg).abs().mean().item()

        log_file.write(
            f"\nFrame {frame_idx} ({i+1}/{len(frame_order)}), "
            f"nominal={nominal_steps}, |rot|={rotations.data.norm():.5f}, "
            f"mask_l1={mask_l1:.5f}\n")
        log_file.flush()

        loss_history, actual_steps = optimize_frame_rotations(
            rotations, root_trans, frame_idx,
            canonical_points, skinning_weights,
            joint_positions, parent_indices, topo_order,
            coord_scale,
            model, args_model, device, resume_step,
            full_rayo, full_rayd, full_rayd_no_norm, full_pix_coords, c2w,
            gt_frames, gt_masks, loss_masks, bkg_color,
            lpips_fn,
            args, H, W, is_first,
            rot_reg_weights=rot_reg_weights,
            delta_depth_maps=delta_depth_maps,
            eval_dir=eval_dir if args.eval_every > 0 else None,
            pbar=pbar, log_file=log_file,
            track_positions=track_positions,
            track_visibility=track_visibility,
            track_point_idx=track_point_idx,
            track_start_frame=track_start_frame,
            track_T=track_T)

        step_diff = actual_steps - nominal_steps
        if step_diff != 0:
            pbar.total += step_diff
            pbar.refresh()

        # Save per-frame results
        all_rotations[frame_idx] = rotations.data.clone().cpu().numpy()
        all_root_trans[frame_idx] = root_trans.data.clone().cpu().numpy()

        # Save per-frame deformed points
        with torch.no_grad():
            deform = forward_kinematics(
                rotations, joint_positions, parent_indices, topo_order,
                root_translation=root_trans)
            deformed = apply_lbs_torch(canonical_points, skinning_weights, deform)
        deformed_np = deformed.cpu().numpy()
        np.save(os.path.join(args.output_dir, f"deformed_{frame_idx:04d}.npy"),
                deformed_np)
        ply_dir = os.path.join(args.output_dir, "ply")
        os.makedirs(ply_dir, exist_ok=True)
        save_points_ply(deformed_np,
                        os.path.join(ply_dir, f"deformed_{frame_idx:04d}.ply"))

        # Detach and re-create for warm start
        rotations = rotations.detach().clone().requires_grad_(True)
        root_trans = root_trans.detach().clone().requires_grad_(True)

        # Save consolidated results incrementally
        frames_sorted = sorted(all_rotations.keys())
        rot_stack = np.stack([all_rotations[f] for f in frames_sorted])
        trans_stack = np.stack([all_root_trans[f] for f in frames_sorted])
        np.savez(os.path.join(args.output_dir, "phase1_results.npz"),
                 frames=np.array(frames_sorted),
                 rotations=rot_stack,           # (T, B, 3)
                 root_translations=trans_stack,  # (T, 3)
                 joint_positions=joint_positions.cpu().numpy(),
                 parent_indices=np.array(parent_indices),
                 canonical_frame=args.canonical_frame)

    pbar.close()
    log_file.close()

    print(f"\nPhase 1 done. Results in: {args.output_dir}")
    print(f"  Frames: {frames_sorted}")
    print(f"  Rotations: {rot_stack.shape}")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: Bundle adjustment
# ═══════════════════════════════════════════════════════════════════════════════

def run_phase2(args):
    """Phase 2: jointly optimize weights + fine-tune rotations across all frames."""

    args.start_frame, args.end_frame = (
        min(args.start_frame, args.end_frame),
        max(args.start_frame, args.end_frame))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    if args.phase1_dir is None:
        raise ValueError("--phase1_dir is required for phase 2")

    # ---- 1. Load model ----
    print("Loading model and dataset...")
    model, dataset, args_model, device, resume_step = load_model_and_dataset(
        args.config, args.checkpoint)
    coord_scale = model.coord_scale
    H, W = model.H, model.W
    print(f"Model loaded. H={H}, W={W}, coord_scale={coord_scale}")

    for p in model.parameters():
        p.requires_grad_(False)

    # ---- 2. Camera rays ----
    if args.cameras_json:
        print(f"Using camera from {args.cameras_json} (idx={args.camera_idx})")
        full_rayo, full_rayd, full_rayd_no_norm, c2w, full_pix_coords = \
            setup_view_from_cameras_json(
                args.cameras_json, args.camera_idx, dataset,
                device, H, W, coord_scale)
    else:
        view_idx = args.view_idx
        print(f"Using PAPR test set view_idx={view_idx}")
        view_data = dataset.get_full_img(view_idx)
        c2w = dataset.get_c2w([view_idx]).to(device)
        full_rayo = view_data["rayo"].to(device)
        full_rayd = view_data["rayd"].to(device)
        full_rayd_no_norm = view_data["rays_d_no_norm"].to(device)
        full_pix_coords = view_data["pix_coords"].to(device)
        if full_pix_coords.dim() == 3:
            full_pix_coords = full_pix_coords.unsqueeze(0)

    # ---- 3. Canonical points ----
    canonical_points = torch.from_numpy(
        load_canonical_points(args.canonical_points)).to(device)
    N = canonical_points.shape[0]

    # ---- 4. Skeleton ----
    joint_positions_np, parent_indices = load_autorig_with_scale_check(
        args.autorig_data, model_coord_scale=coord_scale)
    B = len(parent_indices)
    joint_positions = torch.tensor(joint_positions_np, dtype=torch.float32, device=device)
    topo_order = topological_sort(parent_indices)
    rot_reg_weights = rotation_reg_weights(parent_indices, device)

    # ---- 5. Load Phase 1 rotations ----
    p1 = np.load(os.path.join(args.phase1_dir, "phase1_results.npz"))
    p1_frames = p1['frames'].tolist()
    p1_rotations = p1['rotations']    # (T, B, 3)
    p1_root_trans = p1['root_translations']  # (T, 3)
    print(f"Loaded Phase 1 rotations: {p1_rotations.shape} for frames {p1_frames}")

    # Build frame list (intersection of phase1 frames and requested range)
    frame_indices = []
    for fi in range(args.start_frame, args.end_frame + 1):
        if fi == args.canonical_frame:
            continue
        if fi in p1_frames:
            frame_indices.append(fi)
    T = len(frame_indices)
    print(f"Phase 2 frames: {frame_indices} ({T} total)")

    # Initialize rotations from Phase 1 (frozen base) + CorrectionMLP
    all_rotations_np = np.zeros((T, B, 3), dtype=np.float32)
    all_root_trans_np = np.zeros((T, 3), dtype=np.float32)
    for t_idx, fi in enumerate(frame_indices):
        p1_idx = p1_frames.index(fi)
        all_rotations_np[t_idx] = p1_rotations[p1_idx]
        all_root_trans_np[t_idx] = p1_root_trans[p1_idx]

    # Phase 1 rotations are FROZEN — corrections come from the MLP
    base_rotations = torch.tensor(all_rotations_np, device=device, requires_grad=False)
    # Precompute base rotation matrices (J, 4, 4) per frame for composition
    base_rot_mats = torch.stack([aa_to_matrix4(base_rotations[t]) for t in range(T)])  # (T, J, 4, 4)

    # Root translations remain raw per-frame parameters
    all_root_trans = torch.tensor(all_root_trans_np, device=device, requires_grad=True)

    # Normalized time values for each frame (for MLP input)
    t_range = max(frame_indices[-1] - frame_indices[0], 1)
    t_normalized_all = torch.tensor(
        [(fi - frame_indices[0]) / t_range for fi in frame_indices],
        dtype=torch.float32, device=device)

    # CorrectionMLP: predicts per-joint axis-angle corrections on top of Phase 1
    correction_mlp = CorrectionMLP(
        num_joints=B, num_freqs=args.mlp_num_freqs,
        hidden_dim=args.mlp_hidden_dim).to(device)
    print(f"CorrectionMLP: {sum(p.numel() for p in correction_mlp.parameters())} params, "
          f"hidden_dim={args.mlp_hidden_dim}, num_freqs={args.mlp_num_freqs}")

    # ---- 6. Skinning weights ----
    init_w = np.load(args.init_weights)
    init_weights = torch.tensor(
        init_w['weights'].astype(np.float32), device=device)
    assert init_weights.shape == (N, B)

    with torch.no_grad():
        clamped = init_weights.clamp(min=1e-6)
        weight_logits = (clamped / clamped.sum(dim=-1, keepdim=True)).log()
    weight_logits = weight_logits.clone().requires_grad_(True)

    # ---- 7. GT frames + masks ----
    load_start = min(args.start_frame, args.end_frame)
    load_end = max(args.start_frame, args.end_frame)
    gt_frames, gt_masks, _ = preload_gt_frames(
        args.gt_frames_dir, load_start, load_end, device,
        mask_dir=args.mask_dir)

    sample_gt = next(iter(gt_frames.values()))
    gt_H, gt_W = sample_gt.shape[0], sample_gt.shape[1]
    if gt_H != H or gt_W != W:
        print(f"Resizing GT frames from {gt_H}x{gt_W} to {H}x{W}")
        for frame in gt_frames:
            img = gt_frames[frame].permute(2, 0, 1).unsqueeze(0)
            img = F.interpolate(img, size=(H, W), mode='bilinear', align_corners=False)
            gt_frames[frame] = img.squeeze(0).permute(1, 2, 0)
            msk = gt_masks[frame].permute(2, 0, 1).unsqueeze(0)
            msk = F.interpolate(msk, size=(H, W), mode='nearest')
            gt_masks[frame] = msk.squeeze(0).permute(1, 2, 0)

    loss_masks = {f: torch.ones_like(gt_masks[f]) for f in gt_masks}

    # ---- 8. KNN (for weight smoothness + ARAP) ----
    print(f"Building KNN for smoothness (k={args.knn_k})...")
    knn_idx = torch.tensor(
        build_knn(canonical_points.cpu().numpy(), k=args.knn_k),
        dtype=torch.long, device=device)

    dist_knn_idx = None
    if args.lambda_dist > 0:
        if args.dist_k == args.knn_k:
            dist_knn_idx = knn_idx
            print(f"  Reusing smoothness KNN for ARAP (k={args.dist_k})")
        else:
            print(f"Building KNN for ARAP (k={args.dist_k})...")
            dist_knn_idx = torch.tensor(
                build_knn(canonical_points.cpu().numpy(), k=args.dist_k),
                dtype=torch.long, device=device)

    # ---- 9. Delta-depth maps ----
    delta_depth_maps = None
    if args.delta_depth_dir is not None and args.lambda_delta_depth > 0:
        delta_depth_maps = load_delta_depth_maps(
            args.delta_depth_dir, args.start_frame, args.end_frame, H, W, device)
        print(f"Delta-depth maps: {delta_depth_maps.shape}")

    # ---- 10. LPIPS ----
    lpips_fn = None
    if args.lambda_lpips > 0:
        from models.lpips import LPIPS
        lpips_fn = LPIPS(net='vgg').to(device)
        lpips_fn.eval()
        for p in lpips_fn.parameters():
            p.requires_grad_(False)

    # ---- 11. 2D tracking data ----
    track_positions = track_visibility = track_point_idx = None
    track_start_frame = 0
    track_T = 0
    N_tracked = 0
    if args.track_data and os.path.exists(args.track_data):
        print(f"Loading 2D tracks from {args.track_data}...")
        tdata = np.load(args.track_data)
        track_positions = torch.from_numpy(
            tdata["track_positions"]).float().to(device)  # (N_tracked, T, 2)
        track_visibility = torch.from_numpy(
            tdata["track_visibility"]).float().to(device)  # (N_tracked, T)
        track_point_idx = torch.from_numpy(
            tdata["point_indices"]).long().to(device)      # (N_tracked,)
        track_start_frame = int(tdata["start_frame"])
        track_T = track_positions.shape[1]

        # Filter out-of-range point indices
        valid_mask = track_point_idx < N
        if valid_mask.sum() < len(valid_mask):
            track_point_idx = track_point_idx[valid_mask]
            track_positions = track_positions[valid_mask]
            track_visibility = track_visibility[valid_mask]
        N_tracked = len(track_point_idx)

        # Rescale track positions from GT resolution to model resolution
        # gt_H, gt_W are the original GT frame dimensions (before resize)
        if gt_H != H or gt_W != W:
            track_positions[:, :, 0] *= W / gt_W
            track_positions[:, :, 1] *= H / gt_H
            print(f"  Rescaled tracks from {gt_W}x{gt_H} to {W}x{H}")

        print(f"  Tracks: {N_tracked} points, {track_T} frames, "
              f"start_frame={track_start_frame}")

    # ---- 12. Background color ----
    bkg_color = model.bkg_feats
    if bkg_color is not None:
        bkg_color = bkg_color * model.bkg_scaler.squeeze(0)

    alpha_loss_weight = getattr(args_model.training, 'alpha_loss_weight', 0.0)

    # ---- 13. Optimizer ----
    def build_optimizer(freeze_root_trans, num_steps):
        param_groups = [
            {'params': correction_mlp.parameters(), 'lr': args.lr_correction},
            {'params': [weight_logits], 'lr': args.lr_weights},
        ]
        if freeze_root_trans:
            all_root_trans.requires_grad_(False)
            print("Root translation frozen")
        else:
            param_groups.append({'params': [all_root_trans], 'lr': args.lr_root_trans})
        opt = torch.optim.Adam(param_groups)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=num_steps, eta_min=args.lr_end)
        return opt, sched

    refine_start = args.refine_start if 0 < args.refine_start < args.total_steps else 0
    optimizer, scheduler = build_optimizer(
        args.freeze_root_trans, refine_start or args.total_steps)
    lambda_track = args.lambda_track
    lambda_delta_depth = args.lambda_delta_depth

    # ---- Helper: compose Phase 1 base rotations + MLP corrections → FK ----
    def compose_and_fk(t_idx, correction_aa):
        """Compose base rotation matrices with MLP correction, run FK.

        Args:
            t_idx: index into frame_indices / base_rot_mats
            correction_aa: (J, 3) axis-angle correction from MLP

        Returns:
            deform: (J, 4, 4) deformation matrices
        """
        corr_mat4 = aa_to_matrix4(correction_aa)  # (J, 4, 4)
        composed = torch.einsum('bij,bjk->bik', base_rot_mats[t_idx], corr_mat4)
        return forward_kinematics(
            None, joint_positions, parent_indices, topo_order,
            root_translation=all_root_trans[t_idx],
            local_rot_mats=composed)

    # ---- 14. Optimization loop ----
    log_path = os.path.join(args.output_dir, "loss_log.txt")
    log_file = open(log_path, "w")
    log_file.write(f"Phase 2: bundle adjustment, {T} frames\n")
    log_file.flush()

    eval_dir = os.path.join(args.output_dir, "eval")
    if args.eval_every > 0:
        os.makedirs(eval_dir, exist_ok=True)

    # Pick 3 diagnostic frames for intermediate PLY saves:
    # first frame (near canonical), middle, and last frame
    diag_t_indices = sorted(set([0, T // 2, T - 1]))
    diag_frame_ids = [frame_indices[i] for i in diag_t_indices]
    ply_progress_dir = os.path.join(args.output_dir, "ply", "progress")
    os.makedirs(ply_progress_dir, exist_ok=True)
    print(f"Diagnostic PLY frames: {diag_frame_ids} (saved every {args.save_every} steps)")

    print(f"\nPhase 2: {args.total_steps} steps "
          f"(warmup={args.warmup_steps} rot-only, then joint)")
    t0 = time.time()

    # Step 0 eval: render and save PLYs for the initial state
    if args.eval_every > 0:
        with torch.no_grad():
            w0 = F.softmax(weight_logits, dim=-1)
            zero_corr = torch.zeros(B, 3, device=device)
            for eval_fi in [frame_indices[0], frame_indices[-1]]:
                eval_t_idx = frame_indices.index(eval_fi)
                deform_eval = compose_and_fk(eval_t_idx, zero_corr)
                deformed_eval = apply_lbs_torch(canonical_points, w0, deform_eval)
                eval_deformed_scaled = deformed_eval * coord_scale
                render_comparison(
                    eval_deformed_scaled, eval_fi, 0, gt_frames, gt_masks,
                    model, args_model, device, resume_step,
                    full_rayo, full_rayd, full_rayd_no_norm,
                    full_pix_coords, c2w, eval_dir)
            for dt_idx in diag_t_indices:
                dfi = frame_indices[dt_idx]
                deform_diag = compose_and_fk(dt_idx, zero_corr)
                deformed_diag = apply_lbs_torch(canonical_points, w0, deform_diag)
                ply_path = os.path.join(
                    ply_progress_dir, f"step00000_f{dfi:04d}.ply")
                save_points_ply(deformed_diag.cpu().numpy(), ply_path)
            print("  Saved step-0 eval renders and diagnostic PLYs")

    for step in range(1, args.total_steps + 1):
        if refine_start and step == refine_start + 1:
            # Refine stage: bake the corrections so far, restart the MLP and the
            # optimizer, freeze root translation, and drop tracking + delta-depth.
            with torch.no_grad():
                base_rotations = bake_corrections(
                    base_rot_mats, correction_mlp(t_normalized_all))
            correction_mlp = CorrectionMLP(
                num_joints=B, num_freqs=args.mlp_num_freqs,
                hidden_dim=args.mlp_hidden_dim).to(device)
            optimizer, scheduler = build_optimizer(
                True, args.total_steps - refine_start)
            lambda_track = 0.0
            lambda_delta_depth = 0.0
            msg = (f"[{step:6d}/{args.total_steps}] refine stage: corrections baked, "
                   f"root translation frozen, tracking and delta-depth off")
            print(msg)
            log_file.write(msg + "\n")
            log_file.flush()

        optimizer.zero_grad()
        in_warmup = step <= args.warmup_steps

        # Pick a random frame for this step
        t_idx = random.randint(0, T - 1)
        fi = frame_indices[t_idx]

        # Current weights
        weights = F.softmax(weight_logits, dim=-1)

        # MLP correction → compose with Phase 1 base → FK → LBS
        t_norm = t_normalized_all[t_idx]
        correction_aa = correction_mlp(t_norm)  # (J, 3)
        deform = compose_and_fk(t_idx, correction_aa)
        deformed = apply_lbs_torch(canonical_points, weights, deform)
        deformed_scaled = deformed * coord_scale

        # Full-image rendering when patch covers the entire image, else random patch
        full_image_mode = (args.patch_h >= H and args.patch_w >= W)
        if full_image_mode:
            patch_rayd = full_rayd
            patch_rayd_no_norm = full_rayd_no_norm
            patch_pix_coords = full_pix_coords
            patch_gt = gt_frames[fi].unsqueeze(0)
            patch_fg_mask = gt_masks[fi]
            patch_loss_mask = loss_masks[fi]
        else:
            h_start = random.randint(0, H - args.patch_h)
            w_start = random.randint(0, W - args.patch_w)
            h_end = h_start + args.patch_h
            w_end = w_start + args.patch_w

            patch_rayd = full_rayd[:, h_start:h_end, w_start:w_end, :]
            patch_rayd_no_norm = full_rayd_no_norm[:, h_start:h_end, w_start:w_end, :]
            patch_pix_coords = full_pix_coords[:, h_start:h_end, w_start:w_end, :]
            patch_gt = gt_frames[fi][h_start:h_end, w_start:w_end, :].unsqueeze(0)
            patch_fg_mask = gt_masks[fi][h_start:h_end, w_start:w_end, :]
            patch_loss_mask = loss_masks[fi][h_start:h_end, w_start:w_end, :]

        # Forward through PAPR
        out, int_out, grid, regularizers, hit_pred = model(
            full_rayo, patch_rayd, c2w, patch_pix_coords,
            patch_gt, bkg_color, patch_fg_mask.unsqueeze(0),
            cur_step=-1,
            deformed_points=deformed_scaled,
            rays_d_no_norm=patch_rayd_no_norm
        )
        torch.cuda.synchronize()
        rgb = torch.clamp(model.last_act(out), 0, 1)

        # RGB loss
        loss_mask = patch_loss_mask
        num_loss_px = loss_mask.sum().clamp(min=1.0)
        loss_type = getattr(args, 'loss_type', 'l1_ssim')
        if loss_type == 'l1':
            rgb_loss = ((rgb - patch_gt).abs() * loss_mask).sum() / (num_loss_px * 3)
        elif loss_type == 'l1_ssim':
            l1_val = ((rgb - patch_gt).abs() * loss_mask).sum() / (num_loss_px * 3)
            rgb_bchw = (rgb * loss_mask).permute(0, 3, 1, 2)
            gt_bchw = (patch_gt * loss_mask).permute(0, 3, 1, 2)
            ssim_val = compute_ssim(rgb_bchw, gt_bchw)
            rgb_loss = 0.8 * l1_val + 0.2 * (1.0 - ssim_val)
        else:  # mse
            rgb_loss = ((rgb - patch_gt) ** 2 * loss_mask).sum() / (num_loss_px * 3)
        total_loss = rgb_loss

        # LPIPS
        lpips_loss_val = 0.0
        if lpips_fn is not None:
            rgb_masked = rgb * loss_mask + 0.5 * (1.0 - loss_mask)
            gt_masked = patch_gt * loss_mask + 0.5 * (1.0 - loss_mask)
            if rgb_masked.dim() == 3:
                rgb_masked = rgb_masked.unsqueeze(0)
            if gt_masked.dim() == 3:
                gt_masked = gt_masked.unsqueeze(0)
            lpips_loss = args.lambda_lpips * lpips_fn(rgb_masked, gt_masked)
            lpips_loss_val = lpips_loss.item()
            total_loss = total_loss + lpips_loss

        # Alpha / mask loss
        alpha_loss_val = 0.0
        if alpha_loss_weight > 0 and 'alpha_loss' in regularizers:
            alpha_loss_tensor, alpha_w = regularizers['alpha_loss']
            total_loss = total_loss + alpha_loss_tensor * alpha_w
            alpha_loss_val = alpha_loss_tensor.item()

        # Explicit mask loss (BCE or MSE on rendered opacity vs GT mask).
        # Mirrors the Phase 1 term in optimize_frame_rotations().
        mask_loss_val = 0.0
        if getattr(args, 'lambda_mask', 0.0) > 0:
            rgb_squeezed = rgb.squeeze(0)                      # (pH, pW, 3)
            rendered_opacity = 1.0 - rgb_squeezed.mean(dim=-1)  # (pH, pW)
            gt_mask_patch = patch_fg_mask.squeeze(-1)          # (pH, pW)
            if args.mask_loss_type == 'bce':
                o = rendered_opacity.clamp(1e-6, 1 - 1e-6)
                mask_loss_tensor = -(gt_mask_patch * torch.log(o)
                                     + (1 - gt_mask_patch) * torch.log(1 - o)).mean()
            else:
                mask_loss_tensor = F.mse_loss(rendered_opacity, gt_mask_patch)
            total_loss = total_loss + args.lambda_mask * mask_loss_tensor
            mask_loss_val = (args.lambda_mask * mask_loss_tensor).item()

        # Delta-depth loss
        delta_depth_loss_val = 0.0
        delta_depth_alpha_val = 0.0
        if delta_depth_maps is not None and lambda_delta_depth > 0:
            delta_depth_loss_raw, delta_depth_alpha_val = compute_delta_depth_loss(
                deformed, canonical_points, delta_depth_maps,
                c2w, model.fx, model.fy, model.cx, model.cy,
                H, W, coord_scale, loss_masks, fi,
                args.canonical_frame, args.start_frame,
                args.depth_subsample, device)
            delta_depth_loss = lambda_delta_depth * delta_depth_loss_raw
            delta_depth_loss_val = delta_depth_loss.item()
            total_loss = total_loss + delta_depth_loss

        # 2D tracking loss
        track_loss_val = 0.0
        frame_offset = fi - track_start_frame if track_positions is not None else -1
        if track_positions is not None and lambda_track > 0 and 0 <= frame_offset < track_T:
            n_sub = min(args.track_subsample, N_tracked)
            sub_idx = torch.randint(N_tracked, (n_sub,), device=device)
            pt_idx = track_point_idx[sub_idx]
            pts_sub = canonical_points[pt_idx]
            w_sub = weights[pt_idx]
            deformed_sub = apply_lbs_torch(pts_sub, w_sub, deform)
            deformed_sub_scaled = deformed_sub * coord_scale
            pts_2d, _ = fused_projection(
                deformed_sub_scaled, c2w,
                model.fx, model.fy, model.cx, model.cy, H, W,
                vector=False)
            pts_2d = pts_2d.squeeze(0)
            target_2d = track_positions[sub_idx, frame_offset]
            vis = track_visibility[sub_idx, frame_offset]
            vis_count = vis.sum().clamp(min=1.0)
            track_loss = lambda_track * (
                ((pts_2d - target_2d) ** 2 * vis.unsqueeze(-1)).sum()
                / (vis_count * 2))
            total_loss = total_loss + track_loss
            track_loss_val = track_loss.item()

        # ARAP / distance-preservation loss
        dist_loss_val = 0.0
        if args.lambda_dist > 0 and dist_knn_idx is not None:
            dist_loss = args.lambda_dist * compute_distance_preservation_loss(
                deformed, canonical_points, dist_knn_idx)
            total_loss = total_loss + dist_loss
            dist_loss_val = dist_loss.item()

        # Correction regularization: evaluate MLP at all frames
        corrections_all = correction_mlp(t_normalized_all)  # (T, J, 3)
        l_corr = (corrections_all ** 2).mean()
        total_loss = total_loss + args.lambda_correction * l_corr

        # Temporal smoothness on corrections
        l_temp = temporal_smoothness(corrections_all)
        total_loss = total_loss + args.lambda_temporal * l_temp

        # Rotation regularization on composed rotations (weighted by descendants)
        # Regularize the full composed rotation magnitude, not just corrections.
        # Use Frobenius norm of (R_composed - I) as a proxy for rotation magnitude
        # (equivalent to ||log(R)||^2 for small rotations, and monotonically
        # related for large rotations).
        composed_rot_sq_all = []
        for _t in range(T):
            _corr = corrections_all[_t]  # (J, 3)
            _corr_mat = aa_to_matrix4(_corr)
            _composed = torch.einsum('bij,bjk->bik', base_rot_mats[_t], _corr_mat)
            # ||R - I||_F^2 for each joint
            eye3 = torch.eye(3, device=_composed.device)
            diff = _composed[:, :3, :3] - eye3.unsqueeze(0)
            rot_sq = (diff * diff).sum(dim=(1, 2))  # (J,)
            composed_rot_sq_all.append(rot_sq)
        composed_rot_sq = torch.stack(composed_rot_sq_all)  # (T, J)
        if rot_reg_weights is not None:
            composed_rot_sq = composed_rot_sq * rot_reg_weights[None, :]
        l_rot = composed_rot_sq.mean()
        total_loss = total_loss + args.lambda_rot_reg * l_rot

        # Weight regularization (only after warmup)
        l_smooth_val, l_sparse_val = 0.0, 0.0
        if not in_warmup:
            l_smooth = smoothness_loss(weights, knn_idx)
            l_sparse = sparsity_loss(weights)
            total_loss = total_loss + args.lambda_smooth * l_smooth
            total_loss = total_loss + args.lambda_sparse * l_sparse
            l_smooth_val = l_smooth.item()
            l_sparse_val = l_sparse.item()

        total_loss.backward()

        # During warmup, zero out weight gradients
        if in_warmup and weight_logits.grad is not None:
            weight_logits.grad.zero_()

        optimizer.step()
        scheduler.step()

        # Logging
        if step % 100 == 0 or step == 1:
            phase_str = ("WARMUP" if in_warmup
                         else "REFINE" if refine_start and step > refine_start
                         else "JOINT")
            msg = (f"[{step:6d}/{args.total_steps}] {phase_str} f{fi}  "
                   f"loss={total_loss.item():.5f}  rgb={rgb_loss.item():.5f}  "
                   f"lpips={lpips_loss_val:.5f}  mask={mask_loss_val:.5f}  "
                   f"track={track_loss_val:.5f}  "
                   f"ddepth={delta_depth_loss_val:.5f} dd_a={delta_depth_alpha_val:.3f}  "
                   f"corr={l_corr.item():.5f}  rot_reg={l_rot.item():.5f}  "
                   f"temp={l_temp.item():.5f}  "
                   f"dist={dist_loss_val:.5f}  "
                   f"smooth={l_smooth_val:.5f}  sparse={l_sparse_val:.5f}")
            print(msg)
            log_file.write(msg + "\n")
            log_file.flush()

        # Eval
        if args.eval_every > 0 and (step % args.eval_every == 0 or step == args.total_steps):
            with torch.no_grad():
                w = F.softmax(weight_logits, dim=-1)
                corr_eval_all = correction_mlp(t_normalized_all)
                # Render first and last frame
                for eval_fi in [frame_indices[0], frame_indices[-1]]:
                    eval_t_idx = frame_indices.index(eval_fi)
                    deform_eval = compose_and_fk(eval_t_idx, corr_eval_all[eval_t_idx])
                    deformed_eval = apply_lbs_torch(canonical_points, w, deform_eval)
                    eval_deformed_scaled = deformed_eval * coord_scale
                    render_comparison(
                        eval_deformed_scaled, eval_fi, step, gt_frames, gt_masks,
                        model, args_model, device, resume_step,
                        full_rayo, full_rayd, full_rayd_no_norm,
                        full_pix_coords, c2w, eval_dir)

        # Checkpoint
        if step % args.save_every == 0 or step == args.total_steps:
            with torch.no_grad():
                corr_ckpt_all = correction_mlp(t_normalized_all)  # (T, J, 3)
            ckpt_path = os.path.join(args.output_dir, f'checkpoint_step{step:05d}.npz')
            np.savez(
                ckpt_path,
                skinning_weights=F.softmax(weight_logits, dim=-1).detach().cpu().numpy(),
                weight_logits=weight_logits.detach().cpu().numpy(),
                base_rotations=base_rotations.cpu().numpy(),
                corrections=corr_ckpt_all.detach().cpu().numpy(),
                root_translations=all_root_trans.detach().cpu().numpy(),
                joint_positions=joint_positions.cpu().numpy(),
                parent_indices=np.array(parent_indices),
                frame_indices=np.array(frame_indices),
                canonical_frame=args.canonical_frame,
                step=step,
            )
            # Save MLP weights
            mlp_path = os.path.join(args.output_dir, f'correction_mlp_step{step:05d}.pt')
            torch.save(correction_mlp.state_dict(), mlp_path)
            print(f"  Saved checkpoint: {ckpt_path}")

            # Save diagnostic PLY point clouds for 3 frames
            with torch.no_grad():
                w_ply = F.softmax(weight_logits, dim=-1)
                corr_diag_all = correction_mlp(t_normalized_all)
                for dt_idx in diag_t_indices:
                    dfi = frame_indices[dt_idx]
                    deform_diag = compose_and_fk(dt_idx, corr_diag_all[dt_idx])
                    deformed_diag = apply_lbs_torch(canonical_points, w_ply, deform_diag)
                    ply_path = os.path.join(
                        ply_progress_dir, f"step{step:05d}_f{dfi:04d}.ply")
                    save_points_ply(deformed_diag.cpu().numpy(), ply_path)
                print(f"  Saved diagnostic PLYs for frames {diag_frame_ids}")

    log_file.close()

    # Save final deformed point clouds for all frames
    print("Saving final deformed point clouds...")
    with torch.no_grad():
        w = F.softmax(weight_logits, dim=-1)
        corr_final_all = correction_mlp(t_normalized_all)
        for t_idx, fi in enumerate(frame_indices):
            deform = compose_and_fk(t_idx, corr_final_all[t_idx])
            deformed = apply_lbs_torch(canonical_points, w, deform)
            deformed_np = deformed.cpu().numpy()
            np.save(os.path.join(args.output_dir, f"deformed_{fi:04d}.npy"),
                    deformed_np)
            ply_dir = os.path.join(args.output_dir, "ply")
            os.makedirs(ply_dir, exist_ok=True)
            save_points_ply(deformed_np,
                            os.path.join(ply_dir, f"deformed_{fi:04d}.ply"))
    print(f"Saved {len(frame_indices)} deformed point clouds.")

    print(f"\nPhase 2 done. Total time: {time.time() - t0:.1f}s")
    print(f"Results in: {args.output_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def save_run_metadata(args):
    """Copy this script and save args to the output directory for reproducibility."""
    import shutil
    import json
    os.makedirs(args.output_dir, exist_ok=True)
    # Copy the script itself
    script_path = os.path.abspath(__file__)
    shutil.copy2(script_path, os.path.join(args.output_dir, os.path.basename(script_path)))
    # Save args as JSON
    args_dict = vars(args)
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(args_dict, f, indent=2, default=str)
    # Also save the full command line
    with open(os.path.join(args.output_dir, "command.txt"), "w") as f:
        f.write(" ".join(sys.argv) + "\n")


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)  # also seeds every CUDA device
    save_run_metadata(args)
    if args.phase == 1:
        run_phase1(args)
    elif args.phase == 2:
        run_phase2(args)


if __name__ == "__main__":
    main()
