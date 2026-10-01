"""
Precompute 2D point tracks using PAPR attention-based point selection (Approach 2).

Instead of tracking all visible points, this script:
1. Runs a PAPR forward pass at the canonical frame to extract per-pixel attention.
2. For each foreground pixel, selects the dominant point (highest attention weight).
3. Deduplicates to get unique dominant point indices.
4. Projects each point to its own 2D position (avoiding the off-ray issue).
5. Tracks those positions with CoTracker.

Output format is identical to precompute_tracks.py; pass it to the optimizer
as --track_data. Needs CoTracker 3
(pip install git+https://github.com/facebookresearch/co-tracker); the
checkpoint is downloaded to checkpoints/cotracker3/ on first use.

Usage (see --help for the camera and frame-range options):
    python scripts/precompute_tracks_attn.py \
        --config configs/trex/papr.yml \
        --checkpoint data/trex/papr/model.pth \
        --canonical_points data/trex/papr/canonical_points.npy \
        --gt_frames_dir data/trex/driving/frames \
        --canonical_frame 40 \
        --output data/trex/driving/tracks_2d_attn.npz
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from models.utils import fused_projection
from scripts.precompute_tracks import load_gt_video, ensure_cotracker_ckpt, COTRACKER_CKPT
from rigpapr.papr_render import load_model_and_dataset


def parse_args():
    p = argparse.ArgumentParser(
        description="Precompute attention-based top-1 point tracks")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--canonical_points", type=str, required=True)
    p.add_argument("--gt_frames_dir", type=str, required=True)
    p.add_argument("--start_frame", type=int, default=0)
    p.add_argument("--end_frame", type=int, default=90)
    p.add_argument("--canonical_frame", type=int, default=40)
    p.add_argument("--view_idx", type=int, default=0,
                   help="Camera view index. Ignored when --cameras_json is provided.")
    p.add_argument("--cameras_json", type=str, default=None,
                   help="Path to cameras.json for custom camera views. "
                        "Overrides --view_idx.")
    p.add_argument("--camera_idx", type=int, default=0,
                   help="Camera index within cameras.json (default: 0)")
    p.add_argument("--output", type=str, default="output/tracks_2d_attn.npz")
    p.add_argument("--batch_size", type=int, default=500,
                    help="Max queries per CoTracker batch.")
    p.add_argument("--cotracker_window", type=int, default=60)
    p.add_argument("--min_attn_threshold", type=float, default=0.05,
                    help="Minimum top-1 attention to include a pixel's dominant point.")
    p.add_argument("--white_bg", action="store_true",
                    help="Composite RGBA GT frames onto white instead of black.")
    return p.parse_args()


def compute_final_select_k(model, args_model):
    """Compute the actual K used by the model, accounting for config overrides."""
    final_select_k = model.select_k
    if args_model.geoms.points.select_k_rnd:
        final_select_k = max(final_select_k, args_model.geoms.points.select_k_max)
    if args_model.models.attn.select_prop_scores:
        final_select_k = min(final_select_k, args_model.models.attn.select_prop_scores_topk)
    if args_model.models.attn.select_v_scores:
        final_select_k = min(final_select_k, args_model.models.attn.select_v_scores_topk)
        if args_model.models.attn.select_v_scores_rnd:
            final_select_k = max(final_select_k, args_model.models.attn.select_v_scores_topk_max)
    return final_select_k


def compute_attn_tensor_size(model, args_model, final_select_k):
    """Compute the full attention tensor K-dimension (foreground + background entries)."""
    bkg_seq_len_attn = 0
    if hasattr(model, 'args') and model.args.geoms.no_additional_bkg:
        bkg_seq_len_attn = 0
    elif (model.bkg_feats is not None
          and not model.args.geoms.points.use_bkg_points
          and not model.args.models.attn.append_bkg_points):
        bkg_seq_len_attn = model.bkg_feats.shape[0]

    append_bkg = 1 if (model.args.models.attn.append_bkg_points
                        and not model.args.geoms.no_additional_bkg) else 0

    return final_select_k + append_bkg + bkg_seq_len_attn


def run_papr_forward(model, args_model, rayo, rayd, rayd_no_norm, c2w,
                     pix_coords, resume_step, device):
    """Run model.evaluate() in patches, collect full-image select_k_ind and attn."""
    H = model.H
    W = model.W
    final_select_k = compute_final_select_k(model, args_model)
    attn_k_dim = compute_attn_tensor_size(model, args_model, final_select_k)

    full_select_inds = torch.empty(1, H, W, final_select_k, dtype=torch.long,
                                   device=device)
    full_attn = torch.zeros(1, H, W, attn_k_dim, 1, device=device)

    max_h = getattr(args_model.test, 'max_height', 100)
    max_w = getattr(args_model.test, 'max_width', 100)

    print(f"Running PAPR forward pass in {max_h}x{max_w} patches "
          f"(image {H}x{W}, K={final_select_k})...")

    with torch.no_grad():
        for h_start in range(0, H, max_h):
            for w_start in range(0, W, max_w):
                h_end = min(h_start + max_h, H)
                w_end = min(w_start + max_w, W)

                _, attn_patch, _, _ = model.evaluate(
                    rayo,
                    rayd[:, h_start:h_end, w_start:w_end],
                    c2w,
                    pix_coords[:, h_start:h_end, w_start:w_end],
                    step=resume_step,
                    rays_d_no_norm=rayd_no_norm[:, h_start:h_end, w_start:w_end],
                )

                full_select_inds[:, h_start:h_end, w_start:w_end, :] = \
                    model.select_k_ind
                full_attn[:, h_start:h_end, w_start:w_end, :, :] = attn_patch

    print("  Forward pass complete.")
    return full_select_inds, full_attn, final_select_k


def extract_dominant_points(full_select_inds, full_attn, fg_mask_2d,
                            final_select_k, min_attn_threshold):
    """Per-pixel argmax over K foreground attention entries; filter and deduplicate.

    Returns unique_point_indices as a numpy int64 array of global point indices.
    """
    # Slice to foreground attention only (first final_select_k entries)
    fg_attn = full_attn[0, :, :, :final_select_k, 0]  # (H, W, K)

    # Per-pixel argmax
    max_vals, local_idx = fg_attn.max(dim=-1)  # (H, W), (H, W)

    # Map local K-index to global point index
    dominant_idx = full_select_inds[0].gather(
        dim=-1, index=local_idx.unsqueeze(-1)
    ).squeeze(-1)  # (H, W)

    # Filter by foreground mask and attention threshold
    valid = fg_mask_2d & (max_vals > min_attn_threshold)

    n_fg = fg_mask_2d.sum().item()
    n_valid = valid.sum().item()

    # Deduplicate
    unique_indices = torch.unique(dominant_idx[valid]).cpu().numpy().astype(np.int64)
    N_attn = len(unique_indices)

    # Diagnostics
    valid_attn = max_vals[valid]
    print(f"  Foreground pixels: {n_fg}")
    print(f"  Pixels passing threshold (>{min_attn_threshold}): {n_valid}")
    print(f"  Unique dominant points: {N_attn}")
    if N_attn > 0:
        print(f"  Dominant attention — min: {valid_attn.min():.4f}, "
              f"max: {valid_attn.max():.4f}, mean: {valid_attn.mean():.4f}")

    return unique_indices


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Phase A: Load PAPR model + camera data ───────────────────────────────
    print("Loading PAPR model...")
    model, dataset, args_model, device, resume_step = load_model_and_dataset(
        args.config, args.checkpoint
    )
    coord_scale = model.coord_scale
    H, W = model.H, model.W
    fx, fy = dataset.focal_x, dataset.focal_y
    cx, cy = dataset.cx, dataset.cy
    print(f"  H={H}, W={W}, fx={fx:.1f}, fy={fy:.1f}, coord_scale={coord_scale}")

    # Get full-image ray data for the canonical view
    if args.cameras_json:
        from dataset.utils import get_rays

        print(f"Using camera from {args.cameras_json} (idx={args.camera_idx})")
        with open(args.cameras_json, 'r') as f:
            cam_data = json.load(f)
        camera_angle_x = cam_data["camera_angle_x"]
        frames = cam_data["frames"]
        if args.camera_idx >= len(frames):
            raise ValueError(f"camera_idx={args.camera_idx} out of range "
                             f"(cameras.json has {len(frames)} frames)")
        frame = frames[args.camera_idx]
        c2w_np = np.array(frame["transform_matrix"], dtype=np.float64)
        scale_mat = np.diag([coord_scale, coord_scale, coord_scale, 1.0])
        c2w_scaled = (scale_mat @ c2w_np).astype(np.float32)
        c2w = torch.tensor(c2w_scaled).unsqueeze(0).to(device)

        focal = 0.5 * W / math.tan(0.5 * camera_angle_x)
        fx = fy = focal
        cx, cy = W / 2.0, H / 2.0

        rayo, rayd, _, rayd_no_norm = get_rays(H, W, fx, fy,
                                                torch.tensor(c2w_scaled).unsqueeze(0))
        rayo = rayo.to(device)
        rayd = rayd.to(device)
        rayd_no_norm = rayd_no_norm.to(device)

        i_coords = torch.arange(W, dtype=torch.float32) + 0.5
        j_coords = torch.arange(H, dtype=torch.float32) + 0.5
        ii, jj = torch.meshgrid(i_coords, j_coords, indexing='xy')
        pix_coords = torch.stack([ii, jj], dim=-1).to(device).unsqueeze(0)

        # No dataset mask available — derive from a quick render or use all pixels
        mask = torch.ones(1, H, W, 1)
    else:
        print(f"Using PAPR test set view_idx={args.view_idx}")
        view_data = dataset.get_full_img(args.view_idx)
        c2w = dataset.get_c2w([args.view_idx]).to(device)
        rayo = view_data["rayo"].to(device)
        rayd = view_data["rayd"].to(device)
        rayd_no_norm = view_data["rays_d_no_norm"].to(device)
        pix_coords = view_data["pix_coords"]
        if pix_coords.dim() == 3:
            pix_coords = pix_coords.unsqueeze(0)
        pix_coords = pix_coords.to(device)
        mask = view_data["mask"]  # (1, H, W, 1)

    # Load canonical points and verify correspondence
    pts_path = args.canonical_points
    if pts_path.endswith('.ply'):
        import open3d as o3d
        canonical_np = np.asarray(o3d.io.read_point_cloud(pts_path).points).astype(np.float32)
    else:
        canonical_np = np.load(pts_path).astype(np.float32)
    N_pts = model.points.shape[0]
    if len(canonical_np) != N_pts:
        raise ValueError(
            f"Canonical points ({len(canonical_np)}) != model points ({N_pts}). "
            f"Ensure canonical_points was extracted from the same checkpoint.")
    print(f"  Canonical points: {len(canonical_np)} (matches model)")

    # ── Phase B: Patch-based PAPR forward pass ───────────────────────────────
    full_select_inds, full_attn, final_select_k = run_papr_forward(
        model, args_model, rayo, rayd, rayd_no_norm, c2w, pix_coords,
        resume_step, device
    )

    # Move to CPU to free GPU for CoTracker later
    full_select_inds = full_select_inds.cpu()
    full_attn = full_attn.cpu()

    # ── Phase C: Extract dominant point per pixel ────────────────────────────
    print("Extracting dominant points...")
    fg_mask_2d = mask[0, :, :, 0].cpu() > 0.5  # (H, W) boolean
    unique_point_indices = extract_dominant_points(
        full_select_inds, full_attn, fg_mask_2d,
        final_select_k, args.min_attn_threshold
    )
    N_attn = len(unique_point_indices)
    if N_attn == 0:
        raise RuntimeError("No dominant points found. Try lowering --min_attn_threshold.")

    # ── Phase D: Project dominant points to their own 2D positions ───────────
    print("Projecting dominant points to 2D...")
    pts = torch.from_numpy(canonical_np[unique_point_indices]).float().to(device)
    pts_scaled = pts * coord_scale
    pts_2d, _ = fused_projection(pts_scaled, c2w, fx, fy, cx, cy, H, W,
                                  vector=False)
    query_2d = pts_2d.squeeze(0).cpu().numpy()  # (N_attn, 2)

    # Scale query coordinates from model resolution to GT frame resolution
    # (CoTracker operates on the GT video which may be higher-res)
    sample_frame = os.path.join(args.gt_frames_dir, "frame_0000.png")
    if os.path.exists(sample_frame):
        from PIL import Image
        gt_img = Image.open(sample_frame)
        gt_W, gt_H = gt_img.size
        if gt_W != W or gt_H != H:
            scale_x = gt_W / W
            scale_y = gt_H / H
            query_2d[:, 0] *= scale_x
            query_2d[:, 1] *= scale_y
            print(f"  Scaled query 2D from model res ({W}x{H}) to GT res ({gt_W}x{gt_H})")

    print(f"  Query 2D positions: {query_2d.shape}")

    # ── Phase E: Free model, run CoTracker ───────────────────────────────────
    del model, dataset, full_select_inds, full_attn, pts, pts_scaled, pts_2d
    del rayo, rayd, rayd_no_norm, pix_coords, c2w
    torch.cuda.empty_cache()

    print(f"Loading GT frames {args.start_frame}..{args.end_frame}...")
    video = load_gt_video(args.gt_frames_dir, args.start_frame, args.end_frame,
                          white_bg=args.white_bg)
    T = video.shape[1]
    print(f"  Video shape: {video.shape}  ({T} frames)")

    ensure_cotracker_ckpt()

    from cotracker.predictor import CoTrackerPredictor

    print("Loading CoTracker...")
    tracker = CoTrackerPredictor(
        checkpoint=COTRACKER_CKPT,
        v2=False,
        offline=True,
        window_len=args.cotracker_window,
    ).to(device)

    video_gpu = video.to(device)
    canon_idx_in_video = args.canonical_frame - args.start_frame

    all_tracks = []
    all_vis = []
    bs = args.batch_size
    n_batches = (N_attn + bs - 1) // bs

    print(f"Tracking {N_attn} attention-selected points in {n_batches} batches...")
    for bi in range(n_batches):
        start = bi * bs
        end = min(start + bs, N_attn)
        batch_xy = torch.from_numpy(query_2d[start:end]).float().to(device)

        frame_col = torch.full(
            (1, batch_xy.shape[0], 1), canon_idx_in_video,
            dtype=batch_xy.dtype, device=device
        )
        queries = torch.cat([frame_col, batch_xy.unsqueeze(0)], dim=2)

        with torch.no_grad():
            pred_tracks, pred_visibility = tracker(
                video_gpu,
                queries=queries,
                grid_query_frame=canon_idx_in_video,
                backward_tracking=True,
            )
        # CoTracker may return (1, T, B, 2) or (1, B, T, 2) depending on version.
        # Normalise to (B, T, *) so axis=0 concatenation yields (N_attn, T, *).
        tracks_np = pred_tracks.squeeze(0).cpu().numpy()   # (T, B, 2) or (B, T, 2)
        vis_np = pred_visibility.squeeze(0).cpu().numpy()  # (T, B)   or (B, T)
        n_queries = end - start
        T_video = video_gpu.shape[1]
        if tracks_np.shape[0] == T_video and tracks_np.shape[1] == n_queries:
            # Shape is (T, B, 2) — transpose to (B, T, 2)
            tracks_np = tracks_np.transpose(1, 0, 2)
            vis_np = vis_np.T
        all_tracks.append(tracks_np)
        all_vis.append(vis_np)
        print(f"  Batch {bi+1}/{n_batches}: tracked {end - start} points")

    track_positions = np.concatenate(all_tracks, axis=0)  # (N_attn, T, 2)
    track_visibility = np.concatenate(all_vis, axis=0)     # (N_attn, T)

    # ── Phase F: Save ────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez(
        args.output,
        track_positions=track_positions,
        track_visibility=track_visibility,
        point_indices=unique_point_indices,
        canonical_frame=args.canonical_frame,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    vis_mean = track_visibility.mean()
    print(f"\nSaved {args.output}")
    print(f"  Attention-selected points: {N_attn}")
    print(f"  Frames: {T}")
    print(f"  Mean visibility: {vis_mean:.3f}")


if __name__ == "__main__":
    main()
