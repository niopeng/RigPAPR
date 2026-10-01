"""
Precompute 2D point tracks using CoTracker on GT rendered frames.

Projects PAPR canonical points to 2D at the canonical frame, then tracks
them across all GT frames using CoTracker.  Output is used as supervision
for the skinning weight optimizer (--track_data flag).

Usage:
    python scripts/precompute_tracks.py \
        --config <papr52.yml> \
        --checkpoint <model.pth> \
        --canonical_points output/lbs_experiment/papr_canonical_world.npy \
        --gt_frames_dir output/lbs_experiment/blender/gt_frames \
        --canonical_frame 40 \
        --output output/tracks_2d.npz
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from models.utils import fused_projection
from rigpapr.papr_render import load_model_and_dataset

# CoTracker 3: pip install git+https://github.com/facebookresearch/co-tracker
COTRACKER_CKPT = os.path.join(
    PROJECT_ROOT, "checkpoints", "cotracker3", "scaled_offline.pth"
)
COTRACKER_URL = (
    "https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth"
)


def ensure_cotracker_ckpt():
    """Download CoTracker checkpoint if not present."""
    if os.path.exists(COTRACKER_CKPT):
        return
    os.makedirs(os.path.dirname(COTRACKER_CKPT), exist_ok=True)
    print(f"Downloading CoTracker checkpoint → {COTRACKER_CKPT}")
    torch.hub.download_url_to_file(COTRACKER_URL, COTRACKER_CKPT)
    print("Done.")


def load_gt_video(gt_frames_dir, start_frame, end_frame, white_bg=False):
    """Load GT rendered frames as a video tensor (1, T, 3, H, W)."""
    frames = []
    for f in range(start_frame, end_frame + 1):
        path = os.path.join(gt_frames_dir, f"frame_{f:04d}.png")
        img = Image.open(path)
        if white_bg and img.mode == "RGBA":
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(bg, img).convert("RGB")
        else:
            img = img.convert("RGB")
        arr = np.array(img, dtype=np.float32)
        frames.append(torch.from_numpy(arr))
    video = torch.stack(frames, dim=0)  # (T, H, W, 3)
    video = video.permute(0, 3, 1, 2).unsqueeze(0)  # (1, T, 3, H, W)
    return video


def project_points_to_2d(canonical_np, coord_scale, c2w, fx, fy, cx, cy, H, W,
                          device):
    """Project canonical points → 2D pixel coords.  Returns (N,2) and cam-space coords."""
    pts = torch.from_numpy(canonical_np).float().to(device)
    pts_scaled = pts * coord_scale
    pts_2d, pts_cam = fused_projection(
        pts_scaled, c2w, fx, fy, cx, cy, H, W, vector=False
    )
    return pts_2d, pts_cam


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--canonical_points", type=str, required=True)
    p.add_argument("--gt_frames_dir", type=str, required=True)
    p.add_argument("--start_frame", type=int, default=0)
    p.add_argument("--end_frame", type=int, default=90)
    p.add_argument("--canonical_frame", type=int, default=40)
    p.add_argument("--view_idx", type=int, default=0)
    p.add_argument("--output", type=str, default="output/tracks_2d.npz")
    p.add_argument("--batch_size", type=int, default=500,
                    help="Max queries per CoTracker batch.")
    p.add_argument("--cotracker_window", type=int, default=60)
    p.add_argument("--white_bg", action="store_true",
                    help="Composite RGBA GT frames onto white instead of black.")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 1. Load PAPR model to get camera params ─────────────────────────────
    print("Loading PAPR model for camera parameters...")
    model, dataset, args_model, _, _ = load_model_and_dataset(
        args.config, args.checkpoint
    )
    coord_scale = model.coord_scale
    H, W = model.H, model.W
    fx = dataset.focal_x
    fy = dataset.focal_y
    cx = dataset.cx
    cy = dataset.cy
    c2w = dataset.get_c2w([args.view_idx]).to(device)  # (1, 4, 4)
    print(f"  H={H}, W={W}, fx={fx:.1f}, fy={fy:.1f}, coord_scale={coord_scale}")

    # ── 2. Load canonical points and project to 2D ───────────────────────────
    canonical_np = np.load(args.canonical_points).astype(np.float32)
    N = len(canonical_np)
    print(f"Canonical points: {N}")

    pts_2d, pts_cam = project_points_to_2d(
        canonical_np, coord_scale, c2w, fx, fy, cx, cy, H, W, device
    )
    pts_2d_np = pts_2d.squeeze(0).cpu().numpy()   # (N, 2)
    z_cam = pts_cam.squeeze(0)[..., 2].cpu().numpy()  # (N,)

    # Free PAPR model from GPU — CoTracker needs the memory
    del model, dataset
    torch.cuda.empty_cache()

    # Filter: in front of camera (z < 0 in PAPR convention) and within bounds
    in_front = z_cam < 0
    in_bounds_x = (pts_2d_np[:, 0] >= 0) & (pts_2d_np[:, 0] < W)
    in_bounds_y = (pts_2d_np[:, 1] >= 0) & (pts_2d_np[:, 1] < H)
    visible = in_front & in_bounds_x & in_bounds_y
    visible_indices = np.where(visible)[0]
    N_vis = len(visible_indices)
    print(f"Visible points at canonical frame: {N_vis}/{N}")

    query_2d = pts_2d_np[visible_indices]  # (N_vis, 2) — pixel coords

    # ── 3. Load GT video ─────────────────────────────────────────────────────
    print(f"Loading GT frames {args.start_frame}..{args.end_frame}...")
    video = load_gt_video(args.gt_frames_dir, args.start_frame, args.end_frame,
                          white_bg=args.white_bg)
    T = video.shape[1]
    print(f"  Video shape: {video.shape}  ({T} frames)")

    # ── 4. Download CoTracker checkpoint if needed ───────────────────────────
    ensure_cotracker_ckpt()

    # ── 5. Run CoTracker in batches ──────────────────────────────────────────
    from cotracker.predictor import CoTrackerPredictor

    print("Loading CoTracker...")
    tracker = CoTrackerPredictor(
        checkpoint=COTRACKER_CKPT,
        v2=False,
        offline=True,
        window_len=args.cotracker_window,
    ).to(device)

    video_gpu = video.to(device)

    # Canonical frame index WITHIN the video (frame 40 → index 40 if start=0)
    canon_idx_in_video = args.canonical_frame - args.start_frame

    all_tracks = []
    all_vis = []
    bs = args.batch_size

    n_batches = (N_vis + bs - 1) // bs
    print(f"Tracking {N_vis} points in {n_batches} batches of {bs}...")

    for bi in range(n_batches):
        start = bi * bs
        end = min(start + bs, N_vis)
        batch_xy = torch.from_numpy(query_2d[start:end]).float().to(device)  # (B, 2)

        # CoTracker query format: (1, B, 3) with (frame_idx, x, y)
        frame_col = torch.full(
            (1, batch_xy.shape[0], 1), canon_idx_in_video,
            dtype=batch_xy.dtype, device=device
        )
        queries = torch.cat([frame_col, batch_xy.unsqueeze(0)], dim=2)  # (1, B, 3)

        with torch.no_grad():
            pred_tracks, pred_visibility = tracker(
                video_gpu,
                queries=queries,
                grid_query_frame=canon_idx_in_video,
                backward_tracking=True,
            )
        # pred_tracks: (1, B, T, 2), pred_visibility: (1, B, T)
        all_tracks.append(pred_tracks.squeeze(0).cpu().numpy())
        all_vis.append(pred_visibility.squeeze(0).cpu().numpy())
        print(f"  Batch {bi+1}/{n_batches}: tracked {end - start} points")

    track_positions = np.concatenate(all_tracks, axis=0)  # (N_vis, T, 2)
    track_visibility = np.concatenate(all_vis, axis=0)     # (N_vis, T)

    # ── 6. Save ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez(
        args.output,
        track_positions=track_positions,
        track_visibility=track_visibility,
        point_indices=visible_indices,
        canonical_frame=args.canonical_frame,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    n_vis_mean = track_visibility.mean()
    print(f"\nSaved {args.output}")
    print(f"  Tracked points: {N_vis}")
    print(f"  Frames: {T}")
    print(f"  Mean visibility: {n_vis_mean:.3f}")


if __name__ == "__main__":
    main()
