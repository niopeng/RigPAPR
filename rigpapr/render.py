"""
Render a fitted sequence from one camera, and score it if ground truth is given.

Reads a Phase 2 checkpoint (or a Phase 1 result), rebuilds each frame's
deformed point cloud (FK + LBS) and renders it with the canonical PAPR model.

    python render.py --config configs/trex/papr.yml \
      --checkpoint data/trex/papr/model.pth \
      --canonical_points data/trex/papr/canonical_points.npy \
      --run_dir output/trex/phase2 --view_idx 27 \
      --gt_frames_dir data/trex/driving/frames --mask_dir data/trex/driving/masks \
      --out_dir output/trex/render_view27

Writes <out_dir>/frames/frame_XXXX.png and <out_dir>/render.mp4. With
--gt_frames_dir it also writes <out_dir>/metrics.txt: PSNR over the mask's
foreground pixels, SSIM over the mask's bounding box, and LPIPS (VGG) over the
full frame, per frame and averaged. Without --mask_dir, PSNR and SSIM use the
full frame.
"""
import argparse
import glob
import os

import imageio
import lpips
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity

from rigpapr.data import load_canonical_points
from rigpapr.lbs import aa_to_matrix4, apply_lbs_torch, forward_kinematics, topological_sort
from rigpapr.papr_render import (
    load_model_and_dataset, render_single_frame, setup_view_from_cameras_json)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True, help="canonical PAPR model.pth")
    p.add_argument("--canonical_points", required=True)
    p.add_argument("--run_dir", required=True,
                   help="Phase 2 output dir (or a Phase 1 dir, with --init_weights)")
    p.add_argument("--step", type=int, default=None,
                   help="Phase 2 checkpoint step (default: latest)")
    p.add_argument("--init_weights", default=None,
                   help="skinning weights .npz; only needed to render a Phase 1 run")
    p.add_argument("--view_idx", type=int, default=0,
                   help="camera index in the PAPR dataset (ignored with --cameras_json)")
    p.add_argument("--cameras_json", default=None)
    p.add_argument("--camera_idx", type=int, default=0)
    p.add_argument("--gt_frames_dir", default=None,
                   help="frame_XXXX.png ground truth; enables metrics")
    p.add_argument("--mask_dir", default=None,
                   help="frame_XXXX.png foreground masks for PSNR / SSIM")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--fps", type=int, default=10)
    return p.parse_args()


def load_run(run_dir, step, init_weights, device):
    """Return per-frame local rotation matrices (T, J, 4, 4), root translations
    (T, 3), skinning weights (N, J), joint positions, parent indices, frames."""
    ckpts = sorted(glob.glob(os.path.join(run_dir, "checkpoint_step*.npz")))
    if ckpts:
        path = (os.path.join(run_dir, f"checkpoint_step{step:05d}.npz")
                if step is not None else ckpts[-1])
        d = np.load(path)
        base = torch.tensor(d["base_rotations"], device=device)
        corr = torch.tensor(d["corrections"], device=device)
        rot_mats = torch.stack([
            aa_to_matrix4(base[t]) @ aa_to_matrix4(corr[t]) for t in range(len(base))])
        weights = d["skinning_weights"]
        frames = d["frame_indices"].tolist()
    else:
        path = os.path.join(run_dir, "phase1_results.npz")
        if init_weights is None:
            raise ValueError("rendering a Phase 1 run needs --init_weights")
        d = np.load(path)
        rot = torch.tensor(d["rotations"], device=device)
        rot_mats = torch.stack([aa_to_matrix4(r) for r in rot])
        weights = np.load(init_weights)["weights"].astype(np.float32)
        weights = weights / np.clip(weights.sum(-1, keepdims=True), 1e-8, None)
        frames = d["frames"].tolist()
    print(f"Loaded {path}: {len(frames)} frames")
    return dict(
        rot_mats=rot_mats,
        root_trans=torch.tensor(d["root_translations"], device=device),
        weights=torch.tensor(weights, device=device),
        joint_positions=torch.tensor(d["joint_positions"], device=device),
        parent_indices=d["parent_indices"].tolist(),
        frames=frames)


def load_rgb(path):
    img = Image.open(path)
    if img.mode == "RGBA":  # composite onto white, like PAPR's background
        rgba = np.asarray(img, dtype=np.float32) / 255.0
        return rgba[..., :3] * rgba[..., 3:] + (1.0 - rgba[..., 3:])
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def main():
    args = parse_args()
    device = torch.device("cuda")
    model, dataset, args_model, device, resume_step = load_model_and_dataset(
        args.config, args.checkpoint)
    H, W, coord_scale = model.H, model.W, model.coord_scale

    if args.cameras_json:
        rayo, rayd, rayd_no_norm, c2w, pix_coords = setup_view_from_cameras_json(
            args.cameras_json, args.camera_idx, dataset, device, H, W, coord_scale)
    else:
        view = dataset.get_full_img(args.view_idx)
        c2w = dataset.get_c2w([args.view_idx]).to(device)
        rayo, rayd = view["rayo"].to(device), view["rayd"].to(device)
        rayd_no_norm = view["rays_d_no_norm"].to(device)
        pix_coords = view["pix_coords"].to(device)
        if pix_coords.dim() == 3:
            pix_coords = pix_coords.unsqueeze(0)

    canonical = torch.from_numpy(load_canonical_points(args.canonical_points)).to(device)
    run = load_run(args.run_dir, args.step, args.init_weights, device)
    topo_order = topological_sort(run["parent_indices"])

    frame_dir = os.path.join(args.out_dir, "frames")
    os.makedirs(frame_dir, exist_ok=True)
    lpips_fn = lpips.LPIPS(net="vgg").to(device).eval() if args.gt_frames_dir else None

    video, rows = [], []
    for t, fi in sorted(enumerate(run["frames"]), key=lambda x: x[1]):
        with torch.no_grad():
            deform = forward_kinematics(
                None, run["joint_positions"], run["parent_indices"], topo_order,
                root_translation=run["root_trans"][t], local_rot_mats=run["rot_mats"][t])
            points = apply_lbs_torch(canonical, run["weights"], deform)
            rgb = render_single_frame(model, device, args_model, rayo, rayd, rayd_no_norm,
                                      c2w, pix_coords, points * coord_scale, resume_step)
        render = rgb.squeeze(0).clamp(0, 1).cpu().numpy()
        frame_u8 = (render * 255).astype(np.uint8)
        Image.fromarray(frame_u8).save(os.path.join(frame_dir, f"frame_{fi:04d}.png"))
        video.append(frame_u8)

        if lpips_fn is None:
            continue
        gt = load_rgb(os.path.join(args.gt_frames_dir, f"frame_{fi:04d}.png"))
        mask = None
        if args.mask_dir:
            mask = np.asarray(Image.open(os.path.join(
                args.mask_dir, f"frame_{fi:04d}.png")).convert("L")) > 127
        if gt.shape[:2] != (H, W):
            gt = np.asarray(Image.fromarray((gt * 255).astype(np.uint8)).resize(
                (W, H), Image.LANCZOS), dtype=np.float32) / 255.0
            if mask is not None:
                mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(
                    (W, H), Image.NEAREST)) > 127
        if mask is not None:
            mse = ((render[mask] - gt[mask]) ** 2).mean()
            ys, xs = np.where(mask)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            ssim = structural_similarity(render[y0:y1, x0:x1], gt[y0:y1, x0:x1],
                                         channel_axis=2, data_range=1.0)
        else:
            mse = ((render - gt) ** 2).mean()
            ssim = structural_similarity(render, gt, channel_axis=2, data_range=1.0)
        psnr = -10 * np.log10(mse) if mse > 1e-10 else 100.0
        to_t = lambda x: torch.from_numpy(x).permute(2, 0, 1)[None].float().to(device) * 2 - 1
        with torch.no_grad():
            lp = lpips_fn(to_t(render), to_t(gt)).item()
        rows.append((fi, psnr, ssim, lp))
        print(f"  frame {fi}: PSNR {psnr:.2f}  SSIM {ssim:.4f}  LPIPS {lp:.4f}")

    imageio.mimwrite(os.path.join(args.out_dir, "render.mp4"), video, fps=args.fps,
                     quality=8, macro_block_size=1)
    print(f"Wrote {len(video)} frames and render.mp4 to {args.out_dir}")

    if rows:
        m = np.mean([r[1:] for r in rows], axis=0)
        with open(os.path.join(args.out_dir, "metrics.txt"), "w") as f:
            f.write(f"# {'masked' if args.mask_dir else 'full-frame'} PSNR / SSIM, "
                    f"full-frame LPIPS (VGG)\n# frame     PSNR    SSIM   LPIPS\n")
            for fi, ps, ss, lp in rows:
                f.write(f"{fi:7d}  {ps:7.2f}  {ss:.4f}  {lp:.4f}\n")
            f.write(f"   mean  {m[0]:7.2f}  {m[1]:.4f}  {m[2]:.4f}\n")
        print(f"Mean over {len(rows)} frames: PSNR {m[0]:.2f}  SSIM {m[1]:.4f}  LPIPS {m[2]:.4f}")


if __name__ == "__main__":
    main()
