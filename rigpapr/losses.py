"""Losses and regularisers used by the optimizer."""
import torch
import torch.nn.functional as F

from models.utils import fused_projection


def smoothness_loss(weights, knn_idx):
    """L1 smoothness: nearby points should have similar weights."""
    # weights: (N, B), knn_idx: (N, k)
    N, B = weights.shape
    k = knn_idx.shape[1]
    neighbor_weights = weights[knn_idx.reshape(-1)].reshape(N, k, B)  # (N, k, B)
    diff = (weights.unsqueeze(1) - neighbor_weights).abs()  # (N, k, B)
    return diff.mean()


def sparsity_loss(weights):
    """Entropy penalty — encourages weights to concentrate on few bones."""
    eps = 1e-8
    entropy = -(weights * (weights + eps).log()).sum(dim=-1)  # (N,)
    return entropy.mean()


def temporal_smoothness(all_rotations):
    """L2 penalty on rotation differences between consecutive frames."""
    if all_rotations.shape[0] < 2:
        return torch.tensor(0.0, device=all_rotations.device)
    diff = all_rotations[1:] - all_rotations[:-1]
    return (diff * diff).mean()


def compute_distance_preservation_loss(deformed_points, canonical_points, knn_idx):
    """
    Distance-preservation (simplified ARAP) loss: penalizes changes in pairwise
    distances between each point and its KNN neighbors after deformation.

        L_dist = mean_i mean_{j in N(i)} ( ||p'_i - p'_j|| - ||p_i - p_j|| )^2

    where p is canonical, p' is deformed.  This prevents local neighborhoods
    from stretching or compressing, which is the visible symptom when neighboring
    points have inconsistent skinning weights under large bone rotations.

    Args:
        deformed_points: (N, 3) deformed positions (current frame)
        canonical_points: (N, 3) canonical (rest) positions
        knn_idx: (N, k) LongTensor of neighbor indices (precomputed)

    Returns:
        scalar loss
    """
    # Canonical distances (could be precomputed, but the gather dominates anyway)
    can_neighbors = canonical_points[knn_idx]          # (N, k, 3)
    can_diff = canonical_points.unsqueeze(1) - can_neighbors  # (N, k, 3)
    can_dist = can_diff.norm(dim=-1)                   # (N, k)

    # Deformed distances
    def_neighbors = deformed_points[knn_idx]           # (N, k, 3)
    def_diff = deformed_points.unsqueeze(1) - def_neighbors  # (N, k, 3)
    def_dist = def_diff.norm(dim=-1)                   # (N, k)

    return ((def_dist - can_dist) ** 2).mean()


def compute_delta_depth_loss(deformed_points, canonical_points, delta_depth_maps,
                             c2w, fx, fy, cx, cy, H, W, coord_scale,
                             loss_masks, frame_idx, canonical_frame, start_frame,
                             subsample, device):
    """Delta-depth loss: supervise depth *change* relative to the canonical frame.

    Instead of comparing absolute depth against a monocular estimate (which has
    global bias), compare the per-point depth change (z_deformed - z_canonical)
    against the monocular depth change (mono_t - mono_canonical).  Any consistent
    bias in the monocular estimator cancels out in the difference.

    A per-sample scale factor alpha is fitted via least-squares to absorb any
    remaining unit mismatch between metric depth and monocular depth.

    Args:
        deformed_points: (N, 3) deformed points (world, NOT coord_scale-multiplied)
        canonical_points: (N, 3) canonical points (world)
        delta_depth_maps: (T, H, W) metric monocular depth (e.g. DA3 mono)
        c2w: (1, 4, 4) camera-to-world
        fx, fy, cx, cy: intrinsics
        H, W: image size
        coord_scale: PAPR coord_scale
        loss_masks: {frame: (H, W, 1)} foreground masks
        frame_idx: current frame being optimised
        canonical_frame: canonical frame index
        start_frame: first frame index (for indexing into delta_depth_maps)
        subsample: number of points to sample
        device: torch device

    Returns:
        scalar loss, or 0 if too few valid points.
    """
    N = deformed_points.shape[0]
    n_sub = min(subsample, N)
    sub_idx = torch.randint(N, (n_sub,), device=device)

    pts_def = deformed_points[sub_idx] * coord_scale
    pts_can = canonical_points[sub_idx] * coord_scale

    # Project deformed points -> camera-space z
    pts_2d_def, pts_cam_def = fused_projection(
        pts_def, c2w, fx, fy, cx, cy, H, W, vector=False)
    pts_2d_def = pts_2d_def.squeeze(0)
    pts_cam_def = pts_cam_def.squeeze(0)
    z_deformed = -pts_cam_def[:, 2]
    px_def = pts_2d_def[:, 0]
    py_def = pts_2d_def[:, 1]

    # Project canonical points -> camera-space z
    pts_2d_can, pts_cam_can = fused_projection(
        pts_can, c2w, fx, fy, cx, cy, H, W, vector=False)
    pts_2d_can = pts_2d_can.squeeze(0)
    pts_cam_can = pts_cam_can.squeeze(0)
    z_canonical = -pts_cam_can[:, 2]
    px_can = pts_2d_can[:, 0]
    py_can = pts_2d_can[:, 1]

    # Valid: both projections must be in-bounds and in front of camera
    valid = (
        (z_deformed > 0) & (z_canonical > 0)
        & (px_def >= 0) & (px_def < W) & (py_def >= 0) & (py_def < H)
        & (px_can >= 0) & (px_can < W) & (py_can >= 0) & (py_can < H)
    )

    # Filter by foreground mask (current frame)
    if valid.any():
        lm = loss_masks[frame_idx][:, :, 0]
        px_long = px_def[valid].long().clamp(0, W - 1)
        py_long = py_def[valid].long().clamp(0, H - 1)
        in_fg = lm[py_long, px_long] > 0.5
        valid_indices = valid.nonzero(as_tuple=False).squeeze(-1)
        valid[valid_indices[~in_fg]] = False

    if valid.sum() < 10:
        return torch.tensor(0.0, device=device, requires_grad=True), 0.0

    z_def_valid = z_deformed[valid]
    z_can_valid = z_canonical[valid]  # detached (canonical points are frozen)
    px_def_valid = px_def[valid]
    py_def_valid = py_def[valid]
    px_can_valid = px_can[valid]
    py_can_valid = py_can[valid]

    # Sample monocular depth at projected locations
    fi_t = frame_idx - start_frame
    fi_can = canonical_frame - start_frame
    mono_map_t = delta_depth_maps[fi_t]
    mono_map_can = delta_depth_maps[fi_can]

    def _sample(mono_map, px_v, py_v):
        gx = 2.0 * px_v / (W - 1) - 1.0
        gy = 2.0 * py_v / (H - 1) - 1.0
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).unsqueeze(0)
        return F.grid_sample(
            mono_map.unsqueeze(0).unsqueeze(0), grid,
            mode="bilinear", padding_mode="border", align_corners=True
        ).squeeze()

    mono_t = _sample(mono_map_t, px_def_valid, py_def_valid)
    mono_can = _sample(mono_map_can, px_can_valid, py_can_valid)

    # Delta: depth change from canonical to current
    delta_z = z_def_valid - z_can_valid          # actual depth change
    delta_mono = mono_t - mono_can               # monocular depth change

    # Least-squares scale:  delta_z ≈ alpha * delta_mono
    with torch.no_grad():
        dm_det = delta_mono.detach()
        dz_det = delta_z.detach()
        alpha = (dz_det * dm_det).sum() / (dm_det ** 2).sum().clamp(min=1e-8)

    residual = delta_z - alpha * delta_mono
    loss = (residual ** 2).mean()
    return loss, float(alpha.item())
