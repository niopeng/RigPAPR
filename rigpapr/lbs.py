"""Skeleton kinematics and linear blend skinning (LBS)."""
import torch


def aa_to_matrix4(aa):
    """Axis-angle (J, 3) → 4×4 rotation matrices (J, 4, 4).
    Uses sinc-based Rodrigues formula for stable gradients at aa ≈ 0."""
    J = aa.shape[0]
    theta_sq = (aa * aa).sum(dim=-1)
    theta_sq_safe = theta_sq.clamp(min=1e-12)
    theta = theta_sq_safe.sqrt()

    small = theta_sq < 1e-8
    sinc = torch.where(small, 1.0 - theta_sq_safe / 6.0, theta.sin() / theta)
    cosc = torch.where(small, 0.5 - theta_sq_safe / 24.0, (1.0 - theta.cos()) / theta_sq_safe)

    x, y, z = aa[:, 0], aa[:, 1], aa[:, 2]
    zero = torch.zeros_like(x)
    K = torch.stack([
        torch.stack([zero, -z, y], dim=-1),
        torch.stack([z, zero, -x], dim=-1),
        torch.stack([-y, x, zero], dim=-1),
    ], dim=-2)

    I3 = torch.eye(3, device=aa.device, dtype=aa.dtype).unsqueeze(0)
    R = I3 + sinc[:, None, None] * K + cosc[:, None, None] * (K @ K)

    bottom = torch.zeros(J, 1, 3, device=aa.device, dtype=aa.dtype)
    R4x3 = torch.cat([R, bottom], dim=1)
    last_col = torch.tensor([0., 0., 0., 1.], device=aa.device, dtype=aa.dtype
                            ).reshape(1, 4, 1).expand(J, -1, -1)
    return torch.cat([R4x3, last_col], dim=2)


def topological_sort(parent_indices):
    """Return joint indices in parent-before-child order."""
    J = len(parent_indices)
    visited = [False] * J
    order = []

    def visit(j):
        if visited[j]:
            return
        p = parent_indices[j]
        if p >= 0 and not visited[p]:
            visit(p)
        visited[j] = True
        order.append(j)

    for j in range(J):
        visit(j)
    return order


def forward_kinematics(local_rotations, joint_positions, parent_indices, topo_order,
                        root_translation=None, local_rot_mats=None):
    """
    Compute FK deformation matrices from per-joint local axis-angle rotations.

    Args:
        local_rotations: (J, 3) axis-angle per joint (can be None if local_rot_mats provided)
        joint_positions: (J, 3) rest-pose world positions
        parent_indices: list/array of int, -1 for root
        topo_order: precomputed topological order
        root_translation: (3,) or None – translation for root joint
        local_rot_mats: (J, 4, 4) optional pre-composed rotation matrices
                        (skips aa_to_matrix4 conversion if provided)

    Returns:
        deform_matrices: (J, 4, 4) – maps rest-pose points to posed positions
    """
    if local_rot_mats is not None:
        J = local_rot_mats.shape[0]
        device = local_rot_mats.device
        dtype = local_rot_mats.dtype
    else:
        J = local_rotations.shape[0]
        device = local_rotations.device
        dtype = local_rotations.dtype
        local_rot_mats = aa_to_matrix4(local_rotations)  # (J, 4, 4)

    # Build world-posed transforms via FK (list to avoid in-place issues)
    world_posed = [None] * J

    for j in topo_order:
        p = parent_indices[j]
        if p < 0:
            # Root joint
            t = joint_positions[j].clone()
            if root_translation is not None:
                t = t + root_translation
            # T(t) @ R(aa) — translation then rotation
            # Actually: world_posed = T(t) @ R around origin
            # But we want T(t) to be the position, so:
            # world_posed[j] = [[R, t], [0, 1]]
            # But we need: world_posed such that deform = world_posed @ T(-jp)
            # gives identity at rest. At rest R=I, t=jp, so deform = T(jp)@T(-jp) = I. ✓
            # In posed state: t = jp + root_trans
            mat = local_rot_mats[j].clone()
            mat[:3, 3] = t
            world_posed[j] = mat
        else:
            # Non-root: world_posed[j] = world_posed[parent] @ T(offset) @ R(local)
            offset = joint_positions[j] - joint_positions[p]
            # T(offset) @ R(local): rotation part = R, translation = offset
            local_mat = local_rot_mats[j].clone()
            local_mat[:3, 3] = offset
            world_posed[j] = world_posed[p] @ local_mat

    world_posed = torch.stack(world_posed)  # (J, 4, 4)

    # deform[j] = world_posed[j] @ T(-jp[j])
    # Rotation part stays: deform[:,:3,:3] = world_posed[:,:3,:3]
    # Translation: deform[:,:3,3] = world_posed[:,:3,3] - world_posed[:,:3,:3] @ jp[j]
    deform = world_posed.clone()
    deform[:, :3, 3] = (world_posed[:, :3, 3]
                        - torch.einsum('bij,bj->bi', world_posed[:, :3, :3], joint_positions))
    return deform


def compute_joint_depth_and_descendants(parent_indices):
    """Compute depth and descendant count for each joint.
    Returns depth (J,) and n_descendants (J,) arrays."""
    J = len(parent_indices)
    depth = [0] * J
    children = [[] for _ in range(J)]
    for j in range(J):
        p = parent_indices[j]
        if p >= 0:
            children[p].append(j)

    # Compute depth via BFS from roots
    from collections import deque
    queue = deque()
    for j in range(J):
        if parent_indices[j] < 0:
            depth[j] = 0
            queue.append(j)
    while queue:
        j = queue.popleft()
        for c in children[j]:
            depth[c] = depth[j] + 1
            queue.append(c)

    # Compute descendant count via post-order traversal
    n_desc = [0] * J
    def count_desc(j):
        total = 0
        for c in children[j]:
            total += 1 + count_desc(c)
        n_desc[j] = total
        return total
    for j in range(J):
        if parent_indices[j] < 0:
            count_desc(j)

    return depth, n_desc


def apply_lbs_torch(canonical_points, skinning_weights, deform_matrices):
    """
    Apply LBS deformation using PyTorch (fully differentiable).

    Args:
        canonical_points: (N, 3) tensor, rest-pose positions
        skinning_weights: (N, B) tensor, weights (rows sum to 1)
        deform_matrices: (B, 4, 4) tensor, = pose @ inv(rest)

    Returns:
        (N, 3) tensor, deformed positions
    """
    N = canonical_points.shape[0]

    # Homogenize: (N, 3) -> (N, 4)
    pts_h = torch.cat([
        canonical_points,
        torch.ones(N, 1, device=canonical_points.device, dtype=canonical_points.dtype)
    ], dim=1)

    # Transform each point by each bone: (B, 4, 4) x (N, 4) -> (B, N, 4)
    transformed = torch.einsum('bij,nj->bni', deform_matrices, pts_h)

    # Blend using weights: (N, B) x (B, N, 4) -> (N, 4)
    deformed_h = torch.einsum('nb,bni->ni', skinning_weights, transformed)

    return deformed_h[:, :3]


def build_knn(points, k=12):
    """Compute k-nearest-neighbor indices for each point. Returns (N, k) int."""
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    _, idx = tree.query(points, k=k + 1)  # +1 because first neighbor is self
    return idx[:, 1:]  # exclude self
