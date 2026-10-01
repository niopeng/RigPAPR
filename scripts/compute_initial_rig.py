"""
Compute initial rig and skinning weights for PAPR point cloud.

Simplified pipeline that:
  1. Parses a Puppeteer auto-rig file (joint hierarchy + proxy skinning weights)
  2. Inverse-normalizes the proxy mesh from Puppeteer's normalized space to world space
  3. Transfers skinning weights from proxy mesh to PAPR canonical points via KNN

Outputs:
  - autorig_data.npz: joint_positions, weight_matrix, joint_names, parent_indices,
        and optionally coordinate_scale (multiply positions by this to get PAPR scale)
  - papr_skinning_weights.npz: weights (N, B) for the PAPR point cloud

If --coordinate_scale is provided, it is stored in autorig_data.npz so that
downstream scripts know how to rescale joint positions back to PAPR's internal
coordinate space (i.e., multiply by coordinate_scale).

Usage:
    python scripts/compute_initial_rig.py \
      --rig_file /path/to/puppeteer/final_rigging/rexy.txt \
      --proxy_mesh /path/to/puppeteer/skel_results/rexy_mesh.obj \
      --input_mesh /path/to/puppeteer/input/rexy.ply \
      --canonical_points output/lbs_experiment/papr_canonical_world.npy \
      --coordinate_scale 10.0 \
      --output_dir output/initial_rig
"""
import os
import sys
import argparse
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute initial rig and skinning weights for PAPR")

    # Puppeteer inputs
    parser.add_argument("--rig_file", type=str, required=True,
                        help="Path to Puppeteer rig .txt file")
    parser.add_argument("--proxy_mesh", type=str, required=True,
                        help="Path to proxy mesh OBJ (Puppeteer normalized space)")
    parser.add_argument("--input_mesh", type=str, required=True,
                        help="Path to input mesh OBJ/PLY (Blender world space), "
                             "used to compute inverse-normalization transform")

    # PAPR canonical points
    parser.add_argument("--canonical_points", type=str, required=True,
                        help="Path to papr_canonical_world.npy")

    # Parameters
    parser.add_argument("--k_neighbors", type=int, default=6,
                        help="Number of nearest neighbors for KNN weight transfer")
    parser.add_argument("--coordinate_scale", type=float, default=None,
                        help="PAPR coordinate scale factor (e.g. 10.0). If provided, "
                             "stored in autorig_data.npz. Downstream scripts should "
                             "multiply joint positions by this value to convert from "
                             "normalized space back to PAPR's internal coordinate space.")

    # Output
    parser.add_argument("--output_dir", type=str, default="output/initial_rig")

    return parser.parse_args()


def load_vertices(path):
    """Read vertices from an OBJ or PLY file. Returns (V, 3) float32."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.ply':
        import trimesh
        mesh = trimesh.load(path, process=False)
        return np.array(mesh.vertices, dtype=np.float32)
    # Default: OBJ text format
    verts = []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(verts, dtype=np.float32)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Step 1: Parse Puppeteer rig ──────────────────────────────
    print("=" * 60)
    print("Step 1: Parse Puppeteer Rig")
    print("=" * 60)

    from rigpapr.parse_puppeteer_rig import parse_puppeteer_rig

    joint_names, joint_positions, joint_parents, proxy_weight_matrix = \
        parse_puppeteer_rig(args.rig_file)

    # Convert joint_parents dict → int32 array (-1 = root)
    name_to_idx = {n: i for i, n in enumerate(joint_names)}
    parent_indices = np.full(len(joint_names), -1, dtype=np.int32)
    for jname, pname in joint_parents.items():
        if pname is not None:
            parent_indices[name_to_idx[jname]] = name_to_idx[pname]

    npz_path = os.path.join(args.output_dir, "autorig_data.npz")
    save_dict = dict(
        joint_positions=joint_positions,
        weight_matrix=proxy_weight_matrix,
        joint_names=np.array(joint_names),
        parent_indices=parent_indices,
    )
    if args.coordinate_scale is not None:
        save_dict["coordinate_scale"] = np.float64(args.coordinate_scale)
    np.savez(npz_path, **save_dict)
    print(f"Saved autorig data: {npz_path}")
    print(f"  Joints: {len(joint_names)}")
    print(f"  Proxy mesh vertices: {proxy_weight_matrix.shape[0]}")
    print(f"  Roots at indices: {np.where(parent_indices == -1)[0].tolist()}")
    if args.coordinate_scale is not None:
        print(f"  coordinate_scale: {args.coordinate_scale} "
              f"(multiply positions by this to get PAPR scale)")

    # ── Step 2: Inverse-normalize proxy mesh ─────────────────────
    print("\n" + "=" * 60)
    print("Step 2: Inverse-Normalize Proxy Mesh")
    print("=" * 60)

    input_verts = load_vertices(args.input_mesh)
    bbox_min = input_verts.min(axis=0)
    bbox_max = input_verts.max(axis=0)
    centroid = (bbox_min + bbox_max) / 2.0
    max_extent = (bbox_max - bbox_min).max()

    print(f"Input mesh: {input_verts.shape[0]} vertices")
    print(f"  Centroid: {centroid}, Max extent: {max_extent:.4f}")

    proxy_verts_norm = load_vertices(args.proxy_mesh)
    proxy_verts_world = proxy_verts_norm * max_extent + centroid

    print(f"Proxy mesh: {proxy_verts_norm.shape[0]} vertices")
    print(f"  Normalized bbox: {proxy_verts_norm.min(0)} to {proxy_verts_norm.max(0)}")
    print(f"  World bbox:      {proxy_verts_world.min(0)} to {proxy_verts_world.max(0)}")

    # ── Step 3: Transfer weights to PAPR points ──────────────────
    print("\n" + "=" * 60)
    print("Step 3: Transfer Skinning Weights to PAPR Points")
    print("=" * 60)

    from rigpapr.init_skinning_weights_from_mesh import transfer_skinning_weights

    canonical_points = load_vertices(args.canonical_points) \
        if args.canonical_points.endswith(('.ply', '.obj')) \
        else np.load(args.canonical_points)
    print(f"PAPR canonical points: {canonical_points.shape}")
    print(f"  BBox: {canonical_points.min(0)} to {canonical_points.max(0)}")

    papr_weights = transfer_skinning_weights(
        query_points=canonical_points,
        mesh_vertices=proxy_verts_world,
        mesh_weights=proxy_weight_matrix,
        k=args.k_neighbors,
    )

    weights_path = os.path.join(args.output_dir, "papr_skinning_weights.npz")
    np.savez(weights_path, weights=papr_weights)
    print(f"Saved: {weights_path}  shape={papr_weights.shape}")

    # Diagnostic
    dominant = np.argmax(papr_weights, axis=1)
    unique, counts = np.unique(dominant, return_counts=True)
    print(f"Points assigned to {len(unique)} / {len(joint_names)} joints")

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Autorig data:      {npz_path}")
    print(f"Skinning weights:  {weights_path}")
    print(f"Output directory:  {args.output_dir}")


if __name__ == "__main__":
    main()
