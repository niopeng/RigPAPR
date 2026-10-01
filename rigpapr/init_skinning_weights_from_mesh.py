"""
Transfer skinning weights from mesh vertices to query points (e.g., PAPR point cloud)
using K-nearest-neighbor inverse-distance weighting.

Standalone CLI:
    python -m rigpapr.init_skinning_weights_from_mesh \
      --mesh_weights output/lbs_experiment/blender/skinning_weights.npz \
      --query_points output/lbs_experiment/papr_canonical_world.npy \
      --output output/lbs_experiment/papr_skinning_weights.npz --k 6
"""
import numpy as np
from scipy.spatial import cKDTree
import argparse


def transfer_skinning_weights(query_points, mesh_vertices, mesh_weights, k=6):
    """
    Transfer skinning weights from mesh vertices to query points via KNN.

    Args:
        query_points: (N, 3) float32 - points to assign weights to
        mesh_vertices: (V, 3) float32 - mesh vertices with known weights
        mesh_weights: (V, B) float32 - per-vertex bone weights (rows sum to 1)
        k: int - number of nearest neighbors

    Returns:
        (N, B) float32 - transferred weights (rows sum to 1)
    """
    tree = cKDTree(mesh_vertices)
    distances, indices = tree.query(query_points, k=k)

    # Inverse-distance weighting (avoid div by zero)
    eps = 1e-8
    inv_dist = 1.0 / (distances + eps)
    inv_dist_sum = inv_dist.sum(axis=1, keepdims=True)
    idw = inv_dist / inv_dist_sum  # (N, k)

    # Gather neighbor weights and compute weighted average
    neighbor_weights = mesh_weights[indices]  # (N, k, B)
    transferred = np.einsum('nk,nkb->nb', idw, neighbor_weights)  # (N, B)

    # Re-normalize rows
    row_sums = transferred.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    transferred = transferred / row_sums

    return transferred.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Transfer skinning weights via KNN")
    parser.add_argument("--mesh_weights", type=str, required=True,
                        help="Path to skinning_weights.npz (vertex_positions, weight_matrix)")
    parser.add_argument("--query_points", type=str, required=True,
                        help="Path to query points .npy (N, 3)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for .npz")
    parser.add_argument("--k", type=int, default=6, help="Number of nearest neighbors")
    args = parser.parse_args()

    # Load mesh skinning data
    data = np.load(args.mesh_weights)
    mesh_vertices = data["vertex_positions"]  # (V, 3)
    mesh_weights = data["weight_matrix"]      # (V, B)

    # Load query points
    query_points = np.load(args.query_points)  # (N, 3)
    print(f"Mesh vertices: {mesh_vertices.shape}, Query points: {query_points.shape}")
    print(f"Mesh weights: {mesh_weights.shape}, K={args.k}")

    # Transfer
    transferred = transfer_skinning_weights(query_points, mesh_vertices, mesh_weights, k=args.k)
    print(f"Transferred weights: {transferred.shape}")

    # Save
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez(args.output, weights=transferred)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
