"""
Parse Puppeteer rig text format into structured numpy arrays.

Rig file format (rexy.txt):
    joints joint0 -0.409 1.698 0.037    # joint name + XYZ world space
    ...                                   # 64 joints
    root joint17                          # root joint
    hier joint35 joint34                  # parent child
    ...                                   # hierarchy edges
    skin 0 joint28 0.5 joint29 0.5       # vertex_id joint weight ...
    ...                                   # 25301 skin entries

Usage:
    python -m rigpapr.parse_puppeteer_rig \
      --rig_file <puppeteer_output>/final_rigging/<name>.txt \
      --output output/<name>/autorig_data.npz
"""
import os
import argparse
import numpy as np


def parse_puppeteer_rig(rig_path):
    """
    Parse a Puppeteer rig text file.

    Returns:
        joint_names: list[str]              length J
        joint_positions: (J, 3) float32     Blender world space
        joint_parents: dict[str, str|None]  parent name or None for root
        weight_matrix: (V, J) float32       rows sum to 1; indexed by proxy mesh vertex ID
    """
    joint_names = []
    joint_positions_list = []
    joint_name_to_idx = {}
    root_name = None
    hier_edges = []  # list of (parent_name, child_name)
    skin_entries = {}  # vertex_id -> list of (joint_name, weight)

    with open(rig_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            keyword = tokens[0]

            if keyword == 'joints':
                name = tokens[1]
                x, y, z = float(tokens[2]), float(tokens[3]), float(tokens[4])
                joint_name_to_idx[name] = len(joint_names)
                joint_names.append(name)
                joint_positions_list.append([x, y, z])

            elif keyword == 'root':
                root_name = tokens[1]

            elif keyword == 'hier':
                parent_name = tokens[1]
                child_name = tokens[2]
                hier_edges.append((parent_name, child_name))

            elif keyword == 'skin':
                vid = int(tokens[1])
                pairs = []
                i = 2
                while i + 1 < len(tokens):
                    jname = tokens[i]
                    w = float(tokens[i + 1])
                    pairs.append((jname, w))
                    i += 2
                skin_entries[vid] = pairs

    num_joints = len(joint_names)
    joint_positions = np.array(joint_positions_list, dtype=np.float32)

    # Build parent map
    joint_parents = {name: None for name in joint_names}
    for parent_name, child_name in hier_edges:
        joint_parents[child_name] = parent_name
    # Verify root
    if root_name is not None:
        assert joint_parents[root_name] is None, \
            f"Root joint {root_name} has parent {joint_parents[root_name]}"

    # Build weight matrix
    num_verts = max(skin_entries.keys()) + 1 if skin_entries else 0
    weight_matrix = np.zeros((num_verts, num_joints), dtype=np.float32)
    for vid, pairs in skin_entries.items():
        for jname, w in pairs:
            jidx = joint_name_to_idx[jname]
            weight_matrix[vid, jidx] = w

    # Verify rows sum to ~1
    row_sums = weight_matrix.sum(axis=1)
    bad_rows = np.where(np.abs(row_sums - 1.0) > 1e-3)[0]
    if len(bad_rows) > 0:
        print(f"WARNING: {len(bad_rows)} rows have weight sums far from 1.0")
        print(f"  Example: vertex {bad_rows[0]}, sum = {row_sums[bad_rows[0]]:.6f}")

    print(f"Parsed rig: {num_joints} joints, {num_verts} skinned vertices")
    print(f"  Joint positions shape: {joint_positions.shape}")
    print(f"  Weight matrix shape:   {weight_matrix.shape}")
    print(f"  Root joint: {root_name}")
    print(f"  Weight row sums: min={row_sums.min():.6f}, max={row_sums.max():.6f}")

    return joint_names, joint_positions, joint_parents, weight_matrix


def main():
    parser = argparse.ArgumentParser(description="Parse Puppeteer rig file")
    parser.add_argument("--rig_file", type=str, required=True,
                        help="Path to Puppeteer rig .txt file")
    parser.add_argument("--output", type=str, required=True,
                        help="Output .npz file path")
    args = parser.parse_args()

    joint_names, joint_positions, joint_parents, weight_matrix = \
        parse_puppeteer_rig(args.rig_file)

    # Convert joint_parents dict → int32 array (-1 = root)
    name_to_idx = {n: i for i, n in enumerate(joint_names)}
    parent_indices = np.full(len(joint_names), -1, dtype=np.int32)
    for jname, pname in joint_parents.items():
        if pname is not None:
            parent_indices[name_to_idx[jname]] = name_to_idx[pname]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez(
        args.output,
        joint_positions=joint_positions,
        weight_matrix=weight_matrix,
        joint_names=np.array(joint_names),
        parent_indices=parent_indices,
    )
    print(f"Saved: {args.output}")
    print(f"  joint_positions: {joint_positions.shape}")
    print(f"  weight_matrix:   {weight_matrix.shape}")
    print(f"  parent_indices: {parent_indices.shape}, roots at "
          f"{np.where(parent_indices == -1)[0].tolist()}")


if __name__ == "__main__":
    main()
