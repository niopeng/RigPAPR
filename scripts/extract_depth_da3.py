"""
Per-frame monocular depth for the driving video with Depth Anything 3 (DA3).

Each frame is predicted on its own (no multi-view fusion), which suits a fixed
camera watching a moving object. Output: <output_dir>/<frame>.npz with 'depth'
(and 'conf' when the model provides it), plus a <frame>_vis.png preview. Pass
<output_dir> to the optimizer as --delta_depth_dir.

Run it in an environment with DA3 installed
(https://github.com/ByteDance-Seed/depth-anything-3):

    python scripts/extract_depth_da3.py \
        --image_dir data/trex/driving/frames \
        --output_dir data/trex/driving/depth_da3
"""

import os
import glob
import argparse
import numpy as np
from PIL import Image
from depth_anything_3.api import DepthAnything3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory of driving frames (*.png)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="depth-anything/DA3MONO-LARGE",
                        help="HuggingFace model ID. Options: depth-anything/DA3MONO-LARGE, "
                             "depth-anything/DA3METRIC-LARGE, depth-anything/DA3-LARGE-1.1")
    parser.add_argument("--process_res", type=int, default=504)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model: {args.model}")
    model = DepthAnything3.from_pretrained(args.model)
    model = model.to("cuda")

    frames = sorted(glob.glob(os.path.join(args.image_dir, "*.png")))
    print(f"Found {len(frames)} frames")

    for i, frame_path in enumerate(frames):
        fname = os.path.splitext(os.path.basename(frame_path))[0]

        # Run single-image inference
        pred = model.inference(
            image=[frame_path],
            process_res=args.process_res,
        )

        d_raw = pred.depth[0]
        depth = d_raw.cpu().numpy() if hasattr(d_raw, 'cpu') else np.asarray(d_raw)
        c_raw = pred.conf[0] if pred.conf is not None else None
        conf = c_raw.cpu().numpy() if c_raw is not None and hasattr(c_raw, 'cpu') else (np.asarray(c_raw) if c_raw is not None else None)

        # Save as npz
        save_dict = {"depth": depth}
        if conf is not None:
            save_dict["conf"] = conf
        np.savez(os.path.join(args.output_dir, f"{fname}.npz"), **save_dict)

        # Save depth visualization as png
        d = depth.copy()
        d_min, d_max = d.min(), d.max()
        if d_max - d_min > 0:
            d = (d - d_min) / (d_max - d_min)
        d = (d * 255).astype(np.uint8)
        Image.fromarray(d).save(os.path.join(args.output_dir, f"{fname}_vis.png"))

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(frames)}] {fname}")

    print(f"Done. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
