# RigPAPR

**RigPAPR: Rig-Based Animation of Static Neural Point Clouds from a Fixed-Viewpoint Video**
Shichong Peng, Yanshu Zhang, Ke Li. NeurIPS 2026.

[Project page](https://niopeng.github.io/RigPAPR/) · [arXiv](https://arxiv.org/abs/2606.06685) · [Video](https://www.youtube.com/watch?v=R-ivP4SWkkM)

Given a static [PAPR](https://github.com/zvict/papr) reconstruction of a subject and a single fixed-viewpoint video of it
in motion, captured or produced by an image-to-video (I2V) model, RigPAPR recovers a rigged, re-posable 3D asset that
follows the video.
PAPR stores no shape per point and recomposes each pixel at render time from the deformed points' positions, so linear
blend skinning (LBS) applied directly to the points lets the surface re-form naturally at the joints.
We seed the skeleton and skinning weights by auto-rigging a mesh of the point cloud, used only for this step, and then
fit the rig to the video in two phases.
Phase 1 recovers per-frame joint rotations, supervised by the video and its relative monocular depth; Phase 2 adds 2D
point tracks and refines the skinning weights together with joint-level pose corrections.
The fitted asset renders from new viewpoints and takes new poses in a single forward pass, with no per-frame or
per-view fitting.

This repository contains the code for both phases and the scripts that prepare their inputs.
To try it out, start with the T-Rex example below: we provide all of its inputs precomputed, so it
runs without any of the external tools.

## Installation

Tested with Python 3.9, PyTorch 2.4.1 and CUDA 12.4 on Linux with an NVIDIA GPU.

```bash
conda env create -f environment.yml
conda activate rigpapr
# CUDA extensions, built against the installed torch (needs nvcc):
pip install --no-build-isolation "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.8"
```

## Quick start: T-Rex

Download the T-Rex data from
[Google Drive](https://drive.google.com/file/d/1ubOyepKsztrAhCKqHwgOs7WCVr2-Jzor/view?usp=sharing) (`rigpapr_trex_data.tar.gz`, 40 MB) and unpack it into `data/`:

```bash
# or from the command line:
curl -L -o rigpapr_trex_data.tar.gz "https://drive.usercontent.google.com/download?id=1ubOyepKsztrAhCKqHwgOs7WCVr2-Jzor&export=download&confirm=t"
mkdir -p data && tar -xzf rigpapr_trex_data.tar.gz -C data/
sha256sum rigpapr_trex_data.tar.gz   # d02984aaad1b28be67b2093dce5e71afc90b54906cdd653e971fd9ee65485998
```

```
data/trex/
├── papr/         model.pth (trained PAPR model), canonical_points.npy
├── rig/          autorig_data.npz (skeleton), skinning_weights.npz (initial weights)
├── driving/      frames/, masks/, depth_da3/ (monocular depth), tracks_2d_attn.npz (2D tracks)
└── multiview/    the multi-view renders the PAPR model was trained on (used for the camera)
```

Then run both phases:

```bash
bash scripts/run_trex.sh
```

Results go to `output/trex/phase1` and `output/trex/phase2`: per-frame deformed point clouds
(`deformed_XXXX.npy` / `ply/`), the fitted rig (`checkpoint_stepXXXXX.npz`: skinning weights,
joint rotations, root translations) and training-view renders during optimization (`eval/`).

To render the fitted sequence from the driving camera and score it against the driving frames:

```bash
python render.py --config configs/trex/papr.yml \
    --checkpoint data/trex/papr/model.pth --canonical_points data/trex/papr/canonical_points.npy \
    --run_dir output/trex/phase2 --view_idx 27 \
    --gt_frames_dir data/trex/driving/frames --mask_dir data/trex/driving/masks \
    --out_dir output/trex/render
```

This writes `frames/`, `render.mp4` and `metrics.txt` (PSNR and SSIM inside the foreground mask,
LPIPS over the full frame). Pass `--view_idx` or `--cameras_json` to render from another camera.

## Code layout

- `optimize.py`, `render.py`, `train.py`: entry points (Phase 1 + 2, rendering, PAPR pretraining).
- `rigpapr/`: the method (`optimize.py`, `lbs.py`, `losses.py`, `data.py`, `papr_render.py`, `render.py`)
  and the Puppeteer rig transfer (`parse_puppeteer_rig.py`, `init_skinning_weights_from_mesh.py`).
- `models/`, `dataset/`, `utils.py`: PAPR (adapted).
- `scripts/`: input preparation (rig transfer, depth, tracks) and the T-Rex launcher.

## The pipeline

| step | what | tool |
|---|---|---|
| 1 | Train a static PAPR model on multi-view images of the object | `train.py --opt configs/pretrain_default.yml` |
| 2 | Mesh the PAPR point cloud | [OffsetOPT](https://github.com/EnyaHermite/OffsetOPT) (external) |
| 3 | Auto-rig the mesh | [Puppeteer](https://github.com/Seed3D/Puppeteer) (external) |
| 4 | Transfer the skeleton and skinning weights to the PAPR points | `scripts/compute_initial_rig.py` |
| 5 | Monocular depth for the driving video | `scripts/extract_depth_da3.py` ([Depth Anything 3](https://github.com/ByteDance-Seed/depth-anything-3)) |
| 6 | 2D point tracks for the driving video | `scripts/precompute_tracks_attn.py` ([CoTracker 3](https://github.com/facebookresearch/co-tracker)) |
| 7 | Phase 1: per-frame joint rotations | `optimize.py --phase 1` |
| 8 | Phase 2: skinning weights and pose corrections | `optimize.py --phase 2` |
| 9 | Render the fitted sequence (and score it against the driving video) | `render.py` |

The T-Rex demo data contains the outputs of steps 1 to 6, so the quick start runs only steps 7 to 9.

## Your own object


1. **Multi-view capture** in the NeRF-synthetic (Blender) format (`transforms_{train,val,test}.json` + images).
   Set `dataset.path` in `configs/pretrain_default.yml` and train:
   `python train.py --opt configs/pretrain_default.yml`.
   The model is saved to `experiments/<index>/`, with the point cloud in `points_normed.ply`.
2. **Mesh** `points_normed.ply` with OffsetOPT and **rig** the mesh with Puppeteer, following their instructions.
3. **Transfer the rig** to the PAPR points:
   ```bash
   python scripts/compute_initial_rig.py \
       --rig_file <puppeteer>/final_rigging/<name>.txt \
       --proxy_mesh <puppeteer>/skel_results/<name>_mesh.obj \
       --input_mesh <mesh given to Puppeteer> \
       --canonical_points experiments/<index>/points_normed.ply \
       --coordinate_scale 10.0 \
       --output_dir output/<name>/rig
   ```
   Check the skeleton: an auto-rig can come out with a joint attached to the wrong parent.
4. **Driving video.** The camera must be fixed and must match one of the PAPR model's cameras
   (`--view_idx`, an index into the test split) or be given as `--cameras_json`. Save frames as
   `frame_XXXX.png` with foreground masks of the same names (for example with
   [SAM 2](https://github.com/facebookresearch/sam2)). One frame is the canonical frame: the pose the
   PAPR model was trained in.
5. **Depth and tracks**, each in its own environment:
   ```bash
   python scripts/extract_depth_da3.py --image_dir <frames> --output_dir <depth>        # DA3 env
   python scripts/precompute_tracks_attn.py --config <config> --checkpoint <model.pth> \
       --canonical_points <points> --gt_frames_dir <frames> --canonical_frame <c> \
       --start_frame <s> --end_frame <e> --view_idx <v> --output <tracks.npz>            # CoTracker installed
   ```
6. **Optimize.** Copy `scripts/run_trex.sh` and change the paths, the camera and the frame range.

## License

Apache License 2.0 (see `LICENSE`). This repository includes code from PAPR (MIT), PlenOctrees (BSD-2),
LPIPS (BSD-2), pytorch-ssim (MIT) and SNP (MIT); see `NOTICE`.
The external tools have their own licenses: Depth Anything 3 (Apache-2.0), Puppeteer (Apache-2.0),
CoTracker 3 (CC BY-NC 4.0, non-commercial) and OffsetOPT (no license file).

## Citation

```bibtex
@inproceedings{peng2026rigpapr,
  title     = {RigPAPR: Rig-Based Animation of Static Neural Point Clouds from a Fixed-Viewpoint Video},
  author    = {Peng, Shichong and Zhang, Yanshu and Li, Ke},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
