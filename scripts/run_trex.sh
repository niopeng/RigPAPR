#!/usr/bin/env bash
# ==============================================================================
# RigPAPR on T-Rex: Phase 1 (per-frame joint rotations) -> Phase 2 (skinning
# weights + pose corrections, with a refine stage from REFINE_START on).
#
# Expects the T-Rex bundle under data/trex/ (see README). Outputs go to
# output/trex/{phase1,phase2}.
#
#   bash scripts/run_trex.sh                 # both phases
#   STAGES=phase2 bash scripts/run_trex.sh   # reuse an existing Phase 1
# ==============================================================================

set -euo pipefail

PYTHON="${PYTHON:-python}"
STAGES="${STAGES:-phase1 phase2}"
OUT="${OUT:-output/trex}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

DATA=data/trex
COMMON=(
    --config configs/trex/papr.yml
    --checkpoint "$DATA/papr/model.pth"
    --canonical_points "$DATA/papr/canonical_points.npy"
    --autorig_data "$DATA/rig/autorig_data.npz"
    --init_weights "$DATA/rig/skinning_weights.npz"
    --gt_frames_dir "$DATA/driving/frames"
    --mask_dir "$DATA/driving/masks"
    --delta_depth_dir "$DATA/driving/depth_da3"
    --lambda_delta_depth 1.0
    --view_idx 27
    --canonical_frame 40 --start_frame 40 --end_frame 0
    --loss_type l1_ssim --lambda_lpips 0.01
    --patch_h 200 --patch_w 200
)

for stage in $STAGES; do
    case "$stage" in
    phase1)
        "$PYTHON" optimize.py --phase 1 "${COMMON[@]}" \
            --frame_skip 1 \
            --steps_per_frame 500 --steps_first_frame 1000 \
            --min_steps 200 --max_steps 2000 \
            --early_stop_thresh 1e-5 --early_stop_patience 3 \
            --lr 5e-3 --lr_end 1e-4 \
            --lambda_rot_reg 0.1 \
            --eval_every 500 \
            --output_dir "$OUT/phase1"
        ;;
    phase2)
        "$PYTHON" optimize.py --phase 2 "${COMMON[@]}" \
            --phase1_dir "$OUT/phase1" \
            --track_data "$DATA/driving/tracks_2d_attn.npz" \
            --lambda_track 0.005 --track_subsample 1000 \
            --total_steps 50000 --refine_start 30000 --warmup_steps 0 \
            --lr_correction 3e-4 --lr_weights 1e-2 --lr_root_trans 1e-4 --lr_end 1e-5 \
            --mlp_hidden_dim 128 --mlp_num_freqs 10 \
            --lambda_correction 5e-4 --lambda_temporal 0.1 --lambda_rot_reg 0.01 \
            --lambda_dist 500 --dist_k 8 --knn_k 8 \
            --lambda_smooth 0.03 --lambda_sparse 0.01 \
            --eval_every 500 --save_every 5000 \
            --output_dir "$OUT/phase2"
        ;;
    *) echo "unknown stage: $stage" >&2; exit 2 ;;
    esac
done
