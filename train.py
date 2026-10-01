import yaml
import argparse
import torch
from torch import autocast
import os
import shutil
import zipfile
import matplotlib.pyplot as plt
import numpy as np
import random
import copy
import bisect
import time
import sys
import io
import imageio
from PIL import Image
from utils import *
from dataset import get_dataset, get_loader
from models import get_model, get_loss
import torchvision as tv
import open3d as o3d

# Import wandb for logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with 'pip install wandb' to enable logging.")


def parse_args():
    parser = argparse.ArgumentParser(description="PAPR")
    parser.add_argument('--opt', type=str, default="", help='Option file path')
    parser.add_argument('--resume', type=int, default=0, help='Resume training')
    return parser.parse_args()


def save_points(file_path, points):
    if file_path.endswith('.xyz'):
        np.savetxt(file_path, points, delimiter=' ')
    elif file_path.endswith('.ply'):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(file_path, pcd)
    else:
        raise ValueError(f"Unsupported file format: {file_path.split('.')[-1]}")


def eval_step(steps, model, device, dataset, eval_dataset, batch, loss_fn, train_out, args, train_losses, eval_losses, eval_psnrs, pt_lrs, attn_lrs, train_patch, nearest_sp=None):
    step = steps[-1]
    train_img_idx = batch['idx']
    train_data = dataset.get_full_img(train_img_idx[0])
    train_img = train_data['image']
    train_rayd = train_data['rayd']
    train_rayo = train_data['rayo']
    # train_surface_points = train_data['surface_points']
    eval_data = eval_dataset.get_full_img(args.eval.img_idx)
    img = eval_data['image']
    rayd = eval_data['rayd']
    rays_d_no_norm = eval_data['rays_d_no_norm']
    rayo = eval_data['rayo']
    mask = eval_data['mask']
    pix_coords = eval_data['pix_coords']
    if 'surface_points' in eval_data:
        surface_points = eval_data['surface_points']
        normals = eval_data['normals']
    else:
        surface_points = None
        normals = None
    c2w = eval_dataset.get_c2w([args.eval.img_idx])

    N, H, W, _ = rayd.shape

    rayo = rayo.to(device)
    rayd = rayd.to(device)
    rays_d_no_norm = rays_d_no_norm.to(device)
    img = img.to(device)
    mask = mask.to(device)
    c2w = c2w.to(device)
    pix_coords = pix_coords.to(device)
    if surface_points is not None:
        surface_points = surface_points.to(device)
        normals = normals.to(device)
        print("surface_points:", surface_points.shape, "normals:", normals.shape)

    final_select_k = model.select_k
    if args.geoms.points.select_k_rnd:
        final_select_k = max(final_select_k, args.geoms.points.select_k_max)
    if args.models.attn.select_prop_scores:
        final_select_k = min(final_select_k, args.models.attn.select_prop_scores_topk)
    if args.models.attn.select_v_scores:
        final_select_k = min(final_select_k, args.models.attn.select_v_scores_topk)
        if args.models.attn.select_v_scores_rnd:
            final_select_k = max(final_select_k, args.models.attn.select_v_scores_topk_max)
    selected_points = torch.zeros(1, H, W, final_select_k, 3)

    bkg_seq_len_attn = 0
    if model.args.geoms.no_additional_bkg:
        bkg_seq_len_attn = 0  # No background token when no_additional_bkg is True
    elif model.bkg_feats is not None and not model.args.geoms.points.use_bkg_points and not model.args.models.attn.append_bkg_points:
        bkg_seq_len_attn = model.bkg_feats.shape[0]
    
    append_bkg_points_size = 1 if (model.args.models.attn.append_bkg_points and not model.args.geoms.no_additional_bkg) else 0
    
    feat_map_dim = model.feat_map_dim
    feature_map = torch.empty(N, H, W, 1, feat_map_dim).to(device)
    # When append_bkg_points is True, the appended point is included in final_select_k, so we add append_bkg_points_size
    # When bkg_feats is used (and not append_bkg_points), we add bkg_seq_len_attn
    attn = torch.empty(N, H, W, final_select_k + append_bkg_points_size + bkg_seq_len_attn, 1).to(device)
    
    # Accumulator for embedv (needed for foreground/background separation with cubemap_feature_textures)
    full_embedv = None
    if model.cubemap_feature_textures is not None and model.args.models.attn.append_bkg_points:
        # embedv has shape (N, tile_H, tile_W, K+1, C) where C is output_dim_v
        output_dim_v = model.args.models.attn.output_dim_v
        full_embedv = torch.empty(N, H, W, final_select_k + append_bkg_points_size, output_dim_v).to(device)
    select_inds = torch.empty(N, H, W, final_select_k, dtype=torch.long).to(device)
    min_d2r = torch.empty(N, H, W).to(device)
    min_pd = torch.empty(N, H, W).to(device)
    
    cur_sp = torch.empty(N, H, W, 3).to(device)
    hit_pred = torch.empty(N, H, W).to(device)
    
    sphere_intersection = None
    if model.args.models.attn.append_bkg_points:
        sphere_intersection = torch.empty(N, H, W, 3).to(device)

    points = model.get_points()
    
    with torch.no_grad():
        # Handle cases where max_height/max_width >= image dimensions or invalid values
        tile_height = min(args.eval.max_height, H) if args.eval.max_height > 0 else H
        tile_width = min(args.eval.max_width, W) if args.eval.max_width > 0 else W
        
        for height_start in range(0, H, tile_height):
            for width_start in range(0, W, tile_width):
                height_end = min(height_start + tile_height, H)
                width_end = min(width_start + tile_width, W)

                if surface_points is not None:
                    cur_surface_points = surface_points[:, height_start:height_end, width_start:width_end]
                    cur_normals = normals[:, height_start:height_end, width_start:width_end]
                else:
                    cur_surface_points = None
                    cur_normals = None

                feature_map[:, height_start:height_end, width_start:width_end, :, :], \
                attn[:, height_start:height_end, width_start:width_end, :, :], _, cur_hit_pred = model.evaluate(rayo, \
                                                                                                        rayd[:, height_start:height_end, width_start:width_end], \
                                                                                                        c2w, \
                                                                                                        pix_coords[height_start:height_end, width_start:width_end], \
                                                                                                        step=step, \
                                                                                                        surface_points=cur_surface_points, \
                                                                                                        normals=cur_normals, \
                                                                                                        rays_d_no_norm=rays_d_no_norm[:, height_start:height_end, width_start:width_end])

                select_inds[:, height_start:height_end, width_start:width_end, :] = model.select_k_ind
                min_d2r[:, height_start:height_end, width_start:width_end] = model.proximity_attn.d2r.min(dim=-2)[0].squeeze(-1)
                min_pd[:, height_start:height_end, width_start:width_end] = model.proximity_attn.pd.min(dim=-2)[0].squeeze(-1)
                if model.args.models.attn.append_bkg_points:
                    sphere_intersection[:, height_start:height_end, width_start:width_end, :] = model.proximity_attn.sphere_intersection.squeeze(-2)
                
                # Accumulate embedv for foreground/background separation
                if full_embedv is not None and hasattr(model, 'last_embedv') and model.last_embedv is not None:
                    full_embedv[:, height_start:height_end, width_start:width_end, :, :] = model.last_embedv
                
                selected_points[:, height_start:height_end, width_start:width_end, :, :] = points[model.select_k_ind]
                
                # if model.bkg_feats is not None and not model.args.geoms.points.use_bkg_points:
                #     topk_attn = attn[:, height_start:height_end, width_start:width_end, :-1, :]
                # else:
                #     topk_attn = attn[:, height_start:height_end, width_start:width_end, :, :]
                cur_sp[:, height_start:height_end, width_start:width_end, :] = model.get_surface_points(model.select_k_ind, points, rayo, 
                                                                                                        rayd[:, height_start:height_end, width_start:width_end], 
                                                                                                        attn[:, height_start:height_end, width_start:width_end, :, :], 
                                                                                                        'position').squeeze(-2)
                if cur_hit_pred is not None:
                    hit_pred[:, height_start:height_end, width_start:width_end] = cur_hit_pred
        
        int_foreground_rgb = None
        if args.models.unet.use:
            if model.args.models.unet.double_channel:
                feature_map = torch.cat([feature_map, feature_map], dim=-1)
            foreground_rgb = model.unet(feature_map.squeeze(-2).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).unsqueeze(-2)   # (N, H, W, 1, 3)
        elif args.models.fused_feature_mlp.use:
            feature_map = model.fused_feature_encoder(feature_map.reshape(N*H*W, -1))
            foreground_rgb = model.fused_feature_mlp(feature_map).reshape(N, H, W, 1, 3)
            feature_map = feature_map.reshape(N, H, W, -1)
        else:
            foreground_rgb = feature_map
            
        cur_sp2cam = torch.norm(cur_sp - rayo.reshape(N, 1, 1, 3), dim=-1).squeeze(0)
        if step >= 0:
            if nearest_sp.sum() == 0 or step < 50000:
                nearest_sp = cur_sp2cam
            else:
                nearest_sp = torch.minimum(nearest_sp, cur_sp2cam)
        
        bkg_attn = None
        bkg_color = model.bkg_feats
        bkg_color = bkg_color * model.bkg_scaler.squeeze(0)
        if model.args.geoms.no_additional_bkg:
            # No special background treatment - just use foreground directly
            rgb = foreground_rgb.squeeze(-2)
            bkg_attn = torch.zeros(N, H, W, 1).to(device)
        elif model.bkg_feats is not None and not model.args.geoms.points.use_bkg_points and not model.args.models.attn.append_bkg_points:
            
            bkg_attn = attn[..., -1:, :]
            if args.geoms.background.use_dumb_constant:
                bkg_attn = (bkg_attn * model.dumb_constant).clamp(0, 1)
            bkg_attn = model.bkg_attn_act((bkg_attn + args.models.bkg_attn_shift) * args.models.bkg_attn_scale)

            if args.mask_bkg_rays_thresh > 0:
                fg_mask = min_d2r[..., None] < args.mask_bkg_rays_thresh
                bkg_attn_masked = torch.ones_like(bkg_attn)
                bkg_attn_masked[fg_mask] = bkg_attn[fg_mask]
            else:
                bkg_attn_masked = bkg_attn
                
            if model.args.models.bkg_attn_deno_thresh > 0:
                bkg_deno = bkg_attn_masked.clone().detach()
                bkg_deno[bkg_deno < model.args.models.bkg_attn_deno_thresh] = 1
                bkg_attn_masked = bkg_attn_masked / bkg_deno
            if model.args.models.bkg_attn_sigmoid_scale > 0:
                bkg_attn_masked = torch.sigmoid(bkg_attn_masked * model.args.models.bkg_attn_sigmoid_scale)
                
            # bkg_mask = bkg_attn.squeeze()
            # if args.rnd_background:
            #     bkg_attn_masked[bkg_attn_masked < 0.5] = 0
            #     bkg_attn_masked[bkg_attn_masked >= 0.5] = 1

            # if args.models.normalize_topk_attn:
            #     # bkg_attn_masked = (bkg_attn_masked * model.dumb_constant).clamp(0, 1)
            #     rgb = foreground_rgb * (1 - bkg_attn_masked) + bkg_color.expand(N, H, W, -1, -1) * bkg_attn_masked
            # else:
            #     rgb = foreground_rgb + bkg_color.expand(N, H, W, -1, -1) * bkg_attn_masked
            
            # Use cubemap textures for background if enabled
            cubemap_bkg_color = model.get_cubemap_background_color(rayo, rayd, cur_step=step)
            if cubemap_bkg_color is not None:
                rgb = foreground_rgb * (1 - bkg_attn_masked) + cubemap_bkg_color.unsqueeze(-2) * bkg_attn_masked
            else:
                rgb = foreground_rgb * (1 - bkg_attn_masked) + bkg_color.expand(N, H, W, -1, -1) * bkg_attn_masked
            rgb = rgb.squeeze(-2)
        else:
            rgb = foreground_rgb.squeeze(-2)
            
            # Use cubemap textures for background if enabled
            cubemap_bkg_color = model.get_cubemap_background_color(rayo, rayd, cur_step=step)
            effective_bkg_color = cubemap_bkg_color if cubemap_bkg_color is not None else bkg_color.expand(N, H, W, -1)
            
            if model.args.geoms.points.use_bkg_points and model.args.models.attn.append_bkg_points:
                topk_attn = attn[..., :-1, :]
                bkg_attn = attn[..., -1, :]
                selected_bkg_points_mask = model.bkg_points_mask[select_inds].expand(-1, -1, -1, -1, 1)
                bkg_attn = torch.sum(topk_attn * (1 - selected_bkg_points_mask.float()), dim=3) + bkg_attn
                if model.args.geoms.points.bkg_points_alpha_blend or model.args.models.attn.append_bkg_points_alpha_blend:
                    rgb = rgb * (1 - bkg_attn) + effective_bkg_color * bkg_attn
            elif model.args.geoms.points.use_bkg_points:
                bkg_color = model.bkg_feats
                bkg_color = bkg_color * model.bkg_scaler.squeeze(0)
                effective_bkg_color = cubemap_bkg_color if cubemap_bkg_color is not None else bkg_color.expand(N, H, W, -1)
                selected_bkg_points_mask = model.bkg_points_mask[select_inds].expand(-1, -1, -1, -1, 1)
                bkg_attn = torch.sum(attn * (1 - selected_bkg_points_mask.float()), dim=3)
                if model.args.geoms.points.bkg_points_alpha_blend:
                    rgb = rgb * (1 - bkg_attn) + effective_bkg_color * bkg_attn
            elif model.args.models.attn.append_bkg_points:
                bkg_color = model.bkg_feats
                bkg_color = bkg_color * model.bkg_scaler.squeeze(0)
                effective_bkg_color = cubemap_bkg_color if cubemap_bkg_color is not None else bkg_color.expand(N, H, W, -1)
                bkg_attn = attn[..., -1, :]  # Extract attention for appended background point
                if model.args.models.attn.append_bkg_points_alpha_blend:
                    rgb = rgb * (1 - bkg_attn) + effective_bkg_color * bkg_attn
                
        rgb = model.last_act(rgb)
        rgb = torch.clamp(rgb, 0, 1)

        eval_loss = loss_fn(rgb, img)
        eval_psnr = -10. * np.log(((rgb - img)**2).mean().item()) / np.log(10.)
        
        # Calculate intrinsic loss if available
        eval_int_loss = 0.0
        eval_int_psnr = 0.0

        model.clear_grad()

    eval_losses.append(eval_loss.item())
    eval_psnrs.append(eval_psnr.item())

    print("Eval step:", step, "train_loss:", train_losses[-1], "eval_loss:", eval_losses[-1], "eval_psnr:", eval_psnrs[-1])
    
    # Log to wandb if enabled
    if WANDB_AVAILABLE and wandb.run is not None:
        log_dict = {
            "evaluation/train_loss": train_losses[-1],
            "evaluation/eval_loss": eval_losses[-1], 
            "evaluation/eval_psnr": eval_psnrs[-1],
            "optimization/pt_lr": pt_lrs[-1] if pt_lrs else 0,
            "optimization/attn_lr": attn_lrs[-1] if attn_lrs else 0,
            "step": step,
        }
        
        wandb.log(log_dict)

    log_dir = os.path.join(args.save_dir, args.index)
    os.makedirs(log_dir, exist_ok=True)
    if args.eval.save_fig:
        os.makedirs(os.path.join(log_dir, "train_main_plots"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, "train_pcd_plots"), exist_ok=True)
        if model.cubemap_textures is not None or model.cubemap_feature_textures is not None or model.cubemap_depth_textures is not None:
            os.makedirs(os.path.join(log_dir, "train_cubemap_plots"), exist_ok=True)

        coord_scale = args.dataset.coord_scale
        pt_plot_scale = 1.0 * coord_scale
        if "Barn" in args.dataset.path:
            pt_plot_scale *= 1.8
        if "Family" in args.dataset.path:
            pt_plot_scale *= 0.5    
        if "mvhuman" in args.dataset.type:
            pt_plot_scale *= 100

        # calculate depth, weighted sum the distances from top K points to image plane
        od = -rayo
        D = torch.sum(od * rayo)
        dists = torch.abs(torch.sum(selected_points.to(od.device) * od, -1) - D) / torch.norm(od)
        if model.args.geoms.no_additional_bkg:
            # No background token - dists stays as is
            pass
        elif model.bkg_feats is not None and not model.args.geoms.points.use_bkg_points and not model.args.models.attn.append_bkg_points:
            dists = torch.cat([dists, torch.ones(N, H, W, model.bkg_feats.shape[0]).to(dists.device) * 0], dim=-1)
        elif model.args.models.attn.append_bkg_points:
            dists = torch.cat([dists, torch.norm(sphere_intersection.to(od.device) - rayo.to(od.device), dim=-1, keepdim=True)], dim=-1)
        cur_depth = (torch.sum(attn.squeeze(-1).to(od.device) * dists, dim=-1)).detach().cpu()

        train_tgt_rgb = train_img.squeeze().cpu().numpy().astype(np.float32)
        train_tgt_patch = train_patch[0].detach().cpu().numpy().astype(np.float32)
        train_pred_patch = train_out[0].detach().cpu().numpy().astype(np.float32) if torch.is_tensor(train_out[0]) else np.array(train_out[0], dtype=np.float32)
        test_tgt_rgb = img.squeeze().cpu().numpy().astype(np.float32)
        test_pred_rgb = rgb.squeeze().detach().cpu().numpy().astype(np.float32)
        test_pred_int_rgb = None
        
        # Compute foreground/background RGB when cubemap is used
        test_foreground_rgb = None
        test_background_rgb = None
        if (model.cubemap_textures is not None or model.cubemap_feature_textures is not None) and model.args.models.attn.append_bkg_points:
            # When cubemap_feature_textures is used, the background is the last dimension of embedv
            if model.cubemap_feature_textures is not None and full_embedv is not None:
                # full_embedv shape: (N, H, W, K+1, C), attn shape: (N, H, W, K+1, 1)
                # Foreground: weighted sum of embedv[..., :-1, :] with attn[..., :-1, :]
                foreground_attn = attn[..., :-1, :]
                foreground_embedv = full_embedv[..., :-1, :]
                foreground_fused = torch.sum(foreground_embedv * foreground_attn, dim=3)  # (N, H, W, C)
                # Background: embedv[..., -1, :] (the contribution from the virtual background point)
                background_fused = full_embedv[..., -1, :]
                
                # Apply last_act and clamp
                foreground_fused = model.last_act(foreground_fused)
                foreground_fused = torch.clamp(foreground_fused, 0, 1)
                background_fused = model.last_act(background_fused)
                background_fused = torch.clamp(background_fused, 0, 1)
                
                test_foreground_rgb = foreground_fused.squeeze().detach().cpu().numpy().astype(np.float32)
                test_background_rgb = background_fused.squeeze().detach().cpu().numpy().astype(np.float32)
            elif model.cubemap_textures is not None:
                # When only RGB cubemap is used (with alpha blending), get the cubemap color as background
                cubemap_bkg = model.get_cubemap_background_color(rayo, rayd, cur_step=step)
                if cubemap_bkg is not None:
                    test_background_rgb = cubemap_bkg.squeeze(0).detach().cpu().numpy().astype(np.float32)
                    # Foreground is the rgb without background blending (approximate by using the raw feature_map output)
                    if bkg_attn is not None:
                        fg_contribution = 1 - bkg_attn.reshape(N, H, W, 1)
                        test_foreground_rgb = (rgb * fg_contribution).squeeze().detach().cpu().numpy().astype(np.float32)
                        
        points_np = model.get_points().detach().cpu().numpy()
        depth = cur_depth.squeeze().numpy().astype(np.float32)
        center_ray_select = torch.cat([select_inds.squeeze()[H//4-10:H//4+10, W//2-10:W//2+10, :].reshape(-1), select_inds.squeeze()[H//2-10:H//2+10, W//2-10:W//2+10, :].reshape(-1)], dim=-1).unique(sorted=False).detach().cpu().numpy()
        min_d2r = min_d2r.squeeze().cpu().numpy().astype(np.float32)
        # min_d2r[mask.squeeze().cpu().numpy() < 0.5] = min_d2r.min()
        min_d2r = (min_d2r - min_d2r.min()) / (min_d2r.max() - min_d2r.min() + 1e-8)
        min_pd = min_pd.squeeze().cpu().numpy().astype(np.float32)
        # min_pd[mask.squeeze().cpu().numpy() < 0.5] = min_pd.min()
        min_pd = (min_pd - min_pd.min()) / (min_pd.max() - min_pd.min() + 1e-8)
        points_influ_scores_np = model.get_influ_scores().squeeze().detach().cpu().numpy()
        points_alpha_np = model.get_points_alpha.squeeze().detach().cpu().numpy() if model.points_alpha is not None else None
        if cur_hit_pred is not None:
            test_pred_int_rgb = hit_pred.squeeze().detach().cpu().numpy().astype(np.float32)
        if bkg_attn is not None:
            test_pred_int_rgb = bkg_attn.squeeze().detach().cpu().numpy().astype(np.float32)

        # main plot
        nearest_sp_np = nearest_sp.detach().cpu().numpy().astype(np.float32) if nearest_sp is not None else None
        main_plot = get_training_main_plot(args.index, steps, train_tgt_rgb, train_tgt_patch, train_pred_patch, test_tgt_rgb, test_pred_rgb, train_losses, 
                                           eval_losses, points_np, pt_plot_scale, depth, min_pd, pt_lrs, attn_lrs, eval_psnrs, points_influ_scores_np, min_d2r, test_pred_int_rgb)
        save_name = os.path.join(log_dir, "train_main_plots", "iter_%d.png" % (step))
        main_plot.save(save_name)

        # point cloud plot
        ro = train_rayo.squeeze().detach().cpu().numpy()
        rd = train_rayd.squeeze().detach().cpu().numpy()
        
        pcd_plot = get_training_pcd_plot(args.index, steps[-1], ro, rd, points_np, args.dataset.coord_scale, pt_plot_scale, center_ray_select, points_influ_scores_np, points_alpha_np)
        save_name = os.path.join(log_dir, "train_pcd_plots", "iter_%d.png" % (step))
        pcd_plot.save(save_name)

        # cubemap plot (if cubemap is used and learned)
        if model.cubemap_textures is not None:
            cubemap_plot = get_cubemap_plot(args.index, step, model.cubemap_textures, activation=model.cubemap_activation)
            save_name = os.path.join(log_dir, "train_cubemap_plots", "iter_%d.png" % (step))
            cubemap_plot.save(save_name)
        
        # feature cubemap plot (if feature cubemap is used)
        if model.cubemap_feature_textures is not None:
            os.makedirs(os.path.join(log_dir, "train_cubemap_feature_plots"), exist_ok=True)
            # For feature cubemap, we can visualize the first 3 channels or the norm
            cubemap_feature_plot = get_cubemap_plot(args.index, step, model.cubemap_feature_textures, activation=model.cubemap_feature_activation, is_feature_map=True)
            save_name = os.path.join(log_dir, "train_cubemap_feature_plots", "iter_%d.png" % (step))
            cubemap_feature_plot.save(save_name)
        
        # depth cubemap plot (if depth cubemap is used)
        if model.cubemap_depth_textures is not None:
            os.makedirs(os.path.join(log_dir, "train_cubemap_depth_plots"), exist_ok=True)
            # For depth cubemap, visualize as a heatmap (1 channel, no activation)
            cubemap_depth_plot = get_cubemap_plot(args.index, step, model.cubemap_depth_textures, activation=None, is_depth_map=True)
            save_name = os.path.join(log_dir, "train_cubemap_depth_plots", "iter_%d.png" % (step))
            cubemap_depth_plot.save(save_name)
            
            # Visualize cubemap depth as 3D point cloud
            # Generate uniform sample directions on a unit sphere using Fibonacci lattice
            num_samples = 10000
            indices = np.arange(0, num_samples, dtype=float) + 0.5
            phi = np.arccos(1 - 2 * indices / num_samples)  # polar angle
            theta = np.pi * (1 + 5**0.5) * indices  # azimuthal angle (golden angle)
            
            # Convert to Cartesian coordinates (unit sphere directions)
            directions_np = np.stack([
                np.sin(phi) * np.cos(theta),
                np.cos(phi),  # y is up
                np.sin(phi) * np.sin(theta)
            ], axis=-1).astype(np.float32)
            
            # Convert to tensor and move to device
            directions = torch.from_numpy(directions_np).to(device)  # (num_samples, 3)
            
            # Compute base sphere intersection points
            sphere_radius = model.bkg_sphere_radius * model.coord_scale
            sphere_center = torch.tensor([c * model.coord_scale for c in model.bkg_sphere_center], device=device, dtype=torch.float32)
            base_points = sphere_center + directions * sphere_radius  # (num_samples, 3)
            
            # Sample depth offset from cubemap
            with torch.no_grad():
                depth_offset = model.sample_cubemap(directions, 
                                                    textures=model.cubemap_depth_textures, 
                                                    activation=None)  # (num_samples, 1)
                depth_offset = depth_offset * model.cubemap_depth_scale * model.coord_scale
            
            # Apply depth offset along each direction to get final points
            final_points = base_points + directions * depth_offset  # (num_samples, 3)
            
            # Save as PLY file (normalized by coord_scale)
            os.makedirs(os.path.join(log_dir, "train_cubemap_depth_pcds"), exist_ok=True)
            save_points(os.path.join(log_dir, "train_cubemap_depth_pcds", "depth_surface_iter_%d.ply" % step), 
                       final_points.detach().cpu().numpy())
        
        # Save foreground/background RGB as separate images (when cubemap is used)
        if test_foreground_rgb is not None and test_background_rgb is not None:
            fg_bg_dir = os.path.join(log_dir, "train_fg_bg_plots")
            os.makedirs(fg_bg_dir, exist_ok=True)
            
            # Concatenate foreground and background side by side
            fg_img = (test_foreground_rgb * 255).astype(np.uint8)
            bg_img = (test_background_rgb * 255).astype(np.uint8)
            combined_img = np.concatenate([fg_img, bg_img], axis=1)  # Side by side horizontally
            combined_pil = Image.fromarray(combined_img)
            combined_pil.save(os.path.join(fg_bg_dir, "fg_bg_iter_%d.png" % step))

    model.save(step, log_dir)
    save_points(os.path.join(log_dir, "points_normed.ply"), model.points.detach().cpu().numpy() / model.coord_scale)

    if step % 50000 == 0 or (step % 1000 == 0 and step < 20000):
        torch.save(model.state_dict(), os.path.join(log_dir, "model_%d.pth" % step))
        save_points(os.path.join(log_dir, "points_%d_normed.ply" % step), model.points.detach().cpu().numpy() / model.coord_scale)

    # torch.save(torch.tensor(train_losses), os.path.join(log_dir, "train_losses.pth"))
    # torch.save(torch.tensor(eval_losses), os.path.join(log_dir, "eval_losses.pth"))
    # torch.save(torch.tensor(eval_psnrs), os.path.join(log_dir, "eval_psnrs.pth"))
    # torch.save(torch.tensor(steps), os.path.join(log_dir, "steps.pth"))
    # torch.save(torch.tensor(pt_lrs), os.path.join(log_dir, "pt_lrs.pth"))
    # torch.save(torch.tensor(attn_lrs), os.path.join(log_dir, "attn_lrs.pth"))

    ckpt_dict = {
        "steps": torch.tensor(steps),
        "train_losses": torch.tensor(train_losses),
        "eval_losses": torch.tensor(eval_losses),
        "eval_psnrs": torch.tensor(eval_psnrs),
        "pt_lrs": torch.tensor(pt_lrs),
        "attn_lrs": torch.tensor(attn_lrs)
    }
    torch.save(ckpt_dict, os.path.join(log_dir, "loss_and_lr.pth"))

    return nearest_sp


def train_step(step, model, device, dataset, batch, loss_fn, args):
    normals = None
    surface_points = None
    img_idx = batch['idx']
    tgt = batch['image']
    mask = batch['mask']
    rayd = batch['rayd']
    rays_d_no_norm = batch['rays_d_no_norm']
    rayo = batch['rayo']
    pix_coords = batch['pix_coords']
    pixels = batch['pixels']
    if 'surface_points' in batch:
        surface_points = batch['surface_points']
    if 'normals' in batch:
        normals = batch['normals']
    c2w = dataset.get_c2w(img_idx)

    rayo = rayo.to(device)
    rayd = rayd.to(device)
    rays_d_no_norm = rays_d_no_norm.to(device)
    tgt = tgt.to(device)
    c2w = c2w.to(device)
    mask = mask.to(device)
    pix_coords = pix_coords.to(device)
    pixels = pixels.to(device)
    if surface_points is not None:
        surface_points = surface_points.to(device)

    bkg_color = model.bkg_feats
    if args.rnd_background:
        # bkg_mask = (tgt.sum(-1, keepdim=True) >= (3 - 1e-8)).expand_as(tgt)
        shift_scheduler = lambda x:(x-args.rnd_background_start)/(args.rnd_background_stop-args.rnd_background_start) if args.rnd_background_stop-args.rnd_background_start > 0 and x >= args.rnd_background_start else 0.0
        shift = shift_scheduler(step) + args.rnd_background_shift
        if step % 200 == 0:
            print("**** rnd background shift:", shift, "start:", args.rnd_background_start, "stop:", args.rnd_background_stop)
        rnd_bkg_color = torch.rand(3).to(device) + shift
        # rnd_bkg_color = rnd_bkg_color.clamp(0, 1)
        rnd_bkg_color = rnd_bkg_color * model.bkg_scaler.squeeze(0)
        # tgt = torch.where(tgt.sum(-1, keepdim=True) < 3.0, tgt, rnd_bkg_color)
        tgt[mask.squeeze(-1) < 0.5] = rnd_bkg_color
        bkg_color = rnd_bkg_color[None, :]
    else:
        bkg_color = bkg_color * model.bkg_scaler.squeeze(0)

    model.clear_grad()
    
    with autocast(device_type='cuda', dtype=model.amp_dtype, enabled=model.use_amp):
        out, int_out, grid, regularizers, hit_pred = model(rayo, rayd, c2w, pix_coords, pixels, bkg_color, mask, step, surface_points=surface_points, normals=normals, rays_d_no_norm=rays_d_no_norm)
        out = model.last_act(out)

        if grid is not None:
            tgt = torch.nn.functional.grid_sample(tgt.permute(0, 3, 1, 2), grid, align_corners=True).permute(0, 2, 3, 1)
            
        rgb_loss = loss_fn(out, tgt)
        int_loss = torch.zeros(0, device=device).sum()
        hit_loss = torch.zeros(0, device=device).sum()
        
        if hit_pred is not None:
            hit_loss = -((mask.squeeze(-1) * torch.log(hit_pred + 1e-8)) + ((1 - mask.squeeze(-1)) * torch.log(1 - hit_pred + 1e-8))).mean()
            
        loss = rgb_loss + hit_loss * args.training.hit_loss_weight
    
        for loss_name, (loss_val, loss_weight) in regularizers.items():
            # Regularizers are (loss_tensor, loss_weight) tuples from compute_scheduled_loss
            if loss_weight > 0:
                if args.training.get(loss_name + "_scale_by_rgb_loss", False) and loss_val.item() > rgb_loss.item():
                    # scale the loss_val to the same range as the rgb_loss
                    loss_val = loss_val * (rgb_loss.item() / loss_val.item())
                loss = loss + loss_val * loss_weight
                if step % 200 == 0:
                    print(f" --{loss_name}:", loss_val.item(), "weight:", loss_weight, "weighted loss:", loss_val.item() * loss_weight)

    model.scaler.scale(loss).backward()

    if step % 200 == 0:
        print(" --rgb loss:", rgb_loss.item(), "--hit loss:", hit_loss.item(), "--total loss:", loss.item(), '--bkg scaler:', [f"{x:.4f}" for x in model.bkg_scaler.squeeze(0).tolist()])
    
    with torch.no_grad():
        prune_thresh = args.training.prune_thresh
        skip_prune = False
        if (args.training.prune_steps > 0) and (step < args.training.prune_stop) and (step >= args.training.prune_start):
            if len(args.training.prune_steps_list) > 0:
                prune_thresh = args.training.prune_thresh_list[bisect.bisect_left(args.training.prune_steps_list, step)]
            elif args.training.prune_thresh_quantile is not None or (args.training.prune_thresh_quantile_start is not None and args.training.prune_thresh_quantile_end is not None):
                # Quantile-based pruning: only prune if number of points >= max_num_pts
                if 'max_num_pts' in args and args.max_num_pts > 0 and model.points.shape[0] < args.max_num_pts * 0.95:
                    skip_prune = True
                else:
                    # Compute prune threshold as a quantile of influence scores
                    if args.training.prune_thresh_quantile_start is not None and args.training.prune_thresh_quantile_end is not None:
                        # Linear interpolation of quantile between start and end
                        prune_range = args.training.prune_stop - args.training.prune_start
                        if prune_range > 0:
                            progress = (step - args.training.prune_start) / prune_range
                            progress = max(0.0, min(1.0, progress))  # Clamp to [0, 1]
                            quantile = args.training.prune_thresh_quantile_start + progress * (args.training.prune_thresh_quantile_end - args.training.prune_thresh_quantile_start)
                        else:
                            quantile = args.training.prune_thresh_quantile_start
                    else:
                        quantile = args.training.prune_thresh_quantile
                    influ_scores = model.get_influ_scores().squeeze()
                    prune_thresh = torch.quantile(influ_scores, quantile).item()
            elif args.training.prune_thresh_start is not None and args.training.prune_thresh_end is not None:
                # Linear interpolation of prune threshold between start and end
                prune_range = args.training.prune_stop - args.training.prune_start
                if prune_range > 0:
                    progress = (step - args.training.prune_start) / prune_range
                    progress = max(0.0, min(1.0, progress))  # Clamp to [0, 1]
                    prune_thresh = args.training.prune_thresh_start + progress * (args.training.prune_thresh_end - args.training.prune_thresh_start)
                else:
                    prune_thresh = args.training.prune_thresh_start

            if step % args.training.prune_steps == 0 and not skip_prune:
                num_pruned = model.prune_points(prune_thresh, step=step)
                model.pruned_points = True
                print("Step %d: Pruned %d points, prune threshold %f" % (step, num_pruned, prune_thresh))

        if model.pruned_points and len(args.training.add_steps_list) > 0:
            if step in args.training.add_steps_list:
                cur_add_num = args.training.add_num_list[args.training.add_steps_list.index(step)]
                if 'max_num_pts' in args and args.max_num_pts > 0:
                    cur_add_num = min(cur_add_num, args.max_num_pts - model.points.shape[0])
                
                if cur_add_num > 0:
                    num_added = model.add_points(cur_add_num, step=step, prune_thresh=prune_thresh)
                    model.added_points = True
                    print("Step %d: Added %d points" % (step, num_added))

        elif model.pruned_points and (args.training.add_steps > 0) and (step % args.training.add_steps == 0) and (step < args.training.add_stop) and (step >= args.training.add_start):
            cur_add_num = args.training.add_num
            if 'max_num_pts' in args and args.max_num_pts > 0:
                cur_add_num = min(cur_add_num, args.max_num_pts - model.points.shape[0])
            
            if cur_add_num > 0:
                num_added = model.add_points(args.training.add_num, step=step, prune_thresh=prune_thresh)
                model.added_points = True
                print("Step %d: Added %d points" % (step, num_added))
                
        if args.geoms.alpha.reset and (args.geoms.alpha.reset_steps > 0) and (step < args.geoms.alpha.reset_stop) and (step >= args.geoms.alpha.reset_start) and step % args.geoms.alpha.reset_steps == 0:
            model.reset_alphas()

        # Get gradient clipping parameters from config
        grad_clip_norm = getattr(args.training, 'grad_clip_norm', None)
        grad_clip_value = getattr(args.training, 'grad_clip_value', None)
        model.step(step, grad_clip_norm=grad_clip_norm, grad_clip_value=grad_clip_value)

        if step % 200 == 0:
            print("max points grad:", model.points.grad.max().item())

        if args.scaler_min_scale > 0 and model.scaler.get_scale() < args.scaler_min_scale:
            model.scaler.update(args.scaler_min_scale)
        else:
            model.scaler.update()

    # Prepare logging dictionary with grouped metrics
    log_dict = {
        'losses/total': loss.item(),
        'losses/rgb': rgb_loss.item(),
        'losses/hit': hit_loss.item(),
    }
    
    # Add regularizer losses to log_dict
    for loss_name, (loss_val, loss_weight) in regularizers.items():
        # Regularizers are (loss_tensor, loss_weight) tuples from compute_scheduled_loss
        if loss_weight > 0:
            log_dict[f'losses/regularizer_{loss_name}'] = loss_val.item()
            log_dict[f'losses/regularizer_{loss_name}_weight'] = loss_weight
    
    return loss.item(), out.detach().cpu().numpy(), tgt, log_dict


def compile_model_if_enabled(model, args, device):
    """
    Compile the model with torch.compile if enabled and PyTorch version supports it.
    
    Args:
        model: The model to compile
        args: Arguments containing compile configuration
        device: Device to check compatibility
        
    Returns:
        Compiled model or original model if compilation is disabled/unsupported
    """
    # Check if compilation is enabled
    use_compile = getattr(args.training, 'use_compile', True)  # Default to True for acceleration
    
    if not use_compile:
        print("torch.compile disabled in config")
        return model
    
    # Check PyTorch version
    if not hasattr(torch, 'compile'):
        print("Warning: torch.compile not available (requires PyTorch >= 2.0). Skipping compilation.")
        return model
    
    # Check if CUDA is available (torch.compile works best with CUDA)
    if device.type != 'cuda':
        print("Warning: torch.compile is most effective with CUDA. CPU compilation may not provide speedup.")
    
    try:
        compile_mode = getattr(args.training, 'compile_mode', 'default')  # Options: 'default', 'reduce-overhead', 'max-autotune'
        compile_backend = getattr(args.training, 'compile_backend', None)  # None uses default backend
        
        print(f"Compiling model with mode='{compile_mode}'" + (f", backend='{compile_backend}'" if compile_backend else ""))
        
        if compile_backend:
            compiled_model = torch.compile(model, mode=compile_mode, backend=compile_backend)
        else:
            compiled_model = torch.compile(model, mode=compile_mode)
        
        print("Model compilation successful!")
        return compiled_model
    except Exception as e:
        print(f"Warning: Model compilation failed: {e}. Using uncompiled model.")
        return model


def compile_loss_fn_if_enabled(loss_fn, args, device):
    """
    Compile the loss function with torch.compile if enabled.
    
    Args:
        loss_fn: The loss function to compile
        args: Arguments containing compile configuration
        device: Device to check compatibility
        
    Returns:
        Compiled loss function or original loss function if compilation is disabled/unsupported
    """
    # Check if compilation is enabled
    use_compile_loss = getattr(args.training, 'use_compile_loss', False)  # Default to False as loss is usually simpler
    
    if not use_compile_loss:
        return loss_fn
    
    # Check PyTorch version
    if not hasattr(torch, 'compile'):
        print("Warning: torch.compile not available (requires PyTorch >= 2.0). Skipping loss compilation.")
        return loss_fn
    
    try:
        compile_mode = getattr(args.training, 'compile_mode', 'default')
        print(f"Compiling loss function with mode='{compile_mode}'")
        compiled_loss_fn = torch.compile(loss_fn, mode=compile_mode)
        print("Loss function compilation successful!")
        return compiled_loss_fn
    except Exception as e:
        print(f"Warning: Loss function compilation failed: {e}. Using uncompiled loss function.")
        return loss_fn


def train_and_eval(steps, start_step, model, device, dataset, eval_dataset, losses, lrs, args):
    trainloader = get_loader(dataset, args.dataset, mode="train")
    model.fx = dataset.focal_x
    model.fy = dataset.focal_y
    model.cx = dataset.cx
    model.cy = dataset.cy
    model.H = dataset.H
    model.W = dataset.W

    loss_fn = get_loss(args.training.losses)
    loss_fn = loss_fn.to(device)
    
    # Compile loss function if enabled
    loss_fn = compile_loss_fn_if_enabled(loss_fn, args, device)

    log_dir = os.path.join(args.save_dir, args.index)
        
    # Save the initial points to a ply file using save_points function
    save_points(os.path.join(log_dir, "points_init.ply"), model.points.detach().cpu().numpy())
    
    os.makedirs(os.path.join(log_dir, "test"), exist_ok=True)
    log_dir = os.path.join(log_dir, "test")

    train_losses, eval_losses, eval_psnrs = losses
    pt_lrs, attn_lrs = lrs

    avg_train_loss = 0.
    step = start_step
    eval_step_cnt = 0
    pruned = False
    pc_frames = []
    
    nearest_sp = torch.zeros(model.H, model.W, device=device)

    print("Start step:", start_step, "Total steps:", args.training.steps)
    start_time = time.time()
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    while step < args.training.steps:
        for _, batch in enumerate(trainloader):

            loss, out, tgt, train_log_dict = train_step(step, model, device, dataset, batch, loss_fn, args)
            if step % 200 == 0:
                time_used = time.time() - start_time
                print("Train step:", step, "loss:", loss, "attn_lr:", model.attn_lr, "pts_lr:", model.pts_lr, "scale:", model.scaler.get_scale(), f"time: {time_used:.2f}s")
                
                # Log training metrics to wandb
                if WANDB_AVAILABLE and wandb.run is not None:
                    # Add additional training metrics to the log_dict with grouping
                    train_log_dict.update({
                        "training/loss": train_log_dict['losses/total'],
                        "training/rgb_loss": train_log_dict['losses/rgb'],
                        # "training/intrinsic_loss": train_log_dict['losses/intrinsic'],
                        "optimization/attention_lr": model.attn_lr,
                        "optimization/points_lr": model.pts_lr,
                        "optimization/scaler_scale": model.scaler.get_scale(),
                        "performance/time_per_200_steps": time_used,
                        "step": step,
                    })
                    
                    wandb.log(train_log_dict)
                start_time = time.time()

            # model.points_influ_scores.data = torch.nan_to_num(model.points_influ_scores.data, nan=prune_thresh + 1e-6, posinf=args.geoms.points.influ_max, neginf=args.geoms.points.influ_min)
                
            avg_train_loss += loss
            step += 1
            eval_step_cnt += 1

            if (step % args.eval.step == 0) or (step % 500 == 0 and step < 20000) or step == 1:
                train_losses.append(avg_train_loss / eval_step_cnt)
                pt_lrs.append(model.pts_lr)
                attn_lrs.append(model.attn_lr)
                steps.append(step)
                nearest_sp = eval_step(steps, model, device, dataset, eval_dataset, batch, loss_fn, out, args, train_losses, eval_losses, eval_psnrs, pt_lrs, attn_lrs, tgt, nearest_sp)
                avg_train_loss = 0.
                eval_step_cnt = 0

            if ((step - 1) % 200 == 0) and args.eval.save_fig:
                coord_scale = args.dataset.coord_scale
                pt_plot_scale = 0.8 * coord_scale
                if "Barn" in args.dataset.path:
                    pt_plot_scale *= 1.5
                if "Family" in args.dataset.path:
                    pt_plot_scale *= 0.5    

                pc_dir = os.path.join(log_dir, "point_clouds")
                os.makedirs(pc_dir, exist_ok=True)

                points_np = model.points.detach().cpu().numpy()
                points_influ_scores_np = None
                if model.points_influ_scores is not None:
                    points_influ_scores_np = model.points_influ_scores.squeeze().detach().cpu().numpy()
                pcd_plot = get_training_pcd_single_plot(step, points_np, pt_plot_scale, points_influ_scores_np)
                pc_frames.append(pcd_plot)
                
                if step == 1:
                    pcd_plot.save(os.path.join(pc_dir, "init_pcd.png"))

            if step >= args.training.steps:
                break

    # if args.eval.save_fig and pc_frames != []:
    #     f = os.path.join(log_dir, f"{args.index}-pc.mp4")
    #     imageio.mimwrite(f, pc_frames, fps=30, quality=10)

    print("Training finished!")
    
    # Save nearest_sp in pth
    torch.save(nearest_sp, os.path.join(log_dir, "nearest_sp.pth"))
    print("Saved nearest_sp to %s" % os.path.join(log_dir, "nearest_sp.pth"))

            
def setup_torch_optimizations(args, device):
    """
    Set up PyTorch optimizations that should be configured before model creation/compilation.
    """
    # Enable TensorFloat32 for better performance on Ampere+ GPUs (RTX 30xx, A100, etc.)
    if device.type == 'cuda':
        try:
            torch.set_float32_matmul_precision('high')
            print("Enabled TensorFloat32 (TF32) for better performance")
        except Exception as e:
            print(f"Warning: Could not enable TF32: {e}")
    
    # Configure dynamo to capture scalar outputs (reduces graph breaks from .item() calls)
    # This must be set BEFORE any compilation happens
    try:
        if hasattr(torch, '_dynamo') and hasattr(torch._dynamo.config, 'capture_scalar_outputs'):
            torch._dynamo.config.capture_scalar_outputs = True
            print("Enabled capture_scalar_outputs to reduce graph breaks from .item() calls")
    except Exception as e:
        print(f"Warning: Could not configure capture_scalar_outputs: {e}")
    
    # Increase cache size limits to reduce recompilations
    try:
        if hasattr(torch, '_dynamo') and hasattr(torch._dynamo.config, 'accumulated_cache_size_limit'):
            cache_limit = getattr(args.training, 'compile_cache_size_limit', 512)  # Default to 512
            torch._dynamo.config.accumulated_cache_size_limit = cache_limit
            print(f"Set dynamo accumulated cache size limit to {cache_limit} (default: 256)")
        
        # Increase per-function cache size limit (default is 8)
        if hasattr(torch, '_dynamo') and hasattr(torch._dynamo.config, 'cache_size_limit'):
            per_function_cache_limit = getattr(args.training, 'compile_per_function_cache_limit', 64)  # Default to 64
            torch._dynamo.config.cache_size_limit = per_function_cache_limit
            print(f"Set dynamo per-function cache size limit to {per_function_cache_limit} (default: 8)")
    except Exception as e:
        print(f"Warning: Could not configure cache size limit: {e}")
    
    # Allow tinycudann operations in graph (reduces graph breaks)
    try:
        import torch.compiler
        import tinycudann as tcnn
        # Allow tinycudann modules and operations in the graph
        if hasattr(torch.compiler, 'allow_in_graph'):
            # Allow tinycudann encoding and network operations
            torch.compiler.allow_in_graph(tcnn.Encoding)
            torch.compiler.allow_in_graph(tcnn.Network)
            # Also try to allow the forward methods
            if hasattr(tcnn.Encoding, 'forward'):
                torch.compiler.allow_in_graph(tcnn.Encoding.forward)
            if hasattr(tcnn.Network, 'forward'):
                torch.compiler.allow_in_graph(tcnn.Network.forward)
            print("Allowed tinycudann operations in compiled graph")
    except Exception as e:
        print(f"Warning: Could not configure tinycudann graph integration: {e}")


def main(args, eval_args, resume):
    log_dir = os.path.join(args.save_dir, args.index)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up optimizations BEFORE model creation/compilation
    setup_torch_optimizations(args, device)

    model = get_model(args, device)
    dataset = get_dataset(args.dataset, mode="train")
    eval_dataset = get_dataset(eval_args.dataset, mode=eval_args.dataset.mode)
    model = model.to(device)

    # Initialize wandb if enabled
    if WANDB_AVAILABLE and hasattr(args, 'wandb') and args.wandb.enabled:
        wandb_config = {
            'project': args.wandb.project,
            'entity': args.wandb.entity if args.wandb.entity else None,
            'name': args.index,
            'tags': args.wandb.tags if args.wandb.tags else [],
            'notes': args.wandb.notes if args.wandb.notes else "",
        }
        
        # Add experiment config to wandb
        wandb_config['config'] = {
            'index': args.index,
            'save_dir': args.save_dir,
            'seed': args.seed,
            'use_amp': args.use_amp,
            'dataset': dict(args.dataset),
            'models': dict(args.models),
            'training': dict(args.training),
            'eval': dict(args.eval),
            **args.wandb.config
        }
        
        wandb.init(**wandb_config)
        print(f"Initialized wandb logging for experiment: {args.index}")
    else:
        print("Wandb logging disabled or not available")

    start_step = 0
    steps = []
    losses = [[], [], []]
    lrs = [[], []]
    # exists = os.path.exists(os.path.join(log_dir, "model.pth"))
    if resume > 0:
        start_step = model.load(log_dir)

        # train_losses = torch.load(os.path.join(log_dir, "train_losses.pth")).tolist()
        # eval_losses = torch.load(os.path.join(log_dir, "eval_losses.pth")).tolist()
        # eval_psnrs = torch.load(os.path.join(log_dir, "eval_psnrs.pth")).tolist()
        # steps = torch.load(os.path.join(log_dir, "steps.pth")).tolist()
        # pt_lrs = torch.load(os.path.join(log_dir, "pt_lrs.pth")).tolist()
        # attn_lrs = torch.load(os.path.join(log_dir, "attn_lrs.pth")).tolist()
        # losses = [train_losses, eval_losses, eval_psnrs]
        # lrs = [pt_lrs, attn_lrs]

        ckpt_dict = torch.load(os.path.join(log_dir, "loss_and_lr.pth"), weights_only=True)
        steps = ckpt_dict["steps"].tolist()
        train_losses = ckpt_dict["train_losses"].tolist()
        eval_losses = ckpt_dict["eval_losses"].tolist()
        eval_psnrs = ckpt_dict["eval_psnrs"].tolist()
        pt_lrs = ckpt_dict["pt_lrs"].tolist()
        attn_lrs = ckpt_dict["attn_lrs"].tolist()
        losses = [train_losses, eval_losses, eval_psnrs]
        lrs = [pt_lrs, attn_lrs]

        print("!!!!! Resuming from step %s" % start_step)
        
        # Compile model after loading checkpoint to ensure compiled model has loaded weights
        model = compile_model_if_enabled(model, args, device)
    elif args.load_path:
        if ".pth" in args.load_path:
            load_path = args.load_path
        else:
            load_path = os.path.join(load_path, "model.pth")
        model_state_dict = torch.load(load_path, weights_only=True)
        if len(list(model_state_dict.keys())) == 1:
            resume_step = list(model_state_dict.keys())[0]
            model.load_my_state_dict(model_state_dict[resume_step])
            resume_step = int(resume_step)
        else:
            resume_step = int(os.path.basename(args.load_path).split("_")[-1].split(".")[0])
            model.load_my_state_dict(model_state_dict)
        print("!!!!! Loaded model from %s at step %s" % (args.load_path, resume_step))
        
        # Re-initialize optimizers after loading model state to ensure they reference the new parameters
        # This is critical when the loaded checkpoint has a different number of points than the initial model
        model.init_optimizers(resume_step)
        
        # Compile model after loading from path to ensure compiled model has loaded weights
        model = compile_model_if_enabled(model, args, device)
    
    # Compile model if not already compiled (for fresh training without resume/load_path)
    if resume == 0 and not args.load_path:
        model = compile_model_if_enabled(model, args, device)

    train_and_eval(steps, start_step, model, device, dataset, eval_dataset, losses, lrs, args)
    print(torch.cuda.memory_summary())
    
    # Finish wandb logging if enabled
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()
        print("Wandb logging finished")


if __name__ == '__main__':

    with open("configs/default_fast.yml", 'r') as f:
        default_config = yaml.safe_load(f)

    args = parse_args()
    with open(args.opt, 'r') as f:
        config = yaml.safe_load(f)

    train_config = copy.deepcopy(default_config)
    update_dict(train_config, config)
    
    # Use unified bkg_sphere_radius config (fallback to bkg_points_init_scale[0] for backward compatibility)
    if 'bkg_sphere_radius' not in train_config['geoms']['points']:
        train_config['geoms']['points']['bkg_sphere_radius'] = train_config['geoms']['points']['bkg_points_init_scale'][0]
    train_config['dataset']['bkg_sphere_radius'] = train_config['geoms']['points']['bkg_sphere_radius']
    
    # Use unified bkg_sphere_center config (fallback to [0.0, 0.0, 0.0] for backward compatibility)
    if 'bkg_sphere_center' not in train_config['geoms']['points']:
        train_config['geoms']['points']['bkg_sphere_center'] = [0.0, 0.0, 0.0]
    train_config['dataset']['bkg_sphere_center'] = train_config['geoms']['points']['bkg_sphere_center']

    eval_config = copy.deepcopy(train_config)
    eval_config['dataset'].update(eval_config['eval']['dataset'])
    eval_config = DictAsMember(eval_config)
    train_config = DictAsMember(train_config)

    log_dir = os.path.join(train_config.save_dir, train_config.index)
    os.makedirs(log_dir, exist_ok=True)

    sys.stdout = Logger(os.path.join(log_dir, 'train.log'), sys.stdout)
    sys.stderr = Logger(os.path.join(log_dir, 'train_error.log'), sys.stderr)

    shutil.copyfile(__file__, os.path.join(log_dir, os.path.basename(__file__)))
    shutil.copyfile(args.opt, os.path.join(log_dir, os.path.basename(args.opt)))

    find_all_python_files_and_zip(".", os.path.join(log_dir, "code.zip"))

    setup_seed(train_config.seed)

    main(train_config, eval_config, args.resume)