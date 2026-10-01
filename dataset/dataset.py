import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from copy import deepcopy
import os
import glob
import open3d as o3d
import imageio
import re
from .utils import load_meta_data, get_rays, extract_patches
from utils import get_ray_sphere_intersection


class RINDataset(Dataset):
    """ Ray Image Normal Dataset """

    def __init__(self, args, mode='train', device='cuda'):
        self.args = args
        images, c2w, masks, H, W, focal_x, focal_y, image_paths = load_meta_data(args, mode=mode)
        num_imgs = len(image_paths)

        self.num_imgs = num_imgs
        coord_scale = args.coord_scale
        if coord_scale != 1:
            scaling_matrix = torch.tensor([[coord_scale, 0, 0, 0],
                                           [0, coord_scale, 0, 0],
                                           [0, 0, coord_scale, 0],
                                           [0, 0, 0, 1]], dtype=torch.float32)
            c2w = torch.matmul(scaling_matrix, c2w)
        print("c2w: ", c2w.shape)
        
        self.device = device

        self.num_rays = H * W * num_imgs

        self.H = H
        self.W = W
        self.focal_x = focal_x
        self.focal_y = focal_y
        self.cx = W / 2
        self.cy = H / 2
        self.c2w = c2w.to(device)      # (N, 4, 4)
        self.image_paths = image_paths
        self.images = images.to(device)    # (N, H, W, C) or None
        self.masks = masks.to(device)
        # self.pix_coords = torch.stack(torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy'), dim=-1).float().to(device)
        i, j = torch.meshgrid(
            torch.arange(W, dtype=torch.float32) + 0.5,  # Pixel centers
            torch.arange(H, dtype=torch.float32) + 0.5,  # Pixel centers
            indexing='xy'
        )
        self.pix_coords = torch.stack([i, j], dim=-1).float().to(device)
        # self.pix_coords = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij'), dim=-1).float()
        print("pix_coords: ", self.pix_coords.shape, "H: ", H, "W: ", W, "image shape: ", images.shape)
        
        masks = masks.squeeze(-1) > 0.5
        self.foreground_coords = torch.nonzero(masks)

        rays_o, rays_d, pixels, rays_d_no_norm = get_rays(H, W, focal_x, focal_y, c2w)
        self.rayd = rays_d.to(device)      # (N, H, W, 3)
        self.rayo = rays_o.to(device)      # (N, 3)
        self.rays_d_no_norm = rays_d_no_norm.to(device)
        self.pixels = pixels.to(device)    # (N, H, W, 3)

        if args.load_gt_depth:
            gt_depth_dir = args.gt_depth_dir
            self.gt_depths = []
            self.gt_surface_points = []
            possible_exts = ['.npy', '.tiff', '.png']
            print(f"Loading ground truth depth from {gt_depth_dir}")
            for i in range(self.num_imgs):
                found = False
                for ext in possible_exts:
                    pattern = os.path.join(gt_depth_dir, f"*{i}*{ext}")
                    files = sorted(glob.glob(pattern))
                    selected_file = None
                    for f in files:
                        fname = os.path.basename(f)
                        nums = [int(n) for n in re.findall(r'\d+', fname)]
                        if i in nums:
                            selected_file = f
                            break
                    
                    if selected_file:
                        if ext == '.npy':
                            depth = np.load(selected_file)
                        else:
                            depth = imageio.imread(selected_file)
                        
                        if len(depth.shape) == 3:
                            depth = depth[..., 0]
                        
                        depth = torch.from_numpy(depth).float()
                        
                        if depth.shape[0] != self.H or depth.shape[1] != self.W:
                             depth = depth.unsqueeze(0).unsqueeze(0)
                             depth = torch.nn.functional.interpolate(depth, size=(self.H, self.W), mode='bilinear', align_corners=False)
                             depth = depth.squeeze()
                        
                        if args.coord_scale != 1:
                            depth = depth * args.coord_scale
                        
                        # Compute 3D points
                        ray_o = self.rayo[i].cpu() # (3)
                        ray_d = rays_d_no_norm[i].cpu() # (H, W, 3)
                        depth_cpu = depth.cpu() # (H, W)
                        
                        # points = ray_o + ray_d * depth
                        points = ray_o + ray_d * depth_cpu.unsqueeze(-1) / args.coord_scale # (H, W, 3)
                        fg_mask = self.masks[i].cpu().squeeze() > 0.5
                        
                        # Compute intersection with sphere for background rays using unified function
                        bkg_mask = ~fg_mask
                        if bkg_mask.any():
                            # Get sphere radius from unified config (bkg_sphere_radius is already a scale factor)
                            sphere_radius = args.bkg_sphere_radius * args.coord_scale
                            sphere_center = getattr(args, 'bkg_sphere_center', [0.0, 0.0, 0.0])
                            sphere_center = [c * args.coord_scale for c in sphere_center]
                            
                            # Use unified ray-sphere intersection function
                            bkg_points = get_ray_sphere_intersection(ray_o, ray_d, sphere_center, sphere_radius)
                            
                            # Update points
                            points[bkg_mask] = bkg_points[bkg_mask]
                            
                            # Update depth in the depth map
                            # Compute t from the intersection points
                            ray_d_norm = ray_d / torch.norm(ray_d, dim=-1, keepdim=True)
                            ray_d_len = torch.norm(ray_d, dim=-1)
                            t_sphere = torch.norm(bkg_points - ray_o, dim=-1)
                            bkg_depth = (t_sphere / ray_d_len) * args.coord_scale
                            
                            depth[~fg_mask] = bkg_depth[~fg_mask]

                        if args.save_gt_sp:
                            # Save to ply
                            pcd = o3d.geometry.PointCloud()
                            pcd.points = o3d.utility.Vector3dVector(points.reshape(-1, 3).numpy())
                            save_path = os.path.join(gt_depth_dir, f"frame_{i:04d}.ply")
                            o3d.io.write_point_cloud(save_path, pcd)
                            print(f"Saved point cloud to {save_path}")

                        found = True
                        self.gt_depths.append(depth)
                        self.gt_surface_points.append(torch.tensor(points, dtype=torch.float).to(device))
                        break
                if not found:
                    print(f"Warning: No depth file found for frame {i} in {gt_depth_dir}")
                    self.gt_depths.append(torch.zeros((self.H, self.W)))
                    self.gt_surface_points.append(torch.zeros((self.H, self.W, 3)).to(device))
            
            print("self.gt_surface_points", len(self.gt_surface_points))
            self.gt_depths = torch.stack(self.gt_depths).to(device)
            self.gt_surface_points = torch.stack(self.gt_surface_points)
            print(f"Loaded ground truth depths: {self.gt_depths.shape}")
            
        # self.rayd = self.rayd / torch.norm(self.rayd, dim=-1, keepdim=True)
            

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        if self.args.extract_patch == True:
            start_height = np.random.randint(0, self.H - self.args.patches.height)
            start_width = np.random.randint(0, self.W - self.args.patches.width)
            end_height = start_height + self.args.patches.height
            end_width = start_width + self.args.patches.width

            return_dict = {
                'idx':        idx,
                'patch_idx':  0,
                'image':      self.images[idx][start_height:end_height, start_width:end_width, :],
                'mask':       self.masks[idx][start_height:end_height, start_width:end_width, :],
                'rayd':       self.rayd[idx][start_height:end_height, start_width:end_width, :],
                'rays_d_no_norm': self.rays_d_no_norm[idx][start_height:end_height, start_width:end_width, :],
                'rayo':       self.rayo[idx],
                'pix_coords': self.pix_coords[start_height:end_height, start_width:end_width, :],
                'pixels':     self.pixels[idx][start_height:end_height, start_width:end_width, :]
            }
            if self.args.load_gt_depth:
                return_dict['depth'] = self.gt_depths[idx][start_height:end_height, start_width:end_width]
                return_dict['surface_points'] = self.gt_surface_points[idx][start_height:end_height, start_width:end_width, :]
        else:
            return_dict = {
                'idx':        idx,
                'patch_idx':  0,
                'image':      self.images[idx],
                'mask':       self.masks[idx],
                'rayd':       self.rayd[idx],
                'rays_d_no_norm': self.rays_d_no_norm[idx],
                'rayo':       self.rayo[idx],
                'pix_coords': self.pix_coords,
                'pixels':     self.pixels[idx]
            }
            if self.args.load_gt_depth:
                return_dict['depth'] = self.gt_depths[idx]
                return_dict['surface_points'] = self.gt_surface_points[idx]
        return return_dict

    def get_full_img(self, img_idx):
        return_dict = {
            'idx':        img_idx,
            'patch_idx':  0,
            'image':      self.images[img_idx][None],
            'mask':       self.masks[img_idx][None],
            'rayd':       self.rayd[img_idx][None],
            'rays_d_no_norm': self.rays_d_no_norm[img_idx][None],
            'rayo':       self.rayo[img_idx][None],
            'pix_coords': self.pix_coords,
            'pixels':     self.pixels[img_idx][None],
        }
        if self.args.load_gt_depth:
            return_dict['depth'] = self.gt_depths[img_idx][None]
        return return_dict

    def get_c2w(self, img_idx):
        selected_c2w = self.c2w[img_idx]
        if selected_c2w.dim() == 2:
            selected_c2w = selected_c2w.unsqueeze(0)
        return selected_c2w

    def get_new_rays(self, c2w):
        return get_rays(self.H, self.W, self.focal_x, self.focal_y, c2w)[:3]
    