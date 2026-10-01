import sys
import traceback
import matplotlib.pyplot as plt
import io
from PIL import Image
import numpy as np
import os
import zipfile
import torch
import torch.nn.functional as F
import random
import copy


def get_ray_sphere_intersection(rays_o, rays_d, sphere_center=None, sphere_radius=1.0):
    """
    Compute the intersection point of rays with a sphere.
    
    This is the unified ray-sphere intersection function used throughout the codebase.
    
    Args:
        rays_o: Ray origins. Can be:
            - (N, 3) will be broadcast to match rays_d
            - (N, H, W, 3) full ray origins
            - (3,) single ray origin
            - (H, W, 3) single batch of ray origins
        rays_d: Ray directions (will be normalized internally). Can be:
            - (N, H, W, 3) full ray directions
            - (H, W, 3) single batch of ray directions
        sphere_center: Center of the sphere. Can be:
            - (3,) tensor or list/tuple
            - scalar 0 for origin
            - None for origin (default)
        sphere_radius: Scalar radius of the sphere (default: 1.0)
    
    Returns:
        intersection_points: Intersection points with same spatial shape as rays_d
            - (N, H, W, 3) if rays_d is (N, H, W, 3)
            - (H, W, 3) if rays_d is (H, W, 3)
    """
    device = rays_d.device
    dtype = rays_d.dtype
    
    # Handle different input shapes for rays_d
    original_ndim = rays_d.ndim
    if rays_d.ndim == 3:  # (H, W, 3)
        rays_d = rays_d.unsqueeze(0)  # (1, H, W, 3)
    
    N, H, W, _ = rays_d.shape
    
    # Handle different input shapes for rays_o
    if rays_o.ndim == 1:  # (3,)
        rays_o = rays_o.reshape(1, 1, 1, 3).expand(N, H, W, -1)
    elif rays_o.ndim == 2:  # (N, 3)
        rays_o = rays_o.reshape(N, 1, 1, 3).expand(-1, H, W, -1)
    elif rays_o.ndim == 3:  # (H, W, 3)
        rays_o = rays_o.unsqueeze(0).expand(N, -1, -1, -1)
    # else: (N, H, W, 3) - already correct
    
    # Normalize ray directions
    rays_d_normalized = F.normalize(rays_d, dim=-1, eps=1e-8)
    
    # Handle sphere_center
    if sphere_center is None or (isinstance(sphere_center, (int, float)) and sphere_center == 0):
        sphere_center = torch.zeros(1, 1, 1, 3, dtype=dtype, device=device)
    else:
        if not isinstance(sphere_center, torch.Tensor):
            sphere_center = torch.tensor(sphere_center, dtype=dtype, device=device)
        if sphere_center.ndim == 0:
            sphere_center = torch.zeros(1, 1, 1, 3, dtype=dtype, device=device)
        elif sphere_center.ndim == 1:
            sphere_center = sphere_center.reshape(1, 1, 1, 3)
    
    # Handle sphere_radius
    if not isinstance(sphere_radius, torch.Tensor):
        sphere_radius = torch.tensor(sphere_radius, dtype=dtype, device=device)
    
    # Vector from sphere center to ray origin
    oc = rays_o - sphere_center  # (N, H, W, 3)
    
    # Solve quadratic equation: t^2 + 2*t*(oc·d) + ||oc||^2 - r^2 = 0
    # where t is the parameter along the ray
    oc_dot_d = torch.sum(oc * rays_d_normalized, dim=-1)  # (N, H, W)
    oc_sq = torch.sum(oc * oc, dim=-1)  # (N, H, W)
    r_sq = sphere_radius * sphere_radius
    
    # Discriminant
    discriminant = oc_dot_d * oc_dot_d - oc_sq + r_sq
    
    # Take the positive t (forward direction from ray origin)
    # t = -b + sqrt(b^2 - 4ac) / 2a, where a=1, b=2*(oc·d), c=||oc||^2 - r^2
    # Simplified: t = -(oc·d) + sqrt((oc·d)^2 - ||oc||^2 + r^2)
    sqrt_discriminant = torch.sqrt(torch.clamp(discriminant, min=0.0))
    t = -oc_dot_d + sqrt_discriminant  # (N, H, W)
    
    # Ensure t is positive (forward direction)
    t = torch.clamp(t, min=0.0)
    
    # Compute intersection points
    intersection_points = rays_o + t.unsqueeze(-1) * rays_d_normalized  # (N, H, W, 3)
    
    # Return with original shape
    if original_ndim == 3:
        intersection_points = intersection_points.squeeze(0)  # (H, W, 3)
    
    return intersection_points


class DictAsMember(dict):
    def __getattr__(self, name):
        if name not in self:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        value = self[name]
        if isinstance(value, dict):
            value = DictAsMember(value)
        return value


def update_dict(original, param):
    for key in param.keys():
        if type(param[key]) == dict:
            update_dict(original[key], param[key])
        elif type(param[key]) == list and key == "datasets":
            for i in range(len(param[key])):
                name = param[key][i]['name']
                for j in range(len(original[key])):
                    if original[key][j]['name'] == name:
                        for k in param[key][i].keys():
                            original[key][j][k] = param[key][i][k]
                        break
                else:
                    new_param = copy.deepcopy(original[key][0])
                    update_dict(new_param, param[key][i])
                    original[key].append(new_param)
        else:
            original[key] = param[key]


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def find_all_python_files_and_zip(src_dir, dst_path):
    # find all python files in src_dir
    python_files = []
    for root, dirs, files in os.walk(src_dir):
        # if 'experiment' in root:
        #     continue
        # if root == 'data':
        #     continue
        dirs[:] = [d for d in dirs if d not in ['__pycache__', 'experiments', 'data']]
        for cur_file in files:
            if cur_file.endswith('.py'):
                python_files.append(os.path.join(root, cur_file))

    # zip all python files
    with zipfile.ZipFile(dst_path, 'w') as zip_file:
        for cur_file in python_files:
            zip_file.write(cur_file, os.path.relpath(cur_file, src_dir))


# class Logger(object):
#     def __init__(self, filename='default.log', stream=sys.stdout):
#         self.stream = stream
#         self.file = open(filename, 'a')
#         ct = datetime.now()
#         self.file.write('*'*50 + '\n' + str(ct) + '\n' + '*'*50 + '\n')

#     def write(self, message):
#         self.stream.write(message)
#         self.stream.flush()
#         self.file.write(message)

#     def flush(self):
#         pass


class Logger(object):
    def __init__(self, filename, stream=sys.stdout):
        self.file = open(filename, 'a')
        self.stream = stream

    def __enter__(self):
        sys.stream = self

    def __exit__(self, exc_type, exc_value, tb):
        sys.stream = self.stream
        if exc_type is not None:
            self.file.write(traceback.format_exc())
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stream.write(data)

    def flush(self):
        self.file.flush()
        self.stream.flush()
    
    def isatty(self):
        """Return True if the file is connected to a terminal device."""
        return self.stream.isatty()
    
    def fileno(self):
        """Return the file descriptor number of the stream."""
        return self.stream.fileno()
    
    def readable(self):
        """Return True if the stream can be read from."""
        return getattr(self.stream, 'readable', lambda: False)()
    
    def writable(self):
        """Return True if the stream can be written to."""
        return getattr(self.stream, 'writable', lambda: True)()
    
    def seekable(self):
        """Return True if the stream supports random access."""
        return getattr(self.stream, 'seekable', lambda: False)()
    
    def read(self, size=-1):
        """Read and return up to size bytes."""
        return self.stream.read(size)
    
    def readline(self, size=-1):
        """Read and return one line from the stream."""
        return self.stream.readline(size)
    
    def readlines(self, hint=-1):
        """Read and return a list of lines from the stream."""
        return self.stream.readlines(hint)
    
    def tell(self):
        """Return the current stream position."""
        return getattr(self.stream, 'tell', lambda: 0)()
    
    def seek(self, offset, whence=0):
        """Change the stream position to the given byte offset."""
        return getattr(self.stream, 'seek', lambda offset, whence: None)(offset, whence)
    
    def truncate(self, size=None):
        """Truncate the file to at most size bytes."""
        return getattr(self.stream, 'truncate', lambda size: None)(size)
    
    def close(self):
        """Flush and close the stream."""
        self.flush()
        self.file.close()
        if hasattr(self.stream, 'close'):
            self.stream.close()


def get_colors(weights):
    N = weights.shape[0]
    weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
    colors = np.full((N, 3), [1., 0., 0.])
    colors[:, 0] *= weights[:N]
    colors[:, 2] = (1 - weights[:N])
    return colors


def get_training_main_plot(index, steps, train_tgt_rgb, train_tgt_patch, train_pred_patch, test_tgt_tgb, test_pred_rgb, train_losses,
                           eval_losses, points_np, pt_plot_scale, depth_np, min_d2r, pt_lrs, attn_lrs, eval_psnrs, points_conf_scores_np=None, nearest_sp=None, test_pred_int_rgb=None):
    step = steps[-1]
    fig = plt.figure(figsize=(20, 10))

    ax = fig.add_subplot(2, 6, 1)
    cd = ax.imshow(min_d2r)
    fig.colorbar(cd, ax=ax)
    ax.set_title(f'min pd')

    ax = fig.add_subplot(2, 6, 2)
    ax.imshow(train_tgt_patch)
    ax.set_title(f'Iteration: {step} train norm patch')

    ax = fig.add_subplot(2, 6, 3)
    ax.imshow(train_pred_patch)
    ax.set_title(f'Iteration: {step} train output')

    ax = fig.add_subplot(2, 6, 4)
    ax.plot(steps, train_losses, label='train')
    ax.plot(steps, eval_losses, label='eval')
    ax.set_ylim(0, 0.01)
    ax.legend()
    ax.set_title('losses')

    ax = fig.add_subplot(2, 6, 5, projection='3d')
    ax.set_xlim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_ylim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_zlim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    cur_color = "grey"
    if points_conf_scores_np is not None:
        cur_color = get_colors(points_conf_scores_np)
    ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], c=cur_color)
    ax.set_title('Point Cloud')
    
    if nearest_sp is not None:
        ax = fig.add_subplot(2, 6, 6)
        cd = ax.imshow(nearest_sp)
        fig.colorbar(cd, ax=ax)
        ax.set_title(f'nearest sp to cam')
        
    ax = fig.add_subplot(2, 6, 7)
    cd = ax.imshow(depth_np)
    fig.colorbar(cd, ax=ax)
    ax.set_title(f'depth map')

    ax = fig.add_subplot(2, 6, 8)
    ax.imshow(test_tgt_tgb)
    ax.set_title(f'Iteration: {step} eval norm')

    ax = fig.add_subplot(2, 6, 9)
    ax.imshow(test_pred_rgb)
    ax.set_title(f'Iteration: {step} eval predict')

    ax = fig.add_subplot(2, 6, 10)
    ax.plot(steps, np.log10(pt_lrs), label="pt lr")
    ax.plot(steps, np.log10(attn_lrs), label="attn lr")
    ax.legend()
    ax.set_title('learning rates log10')

    ax = fig.add_subplot(2, 6, 11)
    ax.plot(steps, eval_psnrs)
    ax.set_title('eval psnr')
    
    if test_pred_int_rgb is not None:
        ax = fig.add_subplot(2, 6, 12)
        ax.imshow(test_pred_int_rgb)
        ax.set_title(f'Iteration: {step} eval predict intrinsic')

    fig.suptitle("Main Plot\n%s\niter %d\nnum pts: %d" % (index, step, points_np.shape[0]))

    canvas = fig.canvas
    buffer = io.BytesIO()
    canvas.print_png(buffer)
    data = buffer.getvalue()
    buffer.write(data)
    img = Image.open(buffer)
    plt.close()

    return img


def get_cubemap_plot(index, step, cubemap_textures, activation=None, is_feature_map=False, is_depth_map=False):
    """
    Generate a plot of the 6 cubemap face textures.
    
    Args:
        index: Experiment index/name
        step: Current training step
        cubemap_textures: List of 6 tensors of shape (C, H, W) representing the cubemap faces
                         in order: [+X, -X, +Y, -Y, +Z, -Z]
        activation: Optional activation function to apply before visualization (e.g., sigmoid)
        is_feature_map: If True, the cubemap has more than 3 channels (feature map)
        is_depth_map: If True, the cubemap is a depth/offset map (1 channel, use diverging colormap)
    
    Returns:
        PIL Image of the cubemap visualization
    """
    face_names = ['+X', '-X', '+Y', '-Y', '+Z', '-Z']
    
    # Create figure with 2 rows x 3 columns for 6 faces
    fig = plt.figure(figsize=(15, 10))
    
    # For depth map, we want a consistent color scale across all faces
    if is_depth_map:
        all_values = []
        for texture in cubemap_textures:
            if activation is not None:
                with torch.no_grad():
                    texture = activation(texture)
            if hasattr(texture, 'detach'):
                texture_np = texture.detach().cpu().numpy()
            else:
                texture_np = np.array(texture)
            all_values.append(texture_np.flatten())
        all_values = np.concatenate(all_values)
        vmin, vmax = all_values.min(), all_values.max()
        # Make symmetric around zero for diverging colormap
        abs_max = max(abs(vmin), abs(vmax))
        vmin, vmax = -abs_max, abs_max
    
    for i, (texture, name) in enumerate(zip(cubemap_textures, face_names)):
        ax = fig.add_subplot(2, 3, i + 1)
        
        # Apply activation if provided
        if activation is not None:
            with torch.no_grad():
                texture = activation(texture)
        
        # Convert to numpy
        if hasattr(texture, 'detach'):
            texture_np = texture.detach().cpu().numpy()
        else:
            texture_np = np.array(texture)
        
        if is_depth_map:
            # For depth maps, visualize as a diverging heatmap (1 channel)
            # texture_np shape: (1, H, W)
            depth_values = texture_np.squeeze()  # (H, W)
            im = ax.imshow(depth_values, cmap='RdBu_r', vmin=vmin, vmax=vmax)
            ax.set_title(f'Face {name}')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        elif is_feature_map:
            # For feature maps, visualize the L2 norm of feature vectors
            # texture_np shape: (C, H, W)
            feature_norm = np.linalg.norm(texture_np, axis=0)  # (H, W)
            # Normalize for visualization
            feature_norm = (feature_norm - feature_norm.min()) / (feature_norm.max() - feature_norm.min() + 1e-8)
            ax.imshow(feature_norm, cmap='viridis')
            ax.set_title(f'Face {name} (norm)')
        else:
            # Transpose from (3, H, W) to (H, W, 3)
            texture_np = np.transpose(texture_np, (1, 2, 0))
            
            # Clip to valid range [0, 1]
            texture_np = np.clip(texture_np, 0, 1)
            
            ax.imshow(texture_np)
            ax.set_title(f'Face {name}')
        ax.axis('off')
    
    if is_depth_map:
        title = f"Cubemap Depth Map\n{index}\niter {step}"
    elif is_feature_map:
        title = f"Cubemap Feature Map\n{index}\niter {step}"
    else:
        title = f"Cubemap Textures\n{index}\niter {step}"
    fig.suptitle(title)
    fig.tight_layout()
    
    canvas = fig.canvas
    buffer = io.BytesIO()
    canvas.print_png(buffer)
    data = buffer.getvalue()
    buffer.write(data)
    img = Image.open(buffer)
    plt.close()
    
    return img


def get_training_pcd_plot(index, step, ro, rd, points_np, coord_scale, pt_plot_scale, center_ray_select, points_conf_scores_np=None, points_alpha_np=None):
    num_plots = 6 if points_conf_scores_np is not None else 4
    fig = plt.figure(figsize=(5 * num_plots, 12))

    H, W, _ = rd.shape

    cur_color = "orange"
    if points_conf_scores_np is not None:
        cur_color = get_colors(points_conf_scores_np)

    # set pt_plot_scale to the max of the points
    pt_plot_scale = max(np.abs(points_np[:, 0].max()), np.abs(points_np[:, 1].max()), np.abs(points_np[:, 2].max()))

    ax = fig.add_subplot(2, num_plots, 1, projection='3d')
    ax.view_init(elev=0., azim=90)
    ax.set_xlim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_ylim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_zlim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], c=cur_color, s=0.8 * coord_scale)
    ax.scatter(ro[0], ro[1], ro[2], c="red", s=10)
    ax.quiver(ro[0], ro[1], ro[2], rd[H//2, W//2, 0], rd[H//2, W//2, 1], rd[H//2, W//2, 2], length=2, alpha=1, color="blue")
    ax.set_title('Point Cloud View 1')

    ax = fig.add_subplot(2, num_plots, 2, projection='3d')
    ax.view_init(elev=0., azim=180)
    ax.set_xlim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_ylim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_zlim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], c=cur_color, s=0.8 * coord_scale)
    ax.scatter(ro[0], ro[1], ro[2], c="red", s=10)
    ax.quiver(ro[0], ro[1], ro[2], rd[H//2, W//2, 0], rd[H//2, W//2, 1], rd[H//2, W//2, 2], length=2, alpha=1, color="blue")
    ax.set_title('Point Cloud View 2')

    ax = fig.add_subplot(2, num_plots, 3, projection='3d')
    ax.view_init(elev=0., azim=270)
    if points_alpha_np is not None:
        cur_color = get_colors(points_alpha_np)
        ax.view_init(elev=0., azim=180)
    ax.set_xlim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_ylim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_zlim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], c=cur_color, s=0.8 * coord_scale)
    ax.scatter(ro[0], ro[1], ro[2], c="red", s=10)
    ax.quiver(ro[0], ro[1], ro[2], rd[H//2, W//2, 0], rd[H//2, W//2, 1], rd[H//2, W//2, 2], length=2, alpha=1, color="blue")
    ax.set_title('Point Cloud View 3')

    ax = fig.add_subplot(2, num_plots, 4, projection='3d')
    ax.view_init(elev=89.9, azim=90)
    ax.set_xlim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_ylim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_zlim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], c=cur_color, s=0.8 * coord_scale)
    ax.scatter(ro[0], ro[1], ro[2], c="red", s=10)
    ax.quiver(ro[0], ro[1], ro[2], rd[H//2, W//2, 0], rd[H//2, W//2, 1], rd[H//2, W//2, 2], length=2, alpha=1, color="blue")
    ax.set_title('Point Cloud View 1 Up')

    if points_conf_scores_np is not None:
        ax = fig.add_subplot(2, num_plots, 5)
        ax.scatter(range(len(points_conf_scores_np)), points_conf_scores_np)
        ax.set_title('Confidence Scores scatter plot')

        ax = fig.add_subplot(2, num_plots, 6)
        bins = np.linspace(0, 1, 100).tolist()
        ax.hist(points_conf_scores_np, bins=bins)
        ax.set_title('Confidence Scores histogram')

    selected_points_np = np.zeros_like(points_conf_scores_np)
    selected_points_np[center_ray_select] = 1
    points_scale = np.ones_like(points_conf_scores_np) * 0.2 * coord_scale
    points_scale[center_ray_select] = 1 * coord_scale
    points_alpha = np.ones_like(points_conf_scores_np) * 0.01
    points_alpha[center_ray_select] = 1
    cur_color = get_colors(selected_points_np)
    cur_color = np.concatenate([cur_color, points_alpha[:, None]], axis=-1)

    ax = fig.add_subplot(2, num_plots, num_plots+1, projection='3d')
    ax.view_init(elev=0., azim=90)
    ax.set_xlim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_ylim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_zlim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], c=cur_color, s=points_scale)
    ax.set_title('Point Cloud View 1')

    ax = fig.add_subplot(2, num_plots, num_plots+2, projection='3d')
    ax.view_init(elev=0., azim=180)
    ax.set_xlim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_ylim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_zlim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], c=cur_color, s=points_scale)
    ax.set_title('Point Cloud View 2')

    ax = fig.add_subplot(2, num_plots, num_plots+3, projection='3d')
    ax.view_init(elev=0., azim=270)
    ax.set_xlim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_ylim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_zlim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], c=cur_color, s=points_scale)
    ax.set_title('Point Cloud View 3')

    ax = fig.add_subplot(2, num_plots, num_plots+4, projection='3d')
    ax.view_init(elev=89.9, azim=90)
    ax.set_xlim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_ylim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_zlim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], c=cur_color, s=points_scale)
    ax.set_title('Point Cloud View 1 Up')
    
    if points_alpha_np is not None:
        ax = fig.add_subplot(2, num_plots, 11)
        ax.scatter(range(len(points_alpha_np)), points_alpha_np)
        ax.set_title('Confidence Scores scatter plot')

        ax = fig.add_subplot(2, num_plots, 12)
        bins = np.linspace(-1, 1, 100).tolist()
        ax.hist(points_alpha_np, bins=bins)
        ax.set_title('Confidence Scores histogram')

    fig.suptitle("Point Clouds\n%s\niter %d, num pts: %d" % (index, step, points_np.shape[0]))

    canvas = fig.canvas
    buffer = io.BytesIO()
    canvas.print_png(buffer)
    data = buffer.getvalue()
    buffer.write(data)
    img = Image.open(buffer)
    plt.close()

    return img


def get_training_pcd_single_plot(step, points_np, pt_plot_scale, points_conf_scores_np=None):
    fig = plt.figure(figsize=(5, 5))
    fig.tight_layout()

    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.view_init(elev=20., azim=90 + (step / 500) * (720. / 500))
    ax.set_xlim3d(-pt_plot_scale * 1.5, pt_plot_scale * 1.5)
    ax.set_ylim3d(-pt_plot_scale * 1.5, pt_plot_scale * 1.5)
    ax.set_zlim3d(-pt_plot_scale * 1.5, pt_plot_scale * 1.5)
    ax.set_axis_off()
    cur_color = "orange"
    if points_conf_scores_np is not None:
        cur_color = get_colors(points_conf_scores_np)
    ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], c=cur_color)

    fig.suptitle("iter %d\n#points: %d" % (step, points_np.shape[0]))
    fig.tight_layout()

    canvas = fig.canvas
    buffer = io.BytesIO()
    canvas.print_png(buffer)
    data = buffer.getvalue()
    buffer.write(data)
    img = Image.open(buffer)
    plt.close()
    
    return img


