# src/model.py
import torch
import torch.nn as nn

class GaussianSplatting(nn.Module):
    def __init__(self, num_points):
        super(GaussianSplatting, self).__init__()
        self.means = nn.Parameter(torch.randn(num_points, 3))  # [N, 3] XYZ
        self.scales = nn.Parameter(torch.ones(num_points, 3) * 0.01)  # [N, 3] Scale per axis
        self.quats = nn.Parameter(torch.randn(num_points, 4))  # [N, 4] Quaternion
        self.colors = nn.Parameter(torch.ones(num_points, 3) * 0.5)  # [N, 3] RGB
        self.opacities = nn.Parameter(torch.ones(num_points) * 0.5)  # [N] Alpha

    def compute_covariance(self):
        quats = self.quats / torch.norm(self.quats, dim=-1, keepdim=True)
        rot = torch.zeros(self.means.shape[0], 3, 3, device=self.means.device)
        w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
        rot[:, 0, 0] = 1 - 2*y**2 - 2*z**2
        rot[:, 0, 1] = 2*x*y - 2*w*z
        rot[:, 0, 2] = 2*x*z + 2*w*y
        rot[:, 1, 0] = 2*x*y + 2*w*z
        rot[:, 1, 1] = 1 - 2*x**2 - 2*z**2
        rot[:, 1, 2] = 2*y*z - 2*w*x
        rot[:, 2, 0] = 2*x*z - 2*w*y
        rot[:, 2, 1] = 2*y*z + 2*w*x
        rot[:, 2, 2] = 1 - 2*x**2 - 2*y**2
        scale = torch.diag_embed(self.scales)
        cov = rot @ scale @ scale.transpose(-1, -2) @ rot.transpose(-1, -2)
        return cov

    def forward(self, points, colors, scales):
        cov = self.compute_covariance()
        return self.means, self.colors, cov, self.opacities

def perspective_projection(means, camera_pos, camera_rot, fov=60.0, img_size=(256, 256)):
    H, W = img_size
    device = means.device
    fov_rad = torch.deg2rad(torch.tensor(fov, device=device))
    fx = fy = (W / 2) / torch.tan(fov_rad / 2)
    
    points_camera = means - camera_pos
    points_camera = points_camera @ camera_rot.T
    
    x = points_camera[:, 0]
    y = points_camera[:, 1]
    z = points_camera[:, 2]
    z = z.clamp(min=1e-6)
    
    u = (fx * x / z) + (W / 2)
    v = (fy * y / z) + (H / 2)
    pixels = torch.stack([u, v], dim=-1)
    return pixels, z

def compute_gaussian_bounds(pixels, cov2d, scale_factor=3.0):
    """Compute 2D bounding box for each Gaussian based on covariance."""
    device = pixels.device
    eigenvalues = torch.linalg.eigvalsh(cov2d)  # [N, 2]
    radii = scale_factor * torch.sqrt(eigenvalues)  # 3-sigma bounds
    
    min_bounds = pixels - radii
    max_bounds = pixels + radii
    return min_bounds, max_bounds

def render_gaussians(means, colors, covariances, opacities, img_size=(256, 256),
                     camera_pos=torch.tensor([0., 0., 5.]), camera_rot=torch.eye(3), tile_size=16):
    """Tile-based Gaussian Splatting renderer."""
    device = means.device
    H, W = img_size
    
    # Project to 2D
    pixels, depths = perspective_projection(means, camera_pos, camera_rot, img_size=img_size)
    cov2d = covariances[:, :2, :2]  # 2D projection of covariance
    
    # Sort by depth (back to front)
    order = torch.argsort(depths, descending=True)
    pixels, colors, opacities, cov2d = pixels[order], colors[order], opacities[order], cov2d[order]
    depths = depths[order]
    
    # Compute Gaussian bounds
    min_bounds, max_bounds = compute_gaussian_bounds(pixels, cov2d)
    
    # Tile setup
    tiles_x, tiles_y = (W + tile_size - 1) // tile_size, (H + tile_size - 1) // tile_size
    img = torch.zeros(H, W, 3, device=device)
    
    # Assign Gaussians to tiles
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            tile_min = torch.tensor([tx * tile_size, ty * tile_size], dtype=torch.float32, device=device)
            tile_max = torch.tensor([(tx + 1) * tile_size, (ty + 1) * tile_size], dtype=torch.float32, device=device)
            
            # Find Gaussians overlapping this tile
            mask = (min_bounds[:, 0] < tile_max[0]) & (max_bounds[:, 0] > tile_min[0]) & \
                   (min_bounds[:, 1] < tile_max[1]) & (max_bounds[:, 1] > tile_min[1])
            tile_pixels = pixels[mask]
            tile_colors = colors[mask]
            tile_opacities = opacities[mask]
            tile_cov2d = cov2d[mask]
            tile_depths = depths[mask]
            
            if len(tile_pixels) == 0:
                continue
            
            # Render within tile
            for i in range(len(tile_pixels)):
                u, v = tile_pixels[i]
                u_int, v_int = int(u), int(v)
                if tile_min[0] <= u_int < tile_max[0] and tile_min[1] <= v_int < tile_max[1]:
                    cov = tile_cov2d[i]
                    inv_cov = torch.inverse(cov + torch.eye(2, device=device) * 1e-6)
                    dx = torch.tensor([u - tile_pixels[i][0], v - tile_pixels[i][1]], dtype=torch.float32, device=device)
                    gauss_weight = torch.exp(-0.5 * dx @ inv_cov @ dx)
                    alpha = tile_opacities[i] * gauss_weight
                    
                    current_color = img[v_int, u_int]
                    img[v_int, u_int] = current_color * (1 - alpha) + tile_colors[i] * alpha
    
    return img.clamp(0, 1)

if __name__ == "__main__":
    num_points = 5000
    model = GaussianSplatting(num_points)
    points = torch.randn(num_points, 3)
    colors = torch.rand(num_points, 3)
    scales = torch.ones(num_points, 3) * 0.01
    means, colors, cov, opacities = model(points, colors, scales)
    img = render_gaussians(means, colors, cov, opacities, camera_pos=torch.tensor([0., 0., 0.5]))
    print(f"Rendered image shape: {img.shape}")