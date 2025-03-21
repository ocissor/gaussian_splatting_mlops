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
    """Project 3D points to 2D using a perspective camera."""
    H, W = img_size
    device = means.device
    
    # Camera parameters
    fov_rad = torch.deg2rad(torch.tensor(fov, device=device))
    fx = fy = (W / 2) / torch.tan(fov_rad / 2)  # Focal length
    
    # Transform points to camera space
    points_camera = means - camera_pos  # Translate to camera origin
    points_camera = points_camera @ camera_rot.T  # Rotate to camera orientation
    
    # Perspective projection
    x = points_camera[:, 0]
    y = points_camera[:, 1]
    z = points_camera[:, 2]
    
    # Avoid division by zero
    z = z.clamp(min=1e-6)
    
    # Project to image plane
    u = (fx * x / z) + (W / 2)  # X -> pixel U
    v = (fy * y / z) + (H / 2)  # Y -> pixel V
    
    pixels = torch.stack([u, v], dim=-1)
    return pixels, z  # Return 2D coords and depth

def render_gaussians(means, colors, covariances, opacities, img_size=(256, 256),
                     camera_pos=torch.tensor([0., 0., 5.]), camera_rot=torch.eye(3)):
    """Render 3D Gaussians with a perspective camera."""
    device = means.device
    H, W = img_size
    img = torch.zeros(H, W, 3, device=device)
    depth = torch.ones(H, W, device=device) * float('inf')
 
    
    # Project points to 2D
    pixels, z = perspective_projection(means, camera_pos, camera_rot, img_size=img_size)
    
    # Sort by depth (back to front)
    order = torch.argsort(z, descending=True)
    pixels = pixels[order]
    colors = colors[order]
    opacities = opacities[order]
    cov2d = covariances[order, :2, :2]  # 2D projection of covariance
    
    # Render each Gaussian
    for i in range(means.shape[0]):
        u, v = pixels[i]
        u_int, v_int = int(u), int(v)
        if 0 <= u_int < W and 0 <= v_int < H:
            cov = cov2d[i]
            inv_cov = torch.inverse(cov + torch.eye(2, device=device) * 1e-6)
            dx = torch.tensor([u - pixels[i][0], v - pixels[i][1]], dtype=torch.float32, device=device)
            gauss_weight = torch.exp(-0.5 * dx @ inv_cov @ dx)
            alpha = opacities[i] * gauss_weight
            
            # Alpha blending (simplified, no depth buffer yet)
            if z[i]<depth[u,v]:
                depth[u,v] = z[i]
                current_color = img[v_int, u_int]
                img[v_int, u_int] = current_color * (1 - alpha) + colors[i] * alpha
    
    return img.clamp(0, 1)

if __name__ == "__main__":
    num_points = 5
    model = GaussianSplatting(num_points)
    points = torch.randn(num_points, 3)
    colors = torch.rand(num_points, 3)
    scales = torch.ones(num_points, 3) * 0.01
    means, colors, cov, opacities = model(points, colors, scales)
    img = render_gaussians(means, colors, cov, opacities)
    print(f"Rendered image shape: {img.shape}")