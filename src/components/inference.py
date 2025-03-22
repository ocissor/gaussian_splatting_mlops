# src/inference.py
import torch
from src.components.model import GaussianSplatting, render_gaussians
from src.components.data_preprocessing import load_point_cloud

def load_model(model_path):
    model = GaussianSplatting(num_points=5000)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def infer(model, points, colors, scales, camera_pos=torch.tensor([0., 0., 0.5]), 
          camera_rot=torch.eye(3), tile_size=16):
    with torch.no_grad():
        means, colors_pred, cov, opacities = model(points, colors, scales)
        img = render_gaussians(means, colors_pred, cov, opacities, 
                             camera_pos=camera_pos, camera_rot=camera_rot, tile_size=tile_size)
    return img