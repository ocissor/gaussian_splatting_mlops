# src/train.py
import torch
import torch.nn as nn
from src.components.data_preprocessing import load_point_cloud
from src.components.model import GaussianSplatting, render_gaussians

def train(file_path="data/bun_zipper.ply", num_epochs=200):
    points_np, colors_np, scales_np = load_point_cloud(file_path)
    points = torch.tensor(points_np, dtype=torch.float32)
    colors = torch.tensor(colors_np, dtype=torch.float32)
    scales = torch.tensor(scales_np, dtype=torch.float32)
    
    model = GaussianSplatting(num_points=5000)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    camera_pos = torch.tensor([0., 0., 0.5], dtype=torch.float32)
    camera_rot = torch.eye(3, dtype=torch.float32)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        means, colors_pred, cov, opacities = model(points, colors, scales)
        
        pos_loss = nn.MSELoss()(means, points)
        color_loss = nn.MSELoss()(colors_pred, colors)
        loss = pos_loss + 0.1 * color_loss
        
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            img = render_gaussians(means, colors_pred, cov, opacities, 
                                 camera_pos=camera_pos, camera_rot=camera_rot, tile_size=16)
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}, Image mean: {img.mean().item():.4f}")
    
    torch.save(model.state_dict(), "model.pth")
    print("Model saved to model.pth")
    return model

if __name__ == "__main__":
    train()