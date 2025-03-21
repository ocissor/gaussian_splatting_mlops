# src/data_preprocessing.py
import open3d as o3d
import numpy as np

def load_point_cloud(file_path, target_points=5000, voxel_size=0.005):
    pcd = o3d.io.read_point_cloud(file_path)
    num_points = len(pcd.points)
    if num_points > target_points:
        pcd = pcd.voxel_down_sample(voxel_size)
        while len(pcd.points) > target_points:
            voxel_size *= 1.2
            pcd = pcd.voxel_down_sample(voxel_size)
    
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else np.random.rand(len(points), 3)
    if len(points) < target_points:
        extra_idx = np.random.choice(len(points), target_points - len(points))
        points = np.concatenate([points, points[extra_idx]])
        colors = np.concatenate([colors, colors[extra_idx]])
    
    scales = np.ones((target_points, 3)) * 0.01  # 3D scales
    print(f"Processed to {len(points)} points")
    return points, colors, scales

def visualize_point_cloud(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])