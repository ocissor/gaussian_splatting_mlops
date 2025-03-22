# src/api.py
from fastapi import FastAPI, UploadFile, HTTPException
import torch
from src.inference import load_model, infer
import numpy as np
import open3d as o3d

app = FastAPI(title="Gaussian Splatting API")

MODEL = None

@app.on_event("startup")
async def startup_event():
    global MODEL
    MODEL = load_model("model.pth")
    print("Model loaded at startup")

@app.post("/render", response_model=dict)
async def render_point_cloud(file: UploadFile):
    if not file.filename.endswith(".ply"):
        raise HTTPException(status_code=400, detail="Only .ply files are supported")
    
    with open("temp.ply", "wb") as f:
        f.write(await file.read())
    
    points, colors, scales = load_point_cloud("temp.ply")
    points = torch.tensor(points, dtype=torch.float32)
    colors = torch.tensor(colors, dtype=torch.float32)
    scales = torch.tensor(scales, dtype=torch.float32)
    
    camera_pos = torch.tensor([0., 0., 0.5], dtype=torch.float32)
    camera_rot = torch.eye(3, dtype=torch.float32)
    
    try:
        img = infer(MODEL, points, colors, scales, camera_pos, camera_rot, tile_size=16)
        return {"image_shape": list(img.shape), "message": "Rendering successful"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")