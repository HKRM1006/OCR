import numpy as np
import random
import cv2
import os
from PIL import Image

def generate_paper(size, texture_dir=None, mode="auto") -> Image.Image:
    H, W = size
    if mode == "auto":
        p = random.random()
        if p < 0.2: mode = "clean"
        elif p < 0.6: mode = "noise"
        else: mode = "texture"

    if mode == "clean":
        paper = np.ones((H, W), dtype=np.float32) * random.randint(245, 255)
    
    elif mode == "noise":
        paper = np.ones((H, W), dtype=np.float32) * 235
        noise = np.random.normal(0, 12, (H, W))
        paper = paper + noise
        
    elif mode == "texture":
        paper = None
        if texture_dir and os.path.exists(texture_dir):
            tex_files = [
                os.path.join(texture_dir, f)
                for f in os.listdir(texture_dir)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))
            ]
            if tex_files:
                tex_path = random.choice(tex_files)
                tex = Image.open(tex_path).convert("L").resize((W, H))
                paper = np.asarray(tex, dtype=np.float32)
                paper = cv2.GaussianBlur(paper, (5, 5), 0)
        
        if paper is None:
            paper = np.ones((H, W), dtype=np.float32) * random.randint(230, 245)

    paper = np.clip(paper, 0, 255).astype(np.uint8)
    return Image.fromarray(paper, mode="L")