from synthetic_generator import LineGerenator
import pandas as pd
import argparse
import random
import os
from collections import Counter
from utils import VOCAB
if __name__ == "__main__":
    random.seed(42)
    
    parser = argparse.ArgumentParser(description="Generate synthetic OCR data")
    parser.add_argument("--input", type=str, required=True, help="Path to text file (.txt)")
    parser.add_argument("--output_name", type=str, default="dataset_v1")
    parser.add_argument("--data_root", type=str, default="./data/synthetic")
    parser.add_argument("--font_dir", type=str, default="./fonts")
    args = parser.parse_args()

    out_dir = os.path.join(args.data_root, args.output_name)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(args.font_dir):
        raise FileNotFoundError(f"Không tìm thấy thư mục font tại: {args.font_dir}")
        
    font_paths = [
        os.path.join(args.font_dir, f)
        for f in os.listdir(args.font_dir)
        if f.lower().endswith((".ttf", ".otf"))
    ]

    if not font_paths:
        raise Exception("Thư mục font trống!")

    generator = LineGerenator()
    data = []
    num_sample = 0
    font_size=[22, 26, 30, 34, 38]
    
    with open(args.input, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:
        font = random.choice(font_paths)
        size = random.choice([22, 26, 30, 34, 38])
        margins = [random.randint(10, 20) for _ in range(4)]

        do_rotate = random.random() < 0.3
        do_blur = random.random() < 0.5
        do_dist = random.random() < 0.3

        try:
            img, _ = generator.generate(
                line,
                font,
                size,
                margins,
                1, 
                2, 
                True, 
                False,
                1,
                rotate_angle= random.randint(-3, 3) if do_rotate else 0, 
                blur=random.uniform(0.5, 1.5) if do_blur else 0,
                distorsion=do_dist
            )

            fname = f"{num_sample:07d}.jpg"
            save_path = os.path.join(out_dir, fname)
            
            img.save(
                save_path,
                quality=random.randint(60, 95),
                subsampling=0 
            )
            
            data.append([num_sample, line])
            num_sample += 1
            
        except Exception as e:
            continue

    df = pd.DataFrame(data, columns=["ID", "Text"])
    df.to_csv(os.path.join(out_dir, "result.csv"), index=False, encoding="utf-8-sig")