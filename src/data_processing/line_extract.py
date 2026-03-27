import cv2 
import numpy as np
import os
import argparse

def extract_longest_line(image_bgr):  
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 10))
    merged = cv2.dilate(bw, kernel, iterations=1)
    merged = cv2.morphologyEx(merged, cv2.MORPH_OPEN, np.ones((3, 10), np.uint8), iterations=1)
    
    n, labels, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    best_id, best_w = None, -1
    for i in range(1, n):
        w = stats[i, cv2.CC_STAT_WIDTH]
        if w > best_w:
            best_id, best_w = i, w
            
    if best_id is None:
        return image_bgr
    
    comp_mask = (labels == best_id).astype(np.uint8) * 255
    comp_mask = cv2.dilate(comp_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)
    
    background = cv2.medianBlur(image_bgr, 51)
    line_only = cv2.bitwise_and(image_bgr, image_bgr, mask=comp_mask)
    background_only = cv2.bitwise_and(background, background, mask=cv2.bitwise_not(comp_mask))
    return cv2.add(line_only, background_only)

def line_extract(input_folder, output_folder):
    base_dir = os.getcwd()
    input_path = os.path.join(base_dir, "data", "raw", input_folder)
    output_path = os.path.join(base_dir, "data", "processed", output_folder)
    os.makedirs(output_path, exist_ok=True)
    filenames = sorted([f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    for page_idx, filename in enumerate(filenames, start=1):
        filepath = os.path.join(input_path, filename)
        image = cv2.imread(filepath)

        page_dir = os.path.join(output_path, f"page{page_idx:04d}")
        os.makedirs(page_dir, exist_ok=True)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(gray, 0, 100, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 10))
        dilated = cv2.dilate(threshold, kernel, iterations=1)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])
        
        line_count = 1
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            
            if w < 90 or h < 20 or (w > 1500 and h > 500): 
                continue
                
            cropped = image[y:y+h, x:x+w]
            processed_line = extract_longest_line(cropped)
            
            if processed_line is not None:
                save_path = os.path.join(page_dir, f"line{line_count:04d}.png")
                cv2.imwrite(save_path, processed_line)
                line_count += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trích xuất dòng chữ từ ảnh Light Novel")
    parser.add_argument("--input", type=str, required=True, help="Tên thư mục con trong data/raw")
    parser.add_argument("--output", type=str, required=True, help="Tên thư mục con trong data/processed")
    
    args = parser.parse_args()
    line_extract(args.input, args.output)