from src.model.NeuralNetwork import CRNN
from src.utils import preprocess_real_image
import argparse
import torch
import cv2
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OCR model")
    parser.add_argument("--output", type=str, default="output.txt")
    parser.add_argument("--mode", type=str, default="File")
    parser.add_argument("--path", type=str, default="input")
    parser.add_argument("--model", type=str, default="pretrain.pth")
    args = parser.parse_args()
    checkpoint = torch.load(args.model, map_location=device)
    model = CRNN(checkpoint["vocab"], device=device)
    model.load_state_dict(checkpoint["model_state"])
    if args.mode == "File":
        model.eval()
        img = cv2.imread(args.path, cv2.IMREAD_GRAYSCALE)
        _, img = cv2.threshold(
            img, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        img_tensor = preprocess_real_image(img, 80).to(device)
        with torch.no_grad():
            pred = model(img_tensor)
            text, conf = model.ctc_decode_batch(pred)
            print(f"Kết quả: {text} | Độ tự tin: {conf}")
    elif args.mode == "Folder":
        model.eval()
        results = []
        image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

        for filename in os.listdir(args.path):
            if filename.lower().endswith(image_exts):
                img_path = os.path.join(args.path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.GaussianBlur(img, (3,3), 0)
                _, img = cv2.threshold(
                    img, 0, 255,
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                img = preprocess_real_image(img, 80)
                with torch.no_grad():
                    pred = model(img)
                    text, conf = model.ctc_decode_batch(pred)

                print(f"{filename}: {text[0]} ({conf[0]:.4f})")
                results.append(f"{text[0]}")

        with open(args.output, "w", encoding="utf-8") as f:
            for line in results:
                f.write(line + "\n")