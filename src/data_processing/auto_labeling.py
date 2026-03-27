import easyocr
import cv2
import os
import pandas as pd
import argparse
def labeling(input:str ,output:str):
    reader = easyocr.Reader(['vi'], gpu=False)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(BASE_DIR, "data", "processed", input)
    header = ["page", "line", "text", "conf"]
    data = []
    for page in os.listdir(folder_path):
        page_path = os.path.join(folder_path, page)
        for line in os.listdir(page_path):
            img_path = os.path.join(page_path, line)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            _, img = cv2.threshold(
                img, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            result = reader.readtext(
                img,
                paragraph=False,
                min_size=20,
                text_threshold=0.4,
                low_text=0.2,
                link_threshold=0.2,
                batch_size=4
            )
            if result:
                data.append([int(page[4:8]), int(line[4:8]), result[0][1], result[0][2]])
            else:
                data.append([int(page[4:8]), int(line[4:8]), '', 0])
    dataframe = pd.DataFrame(data, columns=header)
    dataframe.to_csv("D:/ocr_lightnovel/data/autolabel/" + output, index=False, encoding="utf-8-sig")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Auto labeling data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.csv",
        help="Output CSV"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="Input folder"
    )
    args = parser.parse_args()
    labeling(args.input, args.output)

