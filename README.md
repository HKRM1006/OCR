# Vietnamese Novel OCR Pipeline

This repository contains a comprehensive end-to-end pipeline for training a specialized OCR system for Vietnamese, specifically optimized for the typography and layouts found in Light Novels. The system utilizes a CRNN architecture and a hybrid training strategy.

## Key Features

* **Synthetic Data Generation**: Automatically generates large-scale labeled text-line images from raw text using various fonts, background textures, and augmentation effects.
* **MixOCR Dataset Strategy**: A custom PyTorch Dataset that dynamically mixes Synthetic and Real-world data at a specific ratio to prevent overfitting and improve generalization.
* **Two-Stage Finetuning**: Supports both `Stabilize` (frozen CNN backbone) and `Full` (discriminative learning rates) modes.

---

## Project Structure

```text
.
├── src/
│   ├── model/                 # CRNN Architecture (CNN + Bi-LSTM + CTC), train and finetune script
│   ├── data_processing/       # Dataset, Line from image and autolabel using EasyOCR  
│   └── synthetic_generator/   # Synthetic data generate logic
├── generate_synthetic_data.py # Script to generate synthetic images from .txt
├── utils.py                   # Vocabulary, Metrics (CER/WER), Collate functions
└── README.md                  # Description
```

## Execution Guide
### 1. Preparing Real-World Data
Your real-world dataset should follow this structure for the `MixOCRDataset` to work correctly:
- **Image Directory**: `path/to/real_images/page0001/line0001.png`
- **CSV Label File**: A CSV file with the following columns:
  - `page`: Page number (e.g., 1)
  - `line`: Line number (e.g., 1)
  - `text`: The ground truth transcription

### 2. Synthetic Data Generation
Before running the generator, ensure you have:
1. **Text Corpus**: A `.txt` file where each line is a sentence/phrase you want to render.
2. **Fonts**: A folder containing `.ttf` or `.otf` files that support Vietnamese characters.
Run the following command to start generating your dataset:
```bash
python src/generate_synthetic_data.py \
    --input your_text_file.txt \
    --output_name dataset_v1 \
    --font_dir ./fonts \
    --data_root ./data/synthetic
```
You can modify the parameter in the file for suitable data. After running, you will have a folder structured as follows:
```text
data/synthetic/dataset_v1/
├── 0000000.jpg
├── 0000001.jpg
├── ...
└── result.csv   # Contains mapping: ID, Text
```
### 3. Training (Synthetic Data)
Train the CRNN model from scratch using the generated synthetic data to learn the Vietnamese character set.
```bash
python train.py \
    --name base_run \
    --train_directory dataset_v1 \
    --eval_directory eval_set \
    --data_root ./data/synthetic \
    --epoch 50 \
    --learning_rate 1e-4
```
The model will be saved at ./models/CRNN/base_run/
### 4. Finetune
#### Stage 1: Stabilize (CNN Frozen)
Run this first to adapt the sequence modeling to the new font styles.
```bash
python finetune.py \
    --type Stabilize \
    --model ./models/base_crnn.pth \
    --synthetic_path ./data/synthetic \
    --real_csv ./data/real/labels.csv \
    --real_dir ./data/real/images \
    --epoch 10 \
    --learning_rate 1e-4
```
Stage 2: Full Finetuning (All Layers)
```bash
python finetune.py \
    --type Full \
    --model ./checkpoints/CRNN_Stabilize/base_model/epoch_10.pth \
    --synthetic_path ./data/synthetic \
    --real_csv ./data/real/labels.csv \
    --real_dir ./data/real/images \
    --epoch 20 \
    --learning_rate 1e-5
```
### 5. Inference
Once you have a trained model, use this script to transcribe individual images or entire folders of cropped text lines.
```bash
python predict.py \
    --mode File \
    --path ./test_samples/line_01.png \
    --model ./models/CRNN/base_run/epoch_50.pth
```
or for folder.
```bash
python predict.py \
    --mode Folder \
    --path ./data/real/page0001 \
    --model ./checkpoints/CRNN_Full/epoch_20.pth \
    --output transcriptions.txt
```
remember to use the line extract file to get the line image first

## Pre-trained Models
We provide pre-trained checkpoints to help you get started quickly.
| Model | Download Link |
| :--- | :--- | 
| **Synthetic** | [Download (Google Drive)]([#](https://drive.google.com/file/d/1meGSS89eqnwyaBvFpKXI_egs6KrEUw-z/view?usp=sharing)) |

