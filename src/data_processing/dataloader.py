import os
import cv2
import pandas as pd
import torch
import random
from torch.utils.data import Dataset
import numpy as np

class OCRDataset(Dataset):
    def __init__(self, csv_path, img_dir, vocab, img_height=32):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.vocab = vocab
        self.char2idx = {c: i for i, c in enumerate(vocab)}
        self.img_height = img_height
        self.df['full_path'] = self.df['ID'].apply(self._find_img)

    def _find_img(self, img_id):
        base_name = f"{int(img_id):07d}"
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            path = os.path.join(self.img_dir, base_name + ext)
            if os.path.exists(path): return path
        return None

    def __len__(self):
        return len(self.df)

    def encode(self, text):
        return torch.tensor([self.char2idx[c] for c in str(text) if c in self.char2idx], dtype=torch.long)
    
    def decode(self, indices):
        return ''.join([self.vocab[i] for i in indices])

    def preprocess(self, img):
        h, w = img.shape
        scale = self.img_height / h
        new_w = int(w * scale)
        img = cv2.resize(img, (new_w, self.img_height))
        return img.astype(np.float32) / 255.0

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['full_path']
        if img_path is None:
            img = np.ones((self.img_height, 100), dtype=np.uint8) * 255
        else:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None: img = np.ones((self.img_height, 100), dtype=np.uint8) * 255
        img = self.preprocess(img)
        return torch.from_numpy(img).unsqueeze(0), self.encode(row["Text"])

class MixOCRDataset(Dataset):
    def __init__(self, syn_csv, syn_dir, real_csv, real_dir, vocab, img_height=32, real_ratio=0.5):
        self.vocab, self.img_height, self.real_ratio = vocab, img_height, real_ratio
        self.char2idx = {c: i for i, c in enumerate(vocab)}
        
        syn_df = pd.read_csv(syn_csv)
        syn_df["img_path"] = syn_df["ID"].apply(lambda x: os.path.join(syn_dir, f"{int(x):07d}.jpg"))
        syn_df["label"] = syn_df["Text"]
        
        real_df = pd.read_csv(real_csv)
        real_df["img_path"] = real_df.apply(lambda r: os.path.join(real_dir, f"page{int(r['page']):04d}", f"line{int(r['line']):04d}.png"), axis=1)
        real_df["label"] = real_df["text"]

        self.syn_df, self.real_df = syn_df[['img_path', 'label']], real_df[['img_path', 'label']]
        self.syn_len, self.real_len = len(self.syn_df), len(self.real_df)
        self.epoch_len = int(self.real_len / real_ratio) if real_ratio > 0 else self.syn_len
        self.on_epoch_end()

    def on_epoch_end(self):
        num_real = int(self.epoch_len * self.real_ratio)
        chosen_real = random.sample(range(self.real_len), min(num_real, self.real_len))
        real_samples = [(idx, True) for idx in chosen_real]
        syn_samples = [(random.randint(0, self.syn_len - 1), False) for _ in range(self.epoch_len - len(real_samples))]
        self.epoch_indices = real_samples + syn_samples
        random.shuffle(self.epoch_indices)

    def encode(self, text):
        return torch.tensor([self.char2idx[c] for c in str(text) if c in self.char2idx], dtype=torch.long)

    def decode(self, indices):
        return ''.join([self.vocab[i] for i in indices])
    
    def preprocess(self, img):
        h, w = img.shape
        img = cv2.resize(img, (int(w * (self.img_height / h)), self.img_height))
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return img.astype(np.float32) / 255.0

    def __getitem__(self, idx):
        d_idx, is_real = self.epoch_indices[idx]
        row = self.real_df.iloc[d_idx] if is_real else self.syn_df.iloc[d_idx]
        img = cv2.imread(row["img_path"], cv2.IMREAD_GRAYSCALE)
        if img is None: img = np.ones((self.img_height, 100), dtype=np.uint8) * 255
        img = self.preprocess(img)
        return torch.from_numpy(img).unsqueeze(0), self.encode(row["label"])

    def __len__(self):
        return len(self.epoch_indices)