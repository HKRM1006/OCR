import Levenshtein as lev
import torch
import numpy as np
import cv2
VOCAB = (
    ['<blank>'] +
    list("aáàảãạăắằẳẵặâấầẩẫậ"
         "eéèẻẽẹêếềểễệ"
         "iíìỉĩị"
         "oóòỏõọôốồổỗộơớờởỡợ"
         "uúùủũụưứừửữự"
         "yýỳỷỹỵ"
         "bcdđghklmnpqrstvxfwzj") +
    list("AÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬ"
         "EÉÈẺẼẸÊẾỀỂỄỆ"
         "IÍÌỈĨỊ"
         "OÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢ"
         "UÚÙỦŨỤƯỨỪỬỮỰ"
         "YÝỲỶỸỴ"
         "BCDĐGHKLMNPQRSTVXFWZJ") +
    ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] +
    [' ', '.', ',', '?', '!', '‘', '’',':', ';', '=', '_', '+', '—','–', '-', '~', '(', ')', '&', '»', '>', '<', '“', '”', '"', "'", '`', '#', '@', '%', '*', '/', '[', ']', '|']
)
def compute_metrics(preds, targets):
    correct_sentences = 0
    total_dist = 0
    total_chars = 0

    for p, t in zip(preds, targets):
        p, t = str(p), str(t)
        if p == t:
            correct_sentences += 1
        dist = lev.distance(p, t)
        total_dist += dist
        total_chars += len(t)

    w_acc = correct_sentences / len(targets) if len(targets) > 0 else 0
    cer = total_dist / total_chars if total_chars > 0 else 0

    return w_acc, cer

def collate_fn(batch):
    imgs, labels = zip(*batch)

    batch_size = len(imgs)
    C, H = imgs[0].shape[:2]
    widths = [img.shape[2] for img in imgs]
    max_w = max(widths)

    padded_imgs = torch.ones(batch_size, C, H, max_w)
    input_widths = torch.zeros(batch_size, dtype=torch.long)
    target_lengths = torch.zeros(batch_size, dtype=torch.long)
    noise = torch.randn(batch_size, C, H, max_w).abs() * 0.01
    padded_imgs = padded_imgs - noise
    
    for i in range(batch_size):
        w = imgs[i].shape[2]
        padded_imgs[i, :, :, :w] = imgs[i]
        input_widths[i] = w
        target_lengths[i] = len(labels[i])

    labels_concat = torch.cat(labels)
    return padded_imgs, labels_concat, input_widths, target_lengths

def preprocess_real_image(img, img_height=32):
    h, w = img.shape
    scale = img_height / h
    new_w = int(w * scale)
    img = cv2.resize(img, (new_w, img_height))

    img = img.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img).unsqueeze(0) 
    
    C, H, W = img_tensor.shape
    max_w = W + 40
    
    padded_img = torch.ones((C, H, max_w)) 
    padded_img[:, :, :W] = img_tensor
    img_tensor = padded_img.unsqueeze(0)
    
    return img_tensor