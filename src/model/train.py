from src.model.NeuralNetwork import CRNN
from src.data_processing.dataloader import OCRDataset
from src.utils import VOCAB, compute_metrics, collate_fn
from torch.utils.data import DataLoader
import argparse
import torch
import time
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OCR model")
    parser.add_argument("--name", type=str, default="output")
    parser.add_argument("--mode", type=str, default="CRNN")
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--train_directory", type=str, default="input")
    parser.add_argument("--eval_directory", type=str, default="eval")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to initial checkpoint")
    parser.add_argument("--data_root", type=str, default="./data", help="Root directory of data")
    parser.add_argument("--model_dir", type=str, default="./models", help="Directory to save models")
    args = parser.parse_args()
    data_root = args.data_root
    directory_path = os.path.join(data_root, args.train_directory)
    eval_path = os.path.join(data_root, args.eval_directory)
    save_dir = os.path.join(args.model_dir, args.mode, args.name)
    os.makedirs(save_dir, exist_ok=True)
    if args.mode == "CRNN":
        model = CRNN(VOCAB, device).to(device)
        criterion = torch.nn.CTCLoss(
            blank=0,
            zero_infinity=True
        )
        if args.checkpoint is not None:
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint["model_state"])
    else:
        exit()
    
    dataset_path = os.path.join(directory_path, "result.csv")
    eval_dataset_path = os.path.join(eval_path, "result.csv")
    dataset = OCRDataset(dataset_path, directory_path, VOCAB)
    eval_dataset = OCRDataset(eval_dataset_path, eval_path, VOCAB)

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr = float(args.learning_rate), weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3, threshold=1e-6)
    for epoch in range(args.epoch):
        start_time = time.perf_counter()
        model.train()
        total_loss = 0

        for imgs, targets, input_widths, target_lengths in loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device) 

            optimizer.zero_grad()

            log_probs = model(imgs) 
            
            current_batch_size, T_model, _ = log_probs.size()
            
            W_pad = imgs.size(-1)
            ratio = T_model / W_pad
            input_lengths = torch.floor(input_widths * ratio).long().to(device)
            input_lengths = torch.clamp(input_lengths, max=T_model)

            loss = criterion(
                log_probs.permute(1, 0, 2), #[T, B, C]
                targets,
                input_lengths,
                target_lengths
            )
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        text, conf = model.ctc_decode_batch(log_probs)
        print(f"TRAIN PRED: {text[:3]} | CONF: {conf[:3]}")
        avg_loss = total_loss / len(loader)
        print(f"[Epoch {epoch+1}/{args.epoch}] Train loss: {avg_loss:.4f}")

        model.eval()
        total_eval_loss = 0
        all_preds = []
        all_targets = []       
        with torch.no_grad():
            for imgs, targets, input_widths, target_lengths in eval_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                target_lengths = target_lengths.to(device)

                log_probs = model(imgs)
                current_batch_size, T_model, _ = log_probs.size()
                
                input_lengths = torch.floor(input_widths * (T_model / imgs.size(-1))).long().to(device)
                input_lengths = torch.clamp(input_lengths, max=T_model)

                loss = criterion(
                    log_probs.permute(1, 0, 2),
                    targets,
                    input_lengths,
                    target_lengths
                )
                total_eval_loss += loss.item()

                batch_preds, batch_confs = model.ctc_decode_batch(log_probs.permute(1, 0, 2))
                
                batch_targets = []
                current_pos = 0
                targets_cpu = targets.cpu().tolist()
                for length in target_lengths:
                    l = length.item()
                    target_indices = targets_cpu[current_pos : current_pos + l]
                    batch_targets.append(eval_dataset.decode(target_indices))
                    current_pos += l

                all_preds.extend(batch_preds)
                all_targets.extend(batch_targets)


        avg_eval_w_acc, avg_eval_cer = compute_metrics(all_preds, all_targets)
        avg_eval_loss = total_eval_loss / len(eval_loader)
        
        print("-" * 30)
        print(f"EPOCH {epoch+1} EVALUATION:")
        print(f"Loss: {avg_eval_loss:.4f} | CER: {avg_eval_cer:.4f} | W-Acc: {avg_eval_w_acc:.4f}")
        for i in range(min(3, len(all_preds))):
            print(f"Sample {i}: GT: '{all_targets[i]}' | PRED: '{all_preds[i]}'")
        print("-" * 30)

        scheduler.step(avg_eval_loss)
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loss": avg_loss,
                "vocab": VOCAB
            },
            os.path.join(save_dir, f"epoch_{epoch+1}.pth")
        )
        print("Saved model")
        print("Time:", time.perf_counter() - start_time)
