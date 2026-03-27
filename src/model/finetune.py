from src.model.NeuralNetwork import CRNN
from src.data_processing.dataloader import MixOCRDataset
from torch.utils.data import DataLoader
from src.utils import VOCAB, compute_metrics, collate_fn
import torch.nn.functional as F
import argparse
import torch
import time
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune OCR model with MixOCR")
    parser.add_argument("--synthetic_path", type=str, required=True)
    parser.add_argument("--real_csv", type=str, required=True)
    parser.add_argument("--real_dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, help="Path to base checkpoint")
    parser.add_argument("--mode", type=str, default="CRNN")
    parser.add_argument("--type", type=str, choices=["Stabilize", "Full"], default="Stabilize")
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    dataset = MixOCRDataset(
        syn_csv=os.path.join(args.synthetic_path, "result.csv"),
        syn_dir=args.synthetic_path,
        real_csv=args.real_csv,
        real_dir=args.real_dir,
        vocab=VOCAB,
        img_height=80, 
        real_ratio=0.5
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    if args.mode == "CRNN":
        model = CRNN(VOCAB, device).to(device)
        checkpoint = torch.load(args.model, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        model_name = os.path.basename(os.path.dirname(args.model))
    else:
        print("Mode not supported")
        exit(0)
        
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    if args.type == "Stabilize":
        model.freeze_cnn()
        train_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.AdamW(train_params, lr=args.learning_rate, weight_decay=1e-5)
    else:
        optimizer = torch.optim.AdamW([
            {'params': model.backbone.parameters(), 'lr': args.learning_rate / 10.0},
            {'params': model.recurLayer.parameters(), 'lr': args.learning_rate},
            {'params': model.fc.parameters(), 'lr': args.learning_rate}
        ], weight_decay=1e-5)

    save_path = os.path.join(args.save_dir, f"{args.mode}_{args.type}", model_name)
    os.makedirs(save_path, exist_ok=True)

    for epoch in range(args.epoch):
        start_time = time.perf_counter()
        model.train()
        if args.type == "Stabilize":
            model.backbone.eval()

        total_loss = 0.0
        all_preds, all_targets = [], []

        for imgs, targets, input_widths, target_lengths in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            target_lengths = target_lengths.to(device)

            optimizer.zero_grad()
            log_probs = model(imgs) 
            
            batch_size, T_model, _ = log_probs.size()
            W_img = imgs.size(-1)
            ratio = T_model / W_img
            input_lengths = torch.floor(input_widths * ratio).long().to(device)
            input_lengths = torch.clamp(input_lengths, max=T_model)

            loss = criterion(
                log_probs.permute(1, 0, 2),
                targets,
                input_lengths,
                target_lengths
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total_loss += loss.item()

            with torch.no_grad():
                batch_preds, _ = model.ctc_decode_batch(log_probs)
                all_preds.extend(batch_preds)
                
                current_pos = 0
                targets_cpu = targets.cpu().tolist()
                for length in target_lengths:
                    l = length.item()
                    all_targets.append(dataset.decode(targets_cpu[current_pos : current_pos + l]))
                    current_pos += l

        avg_loss = total_loss / len(loader)
        avg_w_acc, avg_cer = compute_metrics(all_preds, all_targets)
        elapsed = time.perf_counter() - start_time

        print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.4f} | CER: {avg_cer:.4f} | W-Acc: {avg_w_acc:.4f} | Time: {elapsed:.2f}s")
        
        for i in range(min(2, len(all_preds))):
            print(f"  > GT: {all_targets[i]} | PRED: {all_preds[i]}")

        torch.save({
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "vocab": VOCAB
        }, os.path.join(save_path, f"epoch_{epoch+1}.pth"))

        dataset.on_epoch_end()