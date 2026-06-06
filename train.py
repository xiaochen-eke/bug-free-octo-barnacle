import os
import sys
import math

# =====================================
# Path
# =====================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CURRENT_DIR)

MODELS_DIR = os.path.join(BACKEND_DIR, "models")
DATASETS_DIR = os.path.join(BACKEND_DIR, "datasets")

sys.path.insert(0, MODELS_DIR)
sys.path.insert(0, DATASETS_DIR)

# =====================================
# Imports
# =====================================

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import amp

from translation_dataset import TranslationDataset
from transformer import TransformerModel


# =====================================
# Config
# =====================================

VOCAB_SIZE = 32000
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 10

LR = 3e-4          # 🔥 提升学习率（关键优化）
WARMUP_STEPS = 4000


# =====================================
# Main
# =====================================

def main():

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", DEVICE)

    # =====================================
    # Dataset
    # =====================================

    train_dataset = TranslationDataset(
        csv_file=r"C:\DLdata\git\ecj\multilingual_transformer\backend\data\processed\train_with_lang.csv",
        sp_model_path=r"C:\DLdata\git\ecj\multilingual_transformer\backend\tokenizer\multilingual.model",
        max_length=MAX_LEN
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    print("Dataset Size:", len(train_dataset))

    # =====================================
    # Model
    # =====================================

    model = TransformerModel(vocab_size=VOCAB_SIZE).to(DEVICE)

    print("Parameters:", sum(p.numel() for p in model.parameters()))

    # =====================================
    # Loss
    # =====================================

    criterion = nn.CrossEntropyLoss(
        ignore_index=0,
        label_smoothing=0.1
    )

    # =====================================
    # Optimizer
    # =====================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-5
    )

    # =====================================
    # AMP
    # =====================================

    scaler = amp.GradScaler("cuda")

    # =====================================
    # Scheduler (Warmup + Cosine)
    # =====================================

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    total_steps = len(train_loader) * EPOCHS

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


    # =====================================
    # Checkpoints
    # =====================================

    checkpoint_dir = os.path.join(BACKEND_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_loss = float("inf")
    global_step = 0

    # =====================================
    # Training Loop
    # =====================================

    for epoch in range(EPOCHS):

        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):

            src = batch["src"].to(DEVICE, non_blocking=True)
            tgt = batch["tgt"].to(DEVICE, non_blocking=True)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            optimizer.zero_grad()

            with amp.autocast(device_type="cuda"):

                logits = model(src, tgt_input)

                loss = criterion(
                    logits.reshape(-1, VOCAB_SIZE),
                    tgt_output.reshape(-1)
                )

            scaler.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            scheduler.step()
            global_step += 1

            total_loss += loss.item()

            # =====================================
            # Metrics
            # =====================================

            ppl = torch.exp(loss).item()

            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                mask = (tgt_output != 0)
                acc = ((pred == tgt_output) & mask).float().sum() / mask.float().sum()

            # =====================================
            # Logging
            # =====================================

            if batch_idx % 50 == 0:
                print(
                    f"Epoch [{epoch+1}/{EPOCHS}] "
                    f"Batch [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"PPL: {ppl:.2f} "
                    f"Acc: {acc.item():.4f} "
                    f"LR: {scheduler.get_last_lr()[0]:.6f}"
                )

        avg_loss = total_loss / len(train_loader)

        print(f"\nEpoch {epoch+1} Avg Loss: {avg_loss:.4f}")

        # =====================================
        # Save Best Model
        # =====================================

        if avg_loss < best_loss:

            best_loss = avg_loss

            save_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)

            print("Best model saved:", save_path)

        # =====================================
        # BLEU Validation Hook (Placeholder)
        # =====================================

        print("Run validation BLEU here (next upgrade step)")

    print("Training Finished.")


# =====================================
# Windows Safe Entry
# =====================================

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()