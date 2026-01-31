import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

@torch.no_grad()
def evaluate_checkpoint(model, checkpoint_path, test_loader, device):
    # load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    acc = accuracy_score(all_labels, all_preds)
    micro_f1 = f1_score(all_labels, all_preds, average="macro")

    return acc, micro_f1

import os
from pathlib import Path
from model.resnet50 import model

checkpoint_dir = Path(__file__).parents[2] / "checkpoints"
data = "plantdoc"
model_name = "resnet50"
runtime1 = "run_20260110-144303"
runtime2 = "run_20260110-183214"
runtime3 = "run_20260110-210726"

checkpoint_paths = [
    checkpoint_dir / data / model_name / runtime1 / "best_checkpoint.pth",
    checkpoint_dir / data / model_name / runtime2 / "best_checkpoint.pth",
    checkpoint_dir /data / model_name / runtime3 / "best_checkpoint.pth"
]

from dataset.plantdoc_dataset import test_dataset
from torch.utils.data import DataLoader


test_ds = DataLoader(test_dataset, batch_size=32, shuffle=True)
acc_list = []
micro_f1_list = []

for ckpt_path in checkpoint_paths:
    acc, micro_f1 = evaluate_checkpoint(
        model=model,
        checkpoint_path=ckpt_path,
        test_loader=test_ds,
        device="cuda"
    )

    acc_list.append(acc)
    micro_f1_list.append(micro_f1)

    # print(f"{ckpt_path}: Acc={acc:.4f}, Micro-F1={micro_f1:.4f}")

acc_mean = np.mean(acc_list)
acc_std  = np.std(acc_list)

f1_mean = np.mean(micro_f1_list)
f1_std  = np.std(micro_f1_list)

print("\n===== FINAL RESULT =====")
print(f"Accuracy   : {acc_mean:.4f} ± {acc_std:.4f}")
print(f"Macro-F1   : {f1_mean:.4f} ± {f1_std:.4f}")
