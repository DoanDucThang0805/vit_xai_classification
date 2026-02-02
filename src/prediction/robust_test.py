"""
Model Inference and Evaluation Script (Multi-run Version).

Features:
    - Runs inference multiple times to account for randomness in robust transformations.
    - Calculates Average Accuracy and Average Micro-F1 Score.
"""

import os
from pathlib import Path
import numpy as np # Import numpy để tính trung bình
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score # Thêm metrics cụ thể
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Import dataset và model của bạn
from dataset.dataset import robust_test_dataset
from model.mobileplantvit import model

# --- CẤU HÌNH ---
model_name = 'mobileplantvit'
run_time = 'run_20260101-103938'
data = 'plantvillage'
num_class = 10
num_runs = 50  # <--- SỐ LẦN CHẠY TEST (Bạn có thể sửa thành 3, 5, 10 tùy ý)
batch_size = 32

# Setup thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- LOAD CHECKPOINT (Giữ nguyên logic của bạn) ---
checkpoint_path = Path(__file__).resolve().parents[2] / 'checkpoints' / data / model_name / run_time / 'best_checkpoint.pth'
if not checkpoint_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

print(f"Loading model from: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=device)

if "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
elif "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)
model = model.to(device)
model.eval() # Chuyển sang chế độ đánh giá

# --- CHUẨN BỊ DỮ LIỆU ---
# Lưu ý: robust_test_dataset thường có random augmentation mỗi lần gọi __getitem__
test_ds = DataLoader(robust_test_dataset, batch_size=batch_size, shuffle=True)

# --- VÒNG LẶP ĐÁNH GIÁ (MULTI-RUN EVALUATION) ---
acc_scores = []
f1_macro_scores = []

print(f"\nStarting evaluation over {num_runs} runs on '{device}'...")
print("-" * 60)

# Tắt gradient để tiết kiệm bộ nhớ
with torch.inference_mode():
    for run in range(num_runs):
        all_preds = []
        all_labels = []
        
        # Chạy qua toàn bộ tập test
        for images, labels in test_ds:
            images, labels = images.to(device), labels.to(device)
            
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
        
        # Tính toán metrics cho lần chạy này
        run_acc = accuracy_score(all_labels, all_preds)
        run_f1 = f1_score(all_labels, all_preds, average='macro')
        
        # Lưu lại kết quả
        acc_scores.append(run_acc)
        f1_macro_scores.append(run_f1)
        
        print(f"Run {run+1}/{num_runs}: Accuracy = {run_acc:.4f} | Macro F1 = {run_f1:.4f}")

# --- TÍNH TOÁN KẾT QUẢ TRUNG BÌNH ---
mean_acc = np.mean(acc_scores)
std_acc = np.std(acc_scores)

mean_f1 = np.mean(f1_macro_scores)
std_f1 = np.std(f1_macro_scores)

print("-" * 60)
print("FINAL RESULTS (Mean ± Std):")
print(f"Average Accuracy: {mean_acc:.4f} ± {std_acc:.4f} ({(mean_acc*100):.2f}%)")
print(f"Average Micro F1: {mean_f1:.4f} ± {std_f1:.4f} ({(mean_f1*100):.2f}%)")
print("-" * 60)
