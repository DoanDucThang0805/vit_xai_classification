"""
F1 Score and Classification Metrics Evaluation.

This module evaluates trained models on test datasets and generates comprehensive
classification reports including precision, recall, and F1-score per class.
Uses the MobileVITXXS model architecture for evaluation.

Output includes:
    - Detailed classification report (per-class metrics)
    - Confusion matrix visualization
    - Model predictions and true labels
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from dataset.dataset import test_dataset
from model.mobilenetv3_small import model


# ==================== CONFIGURATION ====================
MODEL_NAME = "mobilevitxxs"
MODEL_RUN_ID = "run_20251122-011952"
DATA = "plantdoc"
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== HELPER FUNCTIONS ====================
def load_checkpoint(checkpoint_path: Path) -> Dict:
    """Load model checkpoint from file."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    return checkpoint


def extract_state_dict(checkpoint: Dict) -> Dict:
    """Extract state dict from checkpoint with multiple format support."""
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    else:
        return checkpoint


def setup_model(model_obj, checkpoint_path: Path) -> None:
    """Load checkpoint and prepare model for evaluation."""
    checkpoint = load_checkpoint(checkpoint_path)
    state_dict = extract_state_dict(checkpoint)
    
    model_obj.load_state_dict(state_dict)
    model_obj = model_obj.to(DEVICE)
    model_obj.eval()


def get_predictions(model_obj, dataloader: DataLoader) -> Tuple[List, List]:
    """Generate predictions on test dataset."""
    all_preds = []
    all_labels = []
    
    print(f"Starting model {MODEL_NAME} evaluation...")
    
    with torch.inference_mode(True):
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model_obj(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    return all_preds, all_labels


def compute_metrics(all_preds: List, all_labels: List, target_names: List[str]) -> Dict:
    """Compute classification metrics and per-class F1-scores."""
    report_dict = classification_report(
        all_labels,
        all_preds,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )
    
    per_class_f1 = {
        cls: report_dict[cls]['f1-score']
        for cls in target_names if cls in report_dict
    }
    
    return per_class_f1, report_dict


def save_results(per_class_f1: Dict, report_dir: str) -> str:
    """Save F1-scores to CSV file."""
    os.makedirs(report_dir, exist_ok=True)
    
    per_class_df = pd.DataFrame(
        list(per_class_f1.items()),
        columns=['Class', 'F1-Score']
    )
    
    output_csv_path = f"{report_dir}/per_class_f1_scores.csv"
    per_class_df.to_csv(output_csv_path, index=False)
    
    return output_csv_path, per_class_df


def print_metrics(per_class_f1: Dict, per_class_df: pd.DataFrame, output_csv_path: str) -> None:
    """Print evaluation results to console."""
    print("\n" + "="*60)
    print(f"Per-class F1-scores for {MODEL_NAME}:")
    print("="*60)
    for cls, f1 in per_class_f1.items():
        print(f"  {cls:30s}: {f1:.4f}")
    
    print("\n" + "="*60)
    print("Results Summary:")
    print("="*60)
    print(per_class_df.to_string(index=False))
    
    print(f"\n✅ F1-scores saved to: {output_csv_path}")
    print("="*60 + "\n")


# ==================== MAIN EXECUTION ====================
def main():
    """Main evaluation pipeline."""
    
    # Initialize data loader
    test_ds = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    target_names = [test_dataset.idx_to_class[i] for i in range(len(test_dataset.idx_to_class))]
    
    # Setup checkpoint path
    checkpoint_path = (
        Path(__file__).resolve().parents[2] / 'checkpoints' / DATA / MODEL_NAME / 
        MODEL_RUN_ID / 'best_checkpoint.pth'
    )
    
    # Load model
    setup_model(model, checkpoint_path)
    
    # Get predictions
    all_preds, all_labels = get_predictions(model, test_ds)
    
    # Compute metrics
    per_class_f1, report_dict = compute_metrics(all_preds, all_labels, target_names)
    
    # Setup output directory
    report_dir = f"/media/icnlab/Data/Thang/plan_dieases/vit_xai/reports/{DATA}/{MODEL_NAME}"
    
    # Save results
    output_csv_path, per_class_df = save_results(per_class_f1, report_dir)
    
    # Print results
    print_metrics(per_class_f1, per_class_df, output_csv_path)


if __name__ == "__main__":
    main()