import os
from pathlib import Path
from typing import Dict, List

import torch
import pandas as pd
from sklearn.metrics import f1_score

from dataset.plantdoc_dataset import test_dataset
from model.mobileplantvitv2 import model as mobileplantvitv2_model
from model.resnet50 import model as resnet50_model
from model.vgg16 import model as vgg16_model
from model.densnet121 import model as densnet121_model
from model.mobilenetv3_small import model as mobilenetv3_model
from model.shufflenet import model as shufflenet_model
from model.squezzenet import model as squezzenet_model


class F1PerClassInference:
    checkpoint_path: Path
    device: torch.device
    save_path: str

    def __init__(self, checkpoint_path: Path, device: torch.device, save_path: str, model_instance=None):
        """
        Args:
            checkpoint_path: path tới file .pt / .pth
            device: torch.device (cpu / cuda)
            save_path: thư mục lưu kết quả
            model_instance: model instance để sử dụng (nếu None sẽ dùng mobileplantvitv2)
        """
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.save_path = save_path

        os.makedirs(save_path, exist_ok=True)

        self.model = model_instance if model_instance is not None else mobileplantvitv2_model
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.model.to(device)
        self.model.eval()

    def __call__(self):
        all_preds = []
        all_labels = []

        with torch.inference_mode(True):
            for images, labels in test_dataset:
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(images)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds = torch.tensor(all_preds)
        all_labels = torch.tensor(all_labels)

        f1_per_class = f1_score(
            all_labels,
            all_preds,
            average=None,
            labels=list(range(10))
        )

        # Save f1_per_class
        save_path = os.path.join(
            self.save_path,
            f"f1_per_class_{self.checkpoint_path.stem}.pt"
        )
        torch.save(f1_per_class, save_path)

        return f1_per_class


class F1ComparisonMultiModels:
    """
    Compare F1 scores across 7 models and save results to CSV
    """
    
    def __init__(self, checkpoint_dir: Path, device: torch.device, save_path: str):
        """
        Args:
            checkpoint_dir: thư mục chứa tất cả model checkpoints
            device: torch.device (cpu / cuda)
            save_path: thư mục lưu kết quả (sẽ tạo file CSV ở đây)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        self.save_path = save_path
        
        os.makedirs(save_path, exist_ok=True)
        
        # Mapping của 7 mô hình
        self.models_map: Dict[str, object] = {
            "MobilePlantVitV2": mobileplantvitv2_model,
            "ResNet50": resnet50_model,
            "VGG16": vgg16_model,
            "DenseNet121": densnet121_model,
            "MobileNetV3": mobilenetv3_model,
            "ShuffleNet": shufflenet_model,
            "SqueezeNet": squezzenet_model,
        }
    
    def run_inference_all_models(self, checkpoint_paths: Dict[str, Path]) -> pd.DataFrame:
        """
        Chạy inference cho tất cả 7 mô hình và tính F1 score cho từng class
        
        Args:
            checkpoint_paths: Dict mapping model name -> checkpoint path
            
        Returns:
            DataFrame với F1 score của từng class cho mỗi mô hình
        """
        results = {}
        
        for model_name, checkpoint_path in checkpoint_paths.items():
            print(f"\n--- Processing {model_name} ---")
            
            if model_name not in self.models_map:
                print(f"Warning: {model_name} not in models_map, skipping...")
                continue
            
            model_instance = self.models_map[model_name]
            
            # Tạo inference instance
            inference = F1PerClassInference(
                checkpoint_path=checkpoint_path,
                device=self.device,
                save_path=self.save_path,
                model_instance=model_instance
            )
            
            # Chạy inference và lấy F1 scores
            f1_per_class = inference()
            results[model_name] = f1_per_class
            
            print(f"✓ {model_name} - F1 scores: {f1_per_class}")
        
        # Tạo DataFrame từ results
        df_results = pd.DataFrame(results).T
        df_results.columns = [f"Class_{i}" for i in range(len(df_results.columns))]
        
        return df_results
    
    def save_results_to_csv(self, df_results: pd.DataFrame, filename: str = "f1_comparison_7models.csv"):
        """
        Lưu kết quả F1 scores vào file CSV
        
        Args:
            df_results: DataFrame chứa F1 scores
            filename: tên file CSV
        """
        csv_path = os.path.join(self.save_path, filename)
        df_results.to_csv(csv_path, index_label="Model")
        print(f"\n✓ Kết quả đã lưu vào: {csv_path}")
        
        # In chi tiết kết quả
        print("\n" + "="*80)
        print("F1 SCORE COMPARISON - 7 MODELS")
        print("="*80)
        print(df_results.to_string())
        print("="*80)
        
        # Tính toán thống kê
        print("\nSUMMARY STATISTICS:")
        print("-"*80)
        print("Mean F1 per model:")
        print(df_results.mean(axis=1).to_string())
        print("\nMean F1 per class:")
        print(df_results.mean(axis=0).to_string())
        print(f"\nOverall Mean F1 Score: {df_results.values.mean():.4f}")
        print("-"*80)
        
        return csv_path


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Thư mục chứa checkpoints (cần điều chỉnh theo cấu trúc thực tế)
    checkpoint_dir = Path(__file__).parent.parent.parent / "checkpoints" / "plantdoc"
    save_path = Path(__file__).parent.parent.parent / "reports" / "plantdoc" / "f1_comparison"
    
    # Tạo comparator
    comparator = F1ComparisonMultiModels(
        checkpoint_dir=checkpoint_dir,
        device=device,
        save_path=str(save_path)
    )
    
    # Cấu hình đường dẫn checkpoint cho mỗi mô hình
    # (Cần điều chỉnh theo đường dẫn thực tế của bạn)
    checkpoint_paths = {
        "MobilePlantVitV2": checkpoint_dir / "mobileplantvitv2_best.pt",
        "ResNet50": checkpoint_dir / "resnet50_best.pt",
        "VGG16": checkpoint_dir / "vgg16_best.pt",
        "DenseNet121": checkpoint_dir / "densnet121_best.pt",
        "MobileNetV3": checkpoint_dir / "mobilenetv3_small_best.pt",
        "ShuffleNet": checkpoint_dir / "shufflenet_best.pt",
        "SqueezeNet": checkpoint_dir / "squezzenet_best.pt",
    }
    
    # Chạy inference cho tất cả mô hình
    df_results = comparator.run_inference_all_models(checkpoint_paths)
    
    # Lưu kết quả vào CSV
    comparator.save_results_to_csv(df_results)
    