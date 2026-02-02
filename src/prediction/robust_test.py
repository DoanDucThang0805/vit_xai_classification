import torch
import numpy as np

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from model.vgg16 import model as vgg16
from model.resnet50 import model as resnet50
from model.mobilenetv3_small import model as mobilenetv3_small
from model.mobileplantvit import model as mobileplantvit
from model.shufflenet import model as shufflenetv2
from model.squezzenet import model as squeezenet
from model.densnet121 import model as densenet121
from dataset.dataset import robust_test_dataset

models = {
    "VGG16": {
        "model": vgg16,
        "ckpt": "/media/icnlab/Data/Thang/plan_dieases/vit_xai/checkpoints/plantvillage/vgg16/run_20251019-171608/best_checkpoint.pth",
    },
    "ResNet50": {
        "model": resnet50,
        "ckpt": "/media/icnlab/Data/Thang/plan_dieases/vit_xai/checkpoints/plantvillage/resnet50/run_20251019-084733/best_checkpoint.pth",
    },
    "MobileNetV3_Small": {
        "model": mobilenetv3_small,
        "ckpt": "/media/icnlab/Data/Thang/plan_dieases/vit_xai/checkpoints/plantvillage/mobilenetv3_small/run_20251021-151012/best_checkpoint.pth",
    },
    "MobilePlantViT": {
        "model": mobileplantvit,
        "ckpt": "/media/icnlab/Data/Thang/plan_dieases/vit_xai/checkpoints/plantvillage/mobileplantvit/run_20260101-103938/best_checkpoint.pth",
    },
    "ShuffleNetV2": {
        "model": shufflenetv2,
        "ckpt": "/media/icnlab/Data/Thang/plan_dieases/vit_xai/checkpoints/plantvillage/shufflenetv2/run_20251022-132921/best_checkpoint.pth",
    },
    "SqueezeNet": {
        "model": squeezenet,
        "ckpt": "/media/icnlab/Data/Thang/plan_dieases/vit_xai/checkpoints/plantvillage/squeezenet/run_20251021-171131/best_checkpoint.pth",
    },
    "DenseNet121": {
        "model": densenet121,
        "ckpt": "/media/icnlab/Data/Thang/plan_dieases/vit_xai/checkpoints/plantvillage/densenet121/run_20251018-193243/best_checkpoint.pth",
    },
}


num_runs = 50
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
final_results = []

with torch.inference_mode():
    for model_name, cfg in models.items():
        print(f"\n=== Evaluating {model_name} ===")

        model = cfg["model"]
        checkpoint = torch.load(cfg["ckpt"], map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        acc_scores = []
        f1_macro_scores = []

        for run in range(num_runs):
            test_loader = DataLoader(
                robust_test_dataset,
                batch_size=batch_size,
                shuffle=True
            )

            all_preds, all_labels = [], []

            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                logits = model(images)
                preds = torch.argmax(logits, dim=1)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

            run_acc = accuracy_score(all_labels, all_preds)
            run_f1 = f1_score(all_labels, all_preds, average="macro")

            acc_scores.append(run_acc)
            f1_macro_scores.append(run_f1)

            print(
                f"[{model_name}] Run {run+1}/{num_runs} "
                f"| Acc={run_acc:.4f} | Macro-F1={run_f1:.4f}"
            )

        final_results.append({
            "model": model_name,
            "acc_mean": np.mean(acc_scores),
            "acc_std": np.std(acc_scores),
            "f1_macro_mean": np.mean(f1_macro_scores),
            "f1_macro_std": np.std(f1_macro_scores),
        })
    
df = pd.DataFrame(final_results)

df.to_csv("robust_multi_run_results.csv", index=False)
print("\nSaved results to robust_multi_run_results.csv")
print(df)
