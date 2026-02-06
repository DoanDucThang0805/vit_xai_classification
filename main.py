import numpy as np
import pandas as pd

models = {
    "VGG16": {
        "lan1": {"accuracy": 0.9975, "macro_f1": 0.9960},
        "lan2": {"accuracy": 0.9888, "macro_f1": 0.9877},
        "lan3": {"accuracy": 0.9862, "macro_f1": 0.9850},
    },
    "ResNet50": {
        "lan1": {"accuracy": 0.9968, "macro_f1": 0.9956},
        "lan2": {"accuracy": 0.9950, "macro_f1": 0.9949},
        "lan3": {"accuracy": 0.9925, "macro_f1": 0.9912},
    },
    "DenseNet121": {
        "lan1": {"accuracy": 0.9975, "macro_f1": 0.9960},
        "lan2": {"accuracy": 0.9963, "macro_f1": 0.9961},
        "lan3": {"accuracy": 0.9944, "macro_f1": 0.9941},
    },
    "MobilePlantViT": {
        "lan1": {"accuracy": 0.9957, "macro_f1": 0.9945},
        "lan2": {"accuracy": 0.9931, "macro_f1": 0.9928},
        "lan3": {"accuracy": 0.9931, "macro_f1": 0.9921},
    },
    "MobileNetV3-Small": {
        "lan1": {"accuracy": 0.9806, "macro_f1": 0.9800},
        "lan2": {"accuracy": 0.9938, "macro_f1": 0.9933},
        "lan3": {"accuracy": 0.9925, "macro_f1": 0.9927},
    },
    "SqueezeNetV2": {
        "lan1": {"accuracy": 0.9800, "macro_f1": 0.9762},
        "lan2": {"accuracy": 0.9788, "macro_f1": 0.9730},
        "lan3": {"accuracy": 0.9813, "macro_f1": 0.9767},
    },
    "ShuffleNetV2": {
        "lan1": {"accuracy": 0.9912, "macro_f1": 0.9855},
        "lan2": {"accuracy": 0.9888, "macro_f1": 0.9867},
        "lan3": {"accuracy": 0.9688, "macro_f1": 0.9589},
    },
}

rows = []

for model_name, runs in models.items():
    accs = [v["accuracy"] for v in runs.values()]
    f1s  = [v["macro_f1"] for v in runs.values()]

    rows.append({
        "model": model_name,
        "acc_mean": round(np.mean(accs), 4),
        "acc_std": round(np.std(accs), 4),          # ddof=1 nếu muốn sample std
        "f1_macro_mean": round(np.mean(f1s), 4),
        "f1_macro_std": round(np.std(f1s), 4),
    })

df = pd.DataFrame(rows)
print(df)
