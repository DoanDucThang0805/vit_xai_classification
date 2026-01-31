import pandas as pd

# Results from PlantVillage (from the image, converted to decimals)
plantvillage_results = {
    "model_name": [
        "densenet121", "resnet50", "vgg16",
        "mobileplantvit", "shufflenetv2",
        "mobilenetv3_small", "squeezenetv2"
    ],
    "accuracy_pv": [0.9975, 0.9968, 0.9957, 0.9957, 0.9912, 0.9806, 0.9800],
    "macro_f1_pv": [0.9960, 0.9956, 0.9943, 0.9945, 0.9855, 0.9800, 0.9762]
}

# Results from PlantDoc (given by user)
plantdoc_results = {
    "model_name": [
        "densenet121", "mobilenetv3_small", "mobileplantvit",
        "resnet50", "shufflenetv2", "squeezenetv2", "vgg16"
    ],
    "accuracy_pd": [0.78, 0.71, 0.79, 0.78, 0.77, 0.53, 0.57],
    "macro_f1_pd": [0.75, 0.71, 0.75, 0.74, 0.72, 0.44, 0.49]
}

# Create DataFrames
df_pv = pd.DataFrame(plantvillage_results)
df_pd = pd.DataFrame(plantdoc_results)

# Merge on model_name
df = pd.merge(df_pv, df_pd, on="model_name")

# Calculate performance drops
df["accuracy_drop"] = df["accuracy_pv"] - df["accuracy_pd"]
df["macro_f1_drop"] = df["macro_f1_pv"] - df["macro_f1_pd"]

# Save to CSV
csv_path = "/media/icnlab/Data/Thang/plan_dieases/vit_xai/results/plantvillage_plantdoc_comparison.csv"
df.to_csv(csv_path, index=False)

df, csv_path
