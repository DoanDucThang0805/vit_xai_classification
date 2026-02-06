import numpy as np

gradcam_path = "/media/icnlab/Data/Thang/plan_dieases/vit_xai/src/gradcam_pss_all_models.npz"

data = np.load(gradcam_path)
# print(data['VGG16'])
filter_loc = [
    id for id, _ in enumerate(data['sigmas'])
    if _ not in [0.01, 0.03, 0.05, 0.07, 0.09, 0.1]
]
# print("Filter loc:", filter_loc)
def load_pss_gradcam(path: str):
    data = np.load(path)
    pss = {}
    for model_name in data.files:
        if model_name == "sigma":
            continue
        pss_values = data[model_name]
        pss[model_name] = [pss_values[i] for i in range(len(pss_values)) if i not in filter_loc]
    pss['sigmas'] = [data['sigmas'][i] for i in range(len(data['sigmas'])) if i not in filter_loc]
    return pss
pss_gradcam = load_pss_gradcam(gradcam_path)
# print(pss_gradcam)
print("PSS GradCAM:")
import pandas as pd

df = pd.DataFrame(pss_gradcam)
print(df)
print("="*90)
from typing import List
def load_pss__lime(paths: List[str]):
    pss = {}
    for path in paths:
        data = np.load(path)
        for item_name in data.files:
            pss[item_name] = data[item_name]
    return pss
list_lime_paths = [
    "/media/icnlab/Data/Thang/plan_dieases/vit_xai/src/lime_pss_all_models_densenet121.npz",
    "/media/icnlab/Data/Thang/plan_dieases/vit_xai/src/lime_pss_all_models_mobilenetv3_small.npz",
    "/media/icnlab/Data/Thang/plan_dieases/vit_xai/src/lime_pss_all_models_mobileplantvit.npz",
    "/media/icnlab/Data/Thang/plan_dieases/vit_xai/src/lime_pss_all_models_resnet50.npz",
    "/media/icnlab/Data/Thang/plan_dieases/vit_xai/src/lime_pss_all_models_shufflenetv2.npz",
    "/media/icnlab/Data/Thang/plan_dieases/vit_xai/src/lime_pss_all_models_squezzenet.npz",
    "/media/icnlab/Data/Thang/plan_dieases/vit_xai/src/lime_pss_all_models_vgg16.npz"
]
print("PSS LIME:")
pss_lime = load_pss__lime(list_lime_paths)
df_lime = pd.DataFrame(pss_lime)
print(df_lime)
print("="*90)

list_shap_paths = [
    "/media/icnlab/Data/Thang/plan_dieases/vit_xai/src/shap_pss_all_models_MBV.npz",
    "/media/icnlab/Data/Thang/plan_dieases/vit_xai/src/shap_pss_all_models_mobilenetv3_small.npz",
    "/media/icnlab/Data/Thang/plan_dieases/vit_xai/src/shap_pss_all_models_shufflenet.npz",
    "/media/icnlab/Data/Thang/plan_dieases/vit_xai/src/shap_pss_all_models_squezze_densenet.npz",
    "/media/icnlab/Data/Thang/plan_dieases/vit_xai/src/shap_pss_all_models_vgg_resnet.npz"
]
def load_pss__shap(paths: List[str]):
    pss = {}
    for path in paths:
        data = np.load(path)
        for item_name in data.files:
            pss[item_name] = data[item_name]
    return pss
print("PSS SHAP:")
pss_shap = load_pss__shap(list_shap_paths)
df_shap = pd.DataFrame(pss_shap)
print(df_shap)
print("="*90)
