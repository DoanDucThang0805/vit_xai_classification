import numpy as np


data = np.load("/media/icnlab/Data/Thang/plan_dieases/vit_xai/src/gradcam_pss_all_models.npz")
print(data["sigmas"])
import matplotlib.pyplot as plt

sigmas = data["sigmas"]

for model_name in data.files:
    if model_name == "sigmas":
        continue
    plt.plot(sigmas, data[model_name], label=model_name)

plt.xlabel("Sigma")
plt.ylabel("PSS (SHAP)")
plt.legend()
plt.grid(True)
plt.show()
