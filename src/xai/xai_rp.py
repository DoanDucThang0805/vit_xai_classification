import torch
import random
from pathlib import Path
import numpy as np

from dataset.dataset import test_dataset
from .metric import *
from .visualize import *

# Hàm thêm nhiễu Gaussian (cần thiết cho PSS)
def add_gaussian_noise(image_tensor, sigma=0.01):
    """Thêm nhiễu Gaussian vào tensor ảnh (C, H, W)."""
    noise = torch.randn_like(image_tensor) * sigma
    noisy_image = image_tensor.float() + noise
    return torch.clamp(noisy_image, 0, 1)

from model.mobileplantvit import model
model_name = "mobilevitxxs"
num_class = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
project_root = Path(__file__).resolve().parents[2]
checkpoint_path = project_root / 'checkpoints' / 'plantvillage' / 'mobileplantvit' / 'run_20260101-103938' / 'best_checkpoint.pth'
print(f"Đang tải checkpoint từ: {checkpoint_path}")
if not checkpoint_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=device)
state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()
print("Tải model thành công.")

target_names = [test_dataset.idx_to_class[i] for i in range(len(test_dataset.idx_to_class))]

# ===============================================
# == THIẾT LẬP PSS VÀ OUTPUT ==
# ===============================================
K_PSS = 10 # Số lần lặp để tính PSS
SIGMA_PSS = 0.01
pss_results = {'gradcam': [], 'lime': [], 'shap': []}

output_path_str = "/media/icnlab/Data/Thang/plan_dieases/vit_xai/reports/xai/mobilevitxxs"
output_dir = Path(output_path_str)
output_dir.mkdir(parents=True, exist_ok=True)
print(f"Kết quả sẽ được lưu vào thư mục: {output_dir.resolve()}")

# --- Chọn 50 ảnh ngẫu nhiên ---
num_samples_to_run = 50
dataset_size = len(test_dataset)

if dataset_size < num_samples_to_run:
    print(f"Cảnh báo: Dataset chỉ có {dataset_size} ảnh. Sẽ chạy trên toàn bộ dataset.")
    num_samples_to_run = dataset_size
    random_indices = list(range(dataset_size))
else:
    random_indices = random.sample(range(dataset_size), num_samples_to_run)

print("-" * 50)
print(f"Bắt đầu chạy so sánh và tính PSS cho {num_samples_to_run} ảnh ngẫu nhiên...")
print("-" * 50)

for i, idx in enumerate(random_indices):
    print(f"\nĐang xử lý ảnh {i+1}/{num_samples_to_run} (Dataset Index: {idx})...")
    
    image, true_label = test_dataset[idx] # image (CPU tensor)
    
    # Danh sách để lưu trữ K=10 bản đồ cho PSS
    gradcam_maps, lime_maps, shap_maps = [], [], []

    # 💥 VÒNG LẶP PSS (K=10)
    for k in range(K_PSS):
        # 1. Thêm nhiễu và chuyển ảnh nhiễu lên GPU/CPU tương ứng
        noisy_image = add_gaussian_noise(image, sigma=SIGMA_PSS)
        image_gpu_noisy = noisy_image.to(device)
        
        # 2. Chạy XAI trên ảnh nhiễu
        # Grad-CAM (dùng ảnh nhiễu GPU)
        gradcam_maps.append(gradcam_explain(model, image_gpu_noisy, label=true_label, device=device))
        
        # LIME/SHAP (dùng ảnh nhiễu CPU - sẽ được chuyển lên GPU bên trong hàm)
        lime_maps.append(lime_explain(model, noisy_image, label=true_label, device=device))
        shap_maps.append(shap_explain(model, noisy_image, label=true_label, device=device))
    

    with torch.no_grad():
        outputs = model(image.to(device).unsqueeze(0))
        pred_label = outputs.argmax(dim=1).item()
    
    true_name = target_names[true_label]
    pred_name = target_names[pred_label]
    print(f"  True Label: {true_name} | Predicted Label: {pred_name}")
    
    # 3. Tính PSS
    pss_gradcam = calculate_pss(gradcam_maps)
    pss_lime = calculate_pss(lime_maps)
    pss_shap = calculate_pss(shap_maps)
    
    # 4. Lưu PSS của ảnh hiện tại
    pss_results['gradcam'].append(pss_gradcam)
    pss_results['lime'].append(pss_lime)
    pss_results['shap'].append(pss_shap)
    
    print(f"  PSS - GradCAM: {pss_gradcam:.4f} | LIME: {pss_lime:.4f} | SHAP: {pss_shap:.4f}")
    
    # 5. Visualize (dùng bản đồ đầu tiên và ảnh gốc)
    gradcam_map_vis = gradcam_maps[0]
    lime_map_vis = lime_maps[0]
    shap_map_vis = shap_maps[0]
    
    true_name_safe = true_name.replace(" ", "_").replace("/", "-")
    save_filename = f"{model_name}_compare_idx_{idx}_{true_name_safe}.png"
    save_path = output_dir / save_filename
    
    # 💥 TRUYỀN PSS VÀO VISUALIZE:
    visualize_comparison(
        image, 
        gradcam_map_vis, 
        lime_map_vis, 
        shap_map_vis,
        true_label, 
        pred_label, 
        target_names,
        save_path,
        pss_gradcam=pss_gradcam,
        pss_lime=pss_lime,
        pss_shap=pss_shap 
    )
    print(f"  ✅ Đã lưu kết quả vào: {save_path}")

# ===============================================
# == TỔNG KẾT PSS CUỐI CÙNG ==
# ===============================================
print("-" * 50)
print(f"✅ Hoàn tất! Đã lưu {num_samples_to_run} ảnh so sánh vào thư mục '{output_dir.name}'.")
print("-" * 50)
print("📈 KẾT QUẢ ĐÁNH GIÁ ĐỊNH LƯỢNG (PSS Trung bình):")
print(f"   PSS Trung Bình (Grad-CAM): {np.mean(pss_results['gradcam']):.4f}")
print(f"   PSS Trung Bình (LIME): {np.mean(pss_results['lime']):.4f}")
print(f"   PSS Trung Bình (SHAP): {np.mean(pss_results['shap']):.4f}")
print("-" * 50)