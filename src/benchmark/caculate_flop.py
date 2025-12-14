import torch
from torch.utils.data import DataLoader
from pathlib import Path
from thop import profile
import os

def measure_model(model, device="cpu", input_shape=(1, 3, 224, 224)):
    """
    Đo số lượng tham số (Params) và FLOPs.

    Args:
        model (torch.nn.Module): Mô hình PyTorch đã load checkpoint.
        device (str): 'cpu' hoặc 'cuda'.
        input_shape (tuple): Kích thước đầu vào (B, C, H, W).

    Returns:
        dict: {"Params (M)", "FLOPs (G)"}
    """
    model = model.to(device).eval()
    dummy_input = torch.randn(input_shape).to(device)

    # Tính Params & FLOPs
    try:
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        params_m = params / 1_000_000
        flops_g = flops / 1_000_000_000
    except Exception as e:
        print(f"❌ Lỗi khi tính FLOPs/Params: {e}")
        params_m, flops_g = -1, -1

    # Hiển thị kết quả (không in kiến trúc mô hình)
    print("\n📊 Kết quả đo mô hình:")
    print(f"   • Params: {params_m:.2f} M")
    print(f"   • FLOPs: {flops_g:.2f} G")

    return {
        "Params (M)": round(params_m, 2),
        "FLOPs (G)": round(flops_g, 2)
    }

# --- Phần load và test model ---


from model.mobilevitxxs import model
model_name = "mobilenetv3_small"
model_run_id = "run_20251021-151012"

num_class = 10  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# checkpoint_path = Path(__file__).resolve().parents[2] / 'checkpoints' / model_name / model_run_id / 'best_checkpoint.pth'
# if not checkpoint_path.exists():
#     raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
# checkpoint = torch.load(checkpoint_path, map_location=device)
# if "model_state_dict" in checkpoint:
#     state_dict = checkpoint["model_state_dict"]
# elif "state_dict" in checkpoint:
#     state_dict = checkpoint["state_dict"]
# else:
#     state_dict = checkpoint
# model.load_state_dict(state_dict)
# model = model.to(device)


results = measure_model(model, device=device, input_shape=(1, 3, 224, 224))