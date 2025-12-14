from pathlib import Path
import torch
from model.mobilenetv3_small import model

model_name = "mobilevitxxs"
model_run_id = "run_20251111-110041"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_path = (
    Path(__file__).resolve().parents[2] /
    "checkpoints" / model_name / model_run_id / "best_checkpoint.pth"
)
if not checkpoint_path.exists():
    raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint_path}")

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device)
state_dict = (
    checkpoint.get("model_state_dict")
    or checkpoint.get("state_dict")
    or checkpoint
)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

# Đường dẫn lưu file ONNX (ngay thư mục hiện tại)
onnx_output_path = Path(__file__).resolve().parent / f"{model_name}.onnx"

# Tạo input giả (kích thước phải trùng kích thước huấn luyện)
dummy_input = torch.randn(1, 3, 224, 224).to(device)

# Xuất mô hình sang ONNX
torch.onnx.export(
    model,
    dummy_input,
    onnx_output_path,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)

print(f"✅ Đã xuất mô hình sang ONNX: {onnx_output_path}")
