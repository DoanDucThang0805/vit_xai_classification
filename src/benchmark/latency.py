import onnxruntime as ort
import numpy as np
import time

# ----- Cấu hình -----
onnx_path = "/media/icnlab/Data/Thang/plan_dieases/vit_xai/src/export/mobilevitxxs.onnx"
num_images = 100
img_size = (3, 224, 224)

# ----- Tạo session chỉ chạy CPU -----
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 4
sess_options.inter_op_num_threads = 1
sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

session = ort.InferenceSession(
    onnx_path,
    sess_options,
    providers=["CPUExecutionProvider"],
)

print("Provider đang dùng:", session.get_providers())

# ----- Lấy tên input/output -----
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# ----- Tạo batch input (100 ảnh) -----
batch_input = np.random.randn(num_images, *img_size).astype(np.float32)

# ----- Warm-up đầy đủ (10 lần) -----
print("\n🔥 Warm-up...")
for _ in range(10):
    _ = session.run([output_name], {input_name: batch_input[:1]})


# ----- Đo BATCH (so sánh) -----
print("\n🔁 Đo batch processing (100 ảnh cùng lúc)...")
start = time.perf_counter()
outputs = session.run([output_name], {input_name: batch_input})
end = time.perf_counter()

batch_time = (end - start) * 1000
batch_avg = batch_time / num_images

print(f"  • Batch total: {batch_time:.2f} ms")
print(f"  • Batch avg: {batch_avg:.2f} ms/ảnh")
print(f"  • Output shape: {outputs[0].shape}")

# ----- Đo BATCH=1 (latency) -----
print("\n⏱️  Đo latency (batch=1)...")
start = time.perf_counter()
for _ in range(num_images):
    _ = session.run([output_name], {input_name: batch_input[:1]})
end = time.perf_counter()

latency_time = (end - start) * 1000
latency_avg = latency_time / num_images

print(f"  • Total: {latency_time:.2f} ms cho {num_images} lần")
print(f"  • Avg latency: {latency_avg:.2f} ms/ảnh")
