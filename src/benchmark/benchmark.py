import onnxruntime as ort
import numpy as np
import time


onnx_path = "/media/icnlab/Data/Thang/plan_dieases/vit_xai/src/export/mobilevitxxs.onnx"
num_images = 100
img_size = (3, 224, 224)

batch_input = np.random.randn(num_images, *img_size).astype(np.float32)
# Tạo một input đơn lẻ để test latency (batch=1)
single_image_input = batch_input[:1] 
print(f"Model: {onnx_path}")
print(f"Test data: {num_images} lần trên 1 ảnh, shape {single_image_input.shape}")


# PHẦN 1: CẤU HÌNH GỐC (4 THREADS)
print("\n--- 4 THREADS (Baseline) ---")
sess_options_4core = ort.SessionOptions()
sess_options_4core.intra_op_num_threads = 4
sess_options_4core.inter_op_num_threads = 1
sess_options_4core.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

session_4core = ort.InferenceSession(
    onnx_path,
    sess_options_4core,
    providers=["CPUExecutionProvider"],
)

print("Provider đang dùng:", session_4core.get_providers())
input_name = session_4core.get_inputs()[0].name
output_name = session_4core.get_outputs()[0].name

# ----- Warm-up (4 threads) -----
for _ in range(10):
    _ = session_4core.run([output_name], {input_name: single_image_input})

# ----- Đo BATCH=1 (latency, 4 threads) -----
print("Đo latency (batch=1, 100 lần)...")
start_lat_4core = time.perf_counter()
for _ in range(num_images):
    # Chạy 100 lần trên CÙNG MỘT ẢNH (theo yêu cầu)
    _ = session_4core.run([output_name], {input_name: single_image_input})
end_lat_4core = time.perf_counter()

latency_time_4core = (end_lat_4core - start_lat_4core) * 1000
latency_avg_4core = latency_time_4core / num_images
print(f"   • Total (4 threads): {latency_time_4core:.2f} ms cho {num_images} lần")
print(f"   • Avg latency (4 threads): {latency_avg_4core:.2f} ms/ảnh")


# PHẦN 2: GIỚI HẠN TÀI NGUYÊN (1 THREAD)
print("\n--- 1 THREAD (Giới hạn tài nguyên) ---")
sess_options_1core = ort.SessionOptions()
sess_options_1core.intra_op_num_threads = 1 # Giới hạn tài nguyên
sess_options_1core.inter_op_num_threads = 1
sess_options_1core.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

session_1core = ort.InferenceSession(
    onnx_path,
    sess_options_1core,
    providers=["CPUExecutionProvider"],
)

print("Provider đang dùng:", session_1core.get_providers())
input_name_1c = session_1core.get_inputs()[0].name
output_name_1c = session_1core.get_outputs()[0].name

# ----- Warm-up (1 thread) -----
for _ in range(10):
    _ = session_1core.run([output_name_1c], {input_name_1c: single_image_input})

# ----- Đo BATCH=1 (latency, 1 thread) -----
print("Đo latency (batch=1, 100 lần)...")
start_lat_1core = time.perf_counter()
for _ in range(num_images):
    # Chạy 100 lần trên CÙNG MỘT ẢNH
    _ = session_1core.run([output_name_1c], {input_name_1c: single_image_input})
end_lat_1core = time.perf_counter()

latency_time_1core = (end_lat_1core - start_lat_1core) * 1000
latency_avg_1core = latency_time_1core / num_images
print(f"   • Total (1 thread): {latency_time_1core:.2f} ms cho {num_images} lần")
print(f"   • Avg latency (1 thread): {latency_avg_1core:.2f} ms/ảnh")
