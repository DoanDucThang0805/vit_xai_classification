"""
Plant Disease Classification - ONNX Model Benchmark Script

Script này dùng để benchmark hiệu suất của các mô hình ONNX trên Raspberry Pi 4B.
Nó đo lường latency, FPS, mức sử dụng RAM và CPU của từng mô hình với số threads khác nhau.
Kết quả được lưu vào CSV file để dễ phân tích.
"""

import time  # Đo thời gian thực hiện
import psutil  # Lấy thông tin CPU, RAM
import numpy as np  # Xử lý array số
import onnxruntime as ort  # Runtime để chạy mô hình ONNX
import pandas as pd  # Tạo và xử lý dataframe
import os  # Lấy PID process
import gc  # Buộc garbage collection

# ==================== Cấu hình ====================
# Thư mục chứa các mô hình ONNX
checkpoint_dir = "/home/pi/thang/plant_dieasse/checkpoints/plant_village"

# Dictionary chứa tên mô hình và đường dẫn tương ứng
# Các mô hình sẽ được test lần lượt để so sánh hiệu suất
MODEL_PATHS = {
    "VGG16": f"{checkpoint_dir}/vgg16.onnx",
    "MobileNetV3-Small": f"{checkpoint_dir}/mobilenetv3_small.onnx",
    "MobilePlantViT": f"{checkpoint_dir}/mobileplantvit.onnx",
    "DenseNet121": f"{checkpoint_dir}/densnet121.onnx",
    "ResNet50": f"{checkpoint_dir}/resnet50.onnx",
    "SqueezeNetV2": f"{checkpoint_dir}/squezzenetv2.onnx",
    "ShuffleNetV2": f"{checkpoint_dir}/shufflenetv2.onnx",
}

# Hình dạng input ảnh: (batch_size=1, channels=3, height=224, width=224)
INPUT_SHAPE = (1, 3, 224, 224)

# Số lần chạy warmup để đặt bộ nhớ cache (các kết quả này sẽ bị loại bỏ)
N_WARMUP = 15

# Số lần chạy thực tế để thu thập dữ liệu benchmark
N_RUNS = 100

# Danh sách số threads CPU để test. 1 = single-threaded, 4 = multi-threaded
THREAD_SETTINGS = [1, 4]

# Tên file CSV để lưu kết quả benchmark
OUTPUT_CSV = "pi4b_onnx_benchmark.csv"




# ==================== Hàm tạo session ONNX ====================
def create_session(model_path, threads):
    """
    Tạo một ONNX InferenceSession với cấu hình threads được chỉ định.
    
    Args:
        model_path (str): Đường dẫn đến file mô hình ONNX
        threads (int): Số threads CPU sẽ dùng cho inference
    
    Returns:
        ort.InferenceSession: Session đã cấu hình sẵn
    """
    # Tạo SessionOptions để cấu hình chi tiết
    so = ort.SessionOptions()
    
    # Số threads cho phép dùng trong mỗi operation (ví dụ: matrix multiply)
    so.intra_op_num_threads = threads
    
    # Số threads cho các operation song song (độc lập với nhau)
    so.inter_op_num_threads = 1
    
    # Mode thực hiện: ORT_SEQUENTIAL = chạy lần lượt (không song song)
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    
    # Tối ưu hóa graph: ORT_ENABLE_ALL = bật tất cả các tối ưu hóa sẵn có
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Tạo session sử dụng CPU execution provider
    return ort.InferenceSession(
        model_path,
        sess_options=so,
        providers=["CPUExecutionProvider"]  # Chỉ dùng CPU, không dùng GPU
    )




# ==================== Hàm benchmark ====================
def benchmark(model_name, model_path, threads):
    """
    Benchmark một mô hình ONNX bằng cách chạy nó nhiều lần và đo lường các metrics.
    
    Args:
        model_name (str): Tên mô hình (để display và lưu kết quả)
        model_path (str): Đường dẫn đến file .onnx
        threads (int): Số threads CPU để dùng
    
    Returns:
        dict: Dictionary chứa các metrics: Model, Threads, Latency_ms/img, FPS, Peak_RAM_MB, CPU_Usage_%
    """
    print(f"\n▶ {model_name} | Threads={threads}")

    # Tạo session với cấu hình threads đã chỉ định
    session = create_session(model_path, threads)
    
    # Lấy tên input layer (layer đầu tiên của mô hình)
    input_name = session.get_inputs()[0].name
    
    # Tạo dummy input data: random array với shape (1, 3, 224, 224), kiểu float32
    dummy = np.random.rand(*INPUT_SHAPE).astype(np.float32)

    # ===== WARMUP PHASE =====
    # Chạy mô hình N_WARMUP lần để:
    # - Khởi động lại cache CPU
    # - Cho ONNX runtime tối ưu hóa kernel
    for _ in range(N_WARMUP):
        session.run(None, {input_name: dummy})

    # Lấy process hiện tại để monitor RAM
    process = psutil.Process(os.getpid())

    # Danh sách lưu latency (thời gian chạy) của từng lần inference
    latencies = []
    
    # Danh sách lưu mức sử dụng CPU của từng lần inference
    cpu_list = []
    
    # Biến lưu mức RAM cao nhất được dùng
    peak_ram = 0

    # ===== BENCHMARK PHASE =====
    for _ in range(N_RUNS):
        # Buộc garbage collection để xóa bộ nhớ không cần thiết
        gc.collect()

        # Đo mức sử dụng CPU trước inference
        cpu_before = psutil.cpu_percent(interval=None)

        # Đo thời gian bắt đầu (high-resolution timer)
        start = time.perf_counter()
        
        # Chạy model với dummy input
        session.run(None, {input_name: dummy})
        
        # Đo thời gian kết thúc
        end = time.perf_counter()

        # Đo mức sử dụng CPU sau inference
        cpu_after = psutil.cpu_percent(interval=None)

        # Tính latency (thời gian chạy) theo millisecond
        latencies.append((end - start) * 1000)
        
        # Lưu trung bình CPU trước và sau
        cpu_list.append((cpu_before + cpu_after) / 2)

        # Lấy mức RAM hiện tại (convert từ bytes sang MB)
        ram = process.memory_info().rss / 1024 / 1024
        
        # Cập nhật peak RAM (giữ lại giá trị cao nhất)
        peak_ram = max(peak_ram, ram)

    # ===== TÍNH TOÁN METRICS =====
    # Latency trung bình (millisecond)
    latency = np.mean(latencies)
    
    # FPS = 1000ms / latency_ms (số ảnh xử lý được trong 1 giây)
    fps = 1000 / latency

    # Trả về dictionary kết quả
    return {
        "Model": model_name,
        "Threads": threads,
        "Latency_ms/img": round(latency, 2),  # Thời gian (ms)
        "FPS": round(fps, 2),  # Frames per second
        "Peak_RAM_MB": round(peak_ram, 1),  # Mức RAM cao nhất
        "CPU_Usage_%": round(np.mean(cpu_list), 1),  # Trung bình CPU
    }


# ==================== MAIN EXECUTION ====================
# Danh sách lưu kết quả benchmark của tất cả các mô hình và thread settings
results = []

# Loop qua từng mô hình
for model, path in MODEL_PATHS.items():
    # Loop qua từng thread setting
    for t in THREAD_SETTINGS:
        # Chạy benchmark và lưu kết quả vào results
        results.append(benchmark(model, path, t))

# Chuyển danh sách results thành Pandas DataFrame (bảng dữ liệu)
df = pd.DataFrame(results)

# Lưu DataFrame vào CSV file (index=False: không lưu row index)
df.to_csv(OUTPUT_CSV, index=False)

# Hiển thị thông báo hoàn thành và kết quả
print("\n✅ DONE")  # Hoàn thành
print(df)  # Hiển thị bảng kết quả
print(f"\n📁 Saved to {OUTPUT_CSV}")  # Thông báo file đã lưu
