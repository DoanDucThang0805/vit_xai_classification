from torch.utils.data import DataLoader
import time
import onnxruntime as ort
from dataset.dataset import test_dataset 


# Thiết lập device (nhưng cho ONNX, chúng ta sẽ fix dùng CPU)
device = "cpu"
print(f"Đang sử dụng thiết bị: {device}")

# DataLoader
try:
    test_ds = DataLoader(test_dataset, batch_size=1, shuffle=False)
    num_images_to_test = 100

    # Lấy sample input (để kiểm tra kích thước nếu cần)
    sample_image, _ = test_dataset[0]
    dummy_input = sample_image.unsqueeze(0).to(device)
    input_shape = dummy_input.shape
    print(f"Phát hiện kích thước ảnh đầu vào: {input_shape}")

except Exception as e:
    # Thoát nếu không có data
    exit()

def calculate_latency_onnx(model_name, onnx_model_path, test_ds, num_images_to_test):
    # Load ONNX model với CPU provider
    print(f"\nĐang tải mô hình ONNX: {model_name} từ {onnx_model_path}...")
    try:
        session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
    except Exception as e:
        print(f"Lỗi khi tải mô hình ONNX: {e}")
        return
    
    # Lấy input name (thường là input đầu tiên)
    input_name = session.get_inputs()[0].name
    
    # Warmup để tránh overhead ban đầu
    print("Đang warmup mô hình ONNX (chạy 5 lần đầu)...")
    try:
        image, _ = next(iter(test_ds))
        # Chuyển image từ torch tensor sang numpy array (ONNX yêu cầu numpy)
        image_np = image.numpy()
        for _ in range(5):
            _ = session.run(None, {input_name: image_np})
    except StopIteration:
        print("Lỗi: test_ds rỗng, không thể warmup.")
        return
    except Exception as e:
        print(f"Lỗi trong quá trình warmup: {e}")
        return
        
    # Đo latency
    total_time_seconds = 0.0
    latencies_seconds = []  # Lưu latency từng ảnh
    
    for i, (image, _) in enumerate(test_ds):
        if i >= num_images_to_test:
            break
        # Chuyển image từ torch tensor sang numpy array
        image_np = image.numpy()
        
        start_time = time.time()
        _ = session.run(None, {input_name: image_np})
        latency_sec = time.time() - start_time
        
        latencies_seconds.append(latency_sec)
        total_time_seconds += latency_sec
        
    if not latencies_seconds:
        print("Không đo được latency (có thể num_images_to_test = 0 hoặc test_ds rỗng).")
        return

    # Chuyển sang ms
    # Dùng len(latencies_seconds) để chính xác nếu dataset có ít hơn num_images_to_test
    actual_images_tested = len(latencies_seconds)
    avg_latency_ms = (total_time_seconds / actual_images_tested) * 1000
    total_latency_ms = total_time_seconds * 1000
    latencies_ms = [lat * 1000 for lat in latencies_seconds]
 
    print(f"\n--- Kết quả Latency cho mô hình ONNX {model_name} trên {actual_images_tested} ảnh (đơn vị ms, CPU) ---")
    print(f"Latency trung bình: {avg_latency_ms:.2f} ms/ảnh")
    print(f"Tổng thời gian: {total_latency_ms:.2f} ms")
    
# --- PHẦN THỰC THI CHÍNH ---
# Chọn mô hình ONNX bạn muốn kiểm tra
model_to_test_name = "mobilenetv3_small"
onnx_model_path = "./export/mobilenetv3_small.onnx"

# Gọi hàm để đo
try:
    calculate_latency_onnx(
        model_name=model_to_test_name,
        onnx_model_path=onnx_model_path,
        test_ds=test_ds,
        num_images_to_test=num_images_to_test
    )
except Exception as e:
    print(f"\nĐã xảy ra lỗi không xác định trong quá trình thực thi cho {model_to_test_name}: {e}")