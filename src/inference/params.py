"""
Model Parameter Analysis and Benchmarking.

This module analyzes and compares multiple neural network architectures used for
plant disease classification. It computes:
    - FLOPs (Floating Point Operations)
    - Model parameters count
    - Inference latency
    - Memory usage
    - Model efficiency metrics

Supports multiple architectures:
    - DenseNet121
    - ResNet50
    - MobilePlantVIT
    - VGG16
    - ShuffleNetV2
    - MobileNetV3 Small
    - SqueezeNetV2
"""

import time

import torch
from torch.utils.data import DataLoader
from thop import profile
import pandas as pd
from dataset.dataset import test_dataset

from model.densnet121 import model as DenseNet121
from model.resnet50 import model as ResNet50
from model.mobileplantvit import model as MobilePlantVit
from model.vgg16 import model as VGG16
from model.shufflenet import model as ShuffleNetv2
from model.mobilenetv3_small import model as MobileNetV3_small
from model.squezzenet import model as SqueezeNetv2


# Setup environment: Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: **{device}**")

# 1. Get input size from dataset
try:
    sample_image, _ = test_dataset[0]
    # Send dummy input to selected device (CUDA/CPU)
    dummy_input = sample_image.unsqueeze(0).to(device)
    input_shape = dummy_input.shape
    print(f"Detected input image shape: {input_shape}")
except Exception as e:
    print(f"Error getting image shape: {e}")
    # Create dummy input on selected device
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

# 2. Prepare DataLoader for latency measurement
test_ds = DataLoader(test_dataset, batch_size=1, shuffle=False)
num_images_to_test = 100

# 3. Define dictionary containing imported models
models_to_test = {
    "DenseNet121": DenseNet121,
    "ResNet50": ResNet50,
    "MobilePlantVit": MobilePlantVit,
    "VGG16": VGG16,
    "ShuffleNet": ShuffleNetv2,
    "MobileNetV3_Small": MobileNetV3_small,
    "SqueezeNet": SqueezeNetv2
}

# 4. Measurement loop
results = []

# Khởi động (warm-up) và setup Event
if device.type == 'cuda':
    print("\nĐang khởi động (warm-up) GPU và đồng bộ hóa...")
    # Warm-up GPU
    with torch.no_grad():
        for _ in range(20):
            if MobilePlantVit:
                model_warmup = MobilePlantVit.to(device).eval()
                _ = model_warmup(dummy_input)
    torch.cuda.synchronize() # Đảm bảo mọi tác vụ warm-up đã hoàn tất
    
    # Thiết lập CUDA Events để đo thời gian chính xác
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
else:
    # Nếu là CPU, vẫn dùng logic warm-up cũ
    if MobilePlantVit:
        model_warmup = MobilePlantVit.to(device).eval()
        with torch.no_grad():
            for _ in range(10): 
                _ = model_warmup(dummy_input)
    print("Warm-up hoàn tất.")


for name, model in models_to_test.items():
    if model is None:
        print(f"\n--- Bỏ qua mô hình: **{name}** (Import thất bại) ---")
        continue

    print(f"\n--- Đang xử lý mô hình: **{name}** ---")
    model = model.to(device).eval()

    # == A. Tính Params và FLOPs với 'thop' ==
    # THOP LUÔN YÊU CẦU THỰC HIỆN TRÊN CPU
    try:
        model_cpu = model.to('cpu')
        input_cpu = dummy_input.to('cpu')
        
        flops, params = profile(model_cpu, inputs=(input_cpu, ), verbose=False)
        
        # CHUYỂN MÔ HÌNH TRỞ LẠI DEVICE ĐỂ ĐO ĐỘ TRỄ (LATENCY)
        model = model_cpu.to(device)

        params_m = params / 1_000_000
        flops_g = flops / 1_000_000_000
        print(f"Params (M): {params_m:,.2f}")
        print(f"FLOPs (G): {flops_g:,.2f}")
    except Exception as e:
        print(f"Lỗi khi tính FLOPs/Params: {e}")
        params_m = -1
        flops_g = -1

    # == B. Đo Inference Latency (ms) == 
    total_time_ms = 0
    images_processed = 0
    
    with torch.no_grad():
        for images, _ in test_ds:
            if images_processed >= num_images_to_test:
                break
            
            images = images.to(device)
            
            # --- LOGIC ĐO LƯỜNG CẬP NHẬT CHO CUDA/CPU ---
            if device.type == 'cuda':
                # Dùng Event cho GPU (bất đồng bộ)
                start_event.record()
                _ = model(images)
                end_event.record()
                torch.cuda.synchronize() # Đợi GPU hoàn tất
                total_time_ms += start_event.elapsed_time(end_event) # Thời gian tính bằng ms
            else:
                # Dùng time.time() cho CPU (đồng bộ)
                start_time = time.time()
                _ = model(images)
                total_time_ms += (time.time() - start_time) * 1000
            # ---------------------------------------------
            
            images_processed += 1
            
    if images_processed == 0:
        print("Lỗi: Không xử lý được bất kỳ ảnh nào.")
        avg_latency_ms = -1
    else:
        avg_latency_ms = total_time_ms / images_processed
        # Làm tròn kết quả latency
        print(f"Avg. Latency (ms) trên {images_processed} ảnh: **{avg_latency_ms:,.2f}**")

    # Lưu kết quả
    results.append({
        "Model": name,
        "Params (M)": params_m,
        "FLOPs (G)": flops_g,
        "Latency (ms)": avg_latency_ms
    })

# 5. In kết quả cuối cùng
print("\n" + "="*50)
print("             🏆 BẢNG KẾT QUẢ TỔNG HỢP 🏆")
print("="*50)
df_results = pd.DataFrame(results)
print(df_results.to_string(index=False, formatters={
    'Params (M)': '{:,.2f}'.format,
    'FLOPs (G)': '{:,.2f}'.format,
    'Latency (ms)': '{:,.2f}'.format
}))
print("="*50)

# So sánh yêu cầu
print("\n--- YÊU CẦU SO SÁNH (Dựa trên kết quả đo lường) ---")
mobile_vit_row = df_results[df_results['Model'] == 'MobilePlantVit']
resnet_row = df_results[df_results['Model'] == 'ResNet50']
vgg16_row = df_results[df_results['Model'] == 'VGG16']

# ... (Phần so sánh này giữ nguyên) ...
if not mobile_vit_row.empty and not vgg16_row.empty:
    mobile_vit_latency = mobile_vit_row['Latency (ms)'].values[0]
    vgg16_latency = vgg16_row['Latency (ms)'].values[0]
    print(f"So sánh Latency:")
    print(f"  - MobilePlantVit: {mobile_vit_latency:,.2f} ms (Mục tiêu ~5.3 ms)")
    print(f"  - VGG16: {vgg16_latency:,.2f} ms (Mục tiêu ~18.7 ms)")
    print(f"  -> MobilePlantVit {'Nhanh hơn' if mobile_vit_latency < vgg16_latency else 'Chậm hơn'} VGG16: {(vgg16_latency / mobile_vit_latency):,.1f} lần.")

if not mobile_vit_row.empty and not resnet_row.empty:
    mvp_params = mobile_vit_row['Params (M)'].values[0]
    mvp_flops = mobile_vit_row['FLOPs (G)'].values[0]
    res_params = resnet_row['Params (M)'].values[0]
    res_flops = resnet_row['FLOPs (G)'].values[0]
    print(f"\nXác minh tính mới (MobilePlantVit vs ResNet50):")
    print(f"  - MobilePlantVit (Mục tiêu 5.6M, 1.2G): Params={mvp_params:,.2f}M, FLOPs={mvp_flops:,.2f}G")
    print(f"  - ResNet50 (Mục tiêu 25.6M, 4.1G): Params={res_params:,.2f}M, FLOPs={res_flops:,.2f}G")
    
    param_check = "Thấp hơn" if mvp_params < res_params else "Cao hơn"
    flop_check = "Thấp hơn" if mvp_flops < res_flops else "Cao hơn"
    print(f"  -> MobilePlantVit có Params: **{param_check}** ResNet50.")
    print(f"  -> MobilePlantVit có FLOPs: **{flop_check}** ResNet50.")