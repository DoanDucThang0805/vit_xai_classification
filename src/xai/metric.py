import torch
import torch.nn.functional as F
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import numpy as np


def ssim_metric(map1, map2):
    """
    Tính Structural Similarity Index (SSIM) giữa hai bản đồ (tensors).
    map1, map2: torch.Tensor (H, W) hoặc (1, 1, H, W)
    """
    # 1. Thêm chiều Batch/Channel nếu cần
    if map1.dim() == 2:
        map1 = map1.unsqueeze(0).unsqueeze(0)
    if map2.dim() == 2:
        map2 = map2.unsqueeze(0).unsqueeze(0)
        
    # 2. Tính data_range và xử lý lỗi chia cho 0
    # Lấy data_range từ bản đồ đầu tiên
    data_range = map1.max() - map1.min()
    
    # 💥 BƯỚC SỬA LỖI NAN: Xử lý data_range = 0
    # Dùng ngưỡng nhỏ (1e-6) để xử lý lỗi làm tròn số thực
    if data_range.item() < 1e-6:
        # Nếu bản đồ đồng nhất (data_range ≈ 0), coi như ổn định tuyệt đối
        return torch.as_tensor(1.0) 
            
    # 3. Tính SSIM
    # SSIM cần tensor Float
    return ssim(map1.float(), map2.float(), data_range=data_range, size_average=True)


def calculate_pss(list_of_attribution_maps):
    """
    Tính Perturbation Stability Score (PSS) từ K bản đồ.
    PSS = 1 / (K(K-1)) * SUM_{k != l} SSIM(S^(k), S^(l))
    list_of_attribution_maps: List chứa K bản đồ (numpy array hoặc torch tensor (H, W)).
    """
    K = len(list_of_attribution_maps)
    if K < 2:
        return 0.0 # Không thể tính nếu K < 2

    total_ssim = 0.0
    count = 0
    
    # Chuyển tất cả sang tensor để tính toán SSIM
    maps_tensor = [torch.as_tensor(m).squeeze() for m in list_of_attribution_maps]

    # Tính SSIM giữa tất cả các cặp khác nhau (k != l)
    for k in range(K):
        for l in range(K):
            if k != l:
                # Tính SSIM
                ssim_val = ssim_metric(maps_tensor[k], maps_tensor[l])
                total_ssim += ssim_val.item()
                count += 1

    # Công thức PSS
    pss = total_ssim / count if count > 0 else 0.0
    return pss