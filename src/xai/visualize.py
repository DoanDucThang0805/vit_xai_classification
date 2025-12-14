import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import os
from pathlib import Path
from .gradcam import * 
from .lime import *
from .shap import *

def overlay_heatmap(img_np, attr_map, colormap_name='jet', alpha=0.5, diverging=False):
    """
    Hàm helper để chồng heatmap lên ảnh (dùng cho cả tuần tự và phân kỳ).
    img_np: Ảnh gốc (H,W,C) đã denormalize [0,1].
    attr_map: Map giải thích (H,W).
    """
    if diverging:
        # Dùng cho SHAP: chuẩn hóa đối xứng quanh 0
        vmax = np.abs(attr_map).max()
        vmin = -vmax
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
    else:
        # Dùng cho GradCAM/LIME: map đã được chuẩn hóa [0,1]
        norm = plt.Normalize(vmin=0, vmax=1)
    
    cmap = plt.get_cmap(colormap_name)
    heatmap_rgb = cmap(norm(attr_map))[..., :3] # Lấy RGB, bỏ Alpha
    
    overlay = (1 - alpha) * img_np + alpha * heatmap_rgb
    return np.clip(overlay, 0, 1)


def visualize_comparison(image_tensor, gradcam_map, lime_map, shap_map, 
                         true_label, pred_label, class_names, save_path,
                         pss_gradcam=None, pss_lime=None, pss_shap=None): 
    
    img_np = denormalize(image_tensor)
    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    
    true_name = class_names[true_label]
    pred_name = class_names[pred_label]
    
    # 1. Ảnh Gốc
    axes[0].imshow(img_np)
    axes[0].set_title(f"Original\nTrue: {true_name}\nPred: {pred_name}", fontsize=10)
    
    # 2. Grad-CAM (Thêm PSS vào tiêu đề)
    gradcam_overlay = overlay_heatmap(img_np, gradcam_map, colormap_name='jet', diverging=False)
    axes[1].imshow(gradcam_overlay)
    title_gradcam = f"Grad-CAM\nPSS: {pss_gradcam:.4f}" if pss_gradcam is not None else "Grad-CAM"
    axes[1].set_title(title_gradcam)
    
    # 3. LIME (Thêm PSS vào tiêu đề)
    lime_overlay = overlay_heatmap(img_np, lime_map, colormap_name='jet', diverging=False)
    axes[2].imshow(lime_overlay)
    title_lime = f"LIME\nPSS: {pss_lime:.4f}" if pss_lime is not None else "LIME"
    axes[2].set_title(title_lime)
    
    # 4. SHAP (Thêm PSS vào tiêu đề)
    shap_overlay = overlay_heatmap(img_np, shap_map, colormap_name='seismic', diverging=True)
    im_shap = axes[3].imshow(shap_overlay)
    title_shap = f"KernelSHAP\nPSS: {pss_shap:.4f}" if pss_shap is not None else "KernelSHAP"
    axes[3].set_title(title_shap)
    
    vmax = np.abs(shap_map).max()
    vmin = -vmax
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap='seismic', norm=norm)
    sm.set_array([])

    cbar_ax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
    cbar.ax.set_title('SHAP Value', fontsize=8, pad=10)
    
    for ax in axes.flat:
        ax.axis('off')
        
    plt.tight_layout(rect=[0, 0, 0.9, 1]) 
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)