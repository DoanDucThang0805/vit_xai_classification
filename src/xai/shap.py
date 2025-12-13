"""
KernelSHAP (Kernel Shapley Additive exPlanations) Visualization.

This module implements KernelSHAP for generating Shapley value-based explanations.
SHAP values provide a game-theoretic approach to interpreting model predictions,
assigning each feature an importance value based on cooperative game theory.

KernelSHAP approximates true SHAP values by:
1. Generating perturbed samples around the instance
2. Computing model predictions on perturbed inputs
3. Fitting a weighted linear model using Shapley kernel weights
4. Extracting feature importance scores

Reference: Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions", 
NIPS 2017
"""

import torch
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
from captum.attr import KernelShap
from skimage.segmentation import slic


MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

def denormalize(image_tensor):
    """
    Convert normalized image tensor back to original pixel value range.
    
    Reverses ImageNet normalization for display purposes.
    
    Args:
        image_tensor (torch.Tensor): Normalized image tensor of shape (C, H, W)
        
    Returns:
        np.ndarray: Denormalized image array of shape (H, W, C) with values in [0, 1]
    """
    image = image_tensor.cpu().numpy().transpose(1, 2, 0)
    image = image * STD + MEAN
    return np.clip(image, 0, 1)

def segment_fn(image_tensor, target_device):
    """
    Segment image into superpixels using SLIC algorithm.
    
    SLIC (Simple Linear Iterative Clustering) segments the image into meaningful
    regions (superpixels) which serve as interpretable units for KernelSHAP.
    
    Args:
        image_tensor (torch.Tensor): Image tensor of shape (C, H, W)
        target_device (torch.device): Device to move segments to
        
    Returns:
        torch.Tensor: Segment mask of shape (H, W) indicating superpixel assignments
    """
    # Process image (CPU)
    image_np = image_tensor.cpu().numpy().transpose(1, 2, 0)
    image_np = image_np * STD + MEAN
    image_np = np.clip(image_np, 0, 1)
    segments = slic(image_np, n_segments=50, compactness=10, start_label=0) 
    
    return torch.from_numpy(segments).to(target_device)

def shap_explain(model, image, label, device='cpu'):
    """
    Generate KernelSHAP explanation for model predictions.
    
    Computes SHAP values using kernel-weighted regression to approximate
    the true Shapley values. Higher values indicate features that increase
    the model's prediction for the target class.
    
    Args:
        model (nn.Module): Neural network model (in eval mode)
        image (torch.Tensor): Input image tensor of shape (C, H, W)
        label (int): Target class for explanation
        device (str, optional): Device to run on ('cpu' or 'cuda'). Defaults to 'cpu'.
        
    Returns:
        np.ndarray: SHAP attribution map of shape (H, W) with raw SHAP values
    """
    model.eval()
    
    #Chuyển ảnh đầu vào và baselines lên GPU (device)
    x = image.unsqueeze(0).to(device)
    baselines = torch.zeros_like(x).to(device) # baselines cũng phải ở GPU

    k_shap = KernelShap(model) 

    attr = k_shap.attribute(
        x,
        target=label,
        baselines=baselines,
        n_samples=100, 
        perturbations_per_eval=16, 
        feature_mask=segment_fn(image, device)
    )

    # Xử lý đầu ra
    attr = attr.squeeze().detach().cpu().numpy()
    attr = np.transpose(attr, (1, 2, 0))
    attr = np.mean(attr, axis=-1) 
    
    return attr