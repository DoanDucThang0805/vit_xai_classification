"""
LIME (Local Interpretable Model-agnostic Explanations) Visualization.

This module implements LIME for generating local explanations of model predictions.
LIME works by perturbing input samples and fitting a local linear model to explain
predictions in the vicinity of a particular instance.

LIME creates human-interpretable visualizations by:
1. Generating perturbed versions of the input image
2. Computing model predictions on perturbed images
3. Fitting a linear model to approximate the decision boundary
4. Highlighting important features for the prediction

Reference: Ribeiro et al., "Why Should I Trust You?" SIGMOD 2016
"""

import torch
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2  # OpenCV library for color mapping
from captum.attr import Lime
from captum._utils.models.linear_model import SkLearnLasso
from captum.attr._core.lime import get_exp_kernel_similarity_function
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
    regions (superpixels) which serve as interpretable units for LIME.
    
    Args:
        image_tensor (torch.Tensor): Image tensor of shape (C, H, W)
        target_device (torch.device): Device to move segments to
        
    Returns:
        torch.Tensor: Segment mask of shape (H, W) indicating superpixel assignments
    """
    # SLIC algorithm: 50 superpixels
    image_np = image_tensor.cpu().numpy().transpose(1, 2, 0)
    image_np = image_np * STD + MEAN
    image_np = np.clip(image_np, 0, 1)
    
    segments = slic(image_np, n_segments=50, compactness=10, start_label=0) 
    
    return torch.from_numpy(segments).to(target_device)


def lime_explain(model, image, label, device='cpu'):
    """
    Generate LIME explanation for model predictions.
    
    Creates a local linear approximation of the model's decision boundary
    by perturbing the input image and fitting a linear model.
    
    Args:
        model (nn.Module): Neural network model (in eval mode)
        image (torch.Tensor): Input image tensor of shape (C, H, W)
        label (int): Target class for explanation
        device (str, optional): Device to run on ('cpu' or 'cuda'). Defaults to 'cpu'.
        
    Returns:
        np.ndarray: LIME attribution map of shape (H, W), normalized to [0, 1]
    """
    model.eval()
    
    x = image.unsqueeze(0).to(device)
    baselines = torch.zeros_like(x).to(device)
    similarity_func = get_exp_kernel_similarity_function('euclidean', kernel_width=1000)
    lime = Lime(
        model, # Model đang ở GPU (device)
        interpretable_model=SkLearnLasso(alpha=0.05, fit_intercept=True), 
        similarity_func=similarity_func
    )

    attr = lime.attribute(
        x, # x đã ở GPU (device)
        target=label,
        baselines=baselines,
        n_samples=1000, 
        perturbations_per_eval=16, 
        feature_mask=segment_fn(image, device) # 👈 feature_mask sẽ ở GPU
    )

    attr = attr.squeeze().detach().cpu().numpy()
    attr = np.transpose(attr, (1, 2, 0))
    attr = np.mean(attr, axis=-1)
    
    # Chuẩn hóa về [0, 1]
    attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)
    
    return attr