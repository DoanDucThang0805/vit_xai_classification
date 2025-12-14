"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Visualization.

This module implements Grad-CAM for visualizing and understanding which regions
of an input image are most important for a model's prediction. It supports 
multiple model architectures and provides utility functions for image processing.

Grad-CAM works by computing gradients of class predictions with respect to 
convolutional feature maps, providing human-interpretable visual explanations
for deep learning decisions.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks 
via Gradient-based Localization", ICCV 2017
"""

import numpy as np
from torchcam.methods import GradCAM
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import cv2

# Standard ImageNet normalization parameters
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

def denormalize(image_tensor):
    """
    Convert normalized image tensor back to original pixel value range.
    
    Reverses the standard ImageNet normalization (using mean and std) to
    transform an image from normalized form [0, 1] back to displayable form.
    
    Args:
        image_tensor (torch.Tensor): Normalized image tensor of shape (C, H, W)
        
    Returns:
        np.ndarray: Denormalized image array of shape (H, W, C) with values in [0, 1]
    """
    image = image_tensor.cpu().numpy().transpose(1, 2, 0)
    image = image * STD + MEAN
    return np.clip(image, 0, 1)

def gradcam_explain(model, image, label=None, device='cpu'):
    """
    Generate Grad-CAM visualization for model predictions.
    
    Computes the Grad-CAM activation map for a given image and model. Automatically
    selects appropriate layers for different model architectures. The activation
    map highlights regions important for the model's prediction.
    
    Supported Models:
        - ResNet: target_layer = "layer3"
        - MobileVIT/MobileVITPlant: target_layer = "stage4"
        - DenseNet: target_layer = "features.denseblock3"
        - MobileNet: target_layer = "features"
        - VGG: target_layer = "features"
        - ShuffleNet: target_layer = "conv5"
        - SqueezeNet: target_layer = "classifier.1"
        - MobileVITXXS: target_layer = "final_conv"
    
    Args:
        model (nn.Module): Neural network model (in eval mode)
        image (torch.Tensor): Input image tensor of shape (C, H, W), not batched
        label (int, optional): Target class for Grad-CAM. If None, uses model's 
                              predicted class. Defaults to None.
        device (str, optional): Device to run on ('cpu' or 'cuda'). Defaults to 'cpu'.
        
    Returns:
        np.ndarray: Grad-CAM activation map of shape (H, W), normalized to [0, 1]
    """
    mname = model.__class__.__name__.lower()
    if "resnet" in mname:
        target_layer = "layer3"
    elif "mobilevitplant" in mname:
        target_layer = "stage4"
    elif "densenet" in mname:
        target_layer = "features.denseblock3"
    elif "mobilenet" in mname:
        target_layer = "features"
    elif "vgg" in mname:
        target_layer = "features"
    elif "shufflenet" in mname:
        target_layer = "conv5"
    elif "squeezenet" in mname:
        target_layer = "classifier.1"
    elif "mobilevitxxs" in mname:
        target_layer = "final_conv"
    else:
        target_layer = None

    cam_extractor = GradCAM(model, target_layer=target_layer)

    x = image.unsqueeze(0).to(device).float()   # (1,C,H,W)
    x.requires_grad_(True)

    outputs = model(x)
    pred_class = outputs.argmax(dim=1).item()

    target_class = label if label is not None else pred_class

    cams = cam_extractor(target_class, outputs)
    cam = cams[0].squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    H, W = image.shape[1], image.shape[2]
    cam = cv2.resize(cam, (W, H))
    cam_extractor.remove_hooks()

    return cam


