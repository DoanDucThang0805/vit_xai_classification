import os
from typing import List
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


class GradCam:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _preprocess_from_numpy(self, image_np):
        """
        image_np: (H, W, 3), range [0,1]
        """
        image = Image.fromarray((image_np * 255).astype(np.uint8))
        image = image.resize((224, 224))
        input_tensor = self.transform(image).unsqueeze(0)
        return input_tensor.to(self.device), image

    def __call__(
        self,
        model: nn.Module,
        target_layers: List[nn.Module],
        image_np: np.ndarray
    ):
        """
        return: grayscale CAM (224,224)
        """
        model.to(self.device)
        model.eval()

        input_tensor = self._preprocess_from_numpy(image_np)[0]

        cam = GradCAM(
            model=model,
            target_layers=target_layers
        )

        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_class = probs.argmax(dim=1).item()

        targets = [ClassifierOutputTarget(pred_class)]
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets
        )
        input_image = np.float32(self._preprocess_from_numpy(image_np)[1]) / 255.0
        grayscale_cam_map = grayscale_cam[0, :]
        gradcam_image_np = show_cam_on_image(input_image, grayscale_cam_map, use_rgb=True)
        return gradcam_image_np
    

class PSS_GradCam:
    def __init__(
            self, 
            root_dir: str,
            model: nn.Module,
            target_layer: List[nn.Module],
            sigma: float
        ):
        self.root_dir = root_dir
        self.target_layer = target_layer
        self.model = model
        self.sigma = sigma

    def _get_tomato_image_paths(self, root_dir: str, num_images_per_class: int=5):
        class_names = sorted(
            [d for d in os.listdir(root_dir)
             if os.path.isdir(os.path.join(root_dir, d))
             and d.startswith("Tomato")]
        )
        image_paths = []
        for class_name in class_names:
            class_dir = os.path.join(root_dir, class_name)
            class_image_names = sorted(os.listdir(class_dir))[:num_images_per_class]
            for img_name in class_image_names:
                img_path =  os.path.join(class_dir, img_name)
                image_paths.append(img_path)
        return image_paths
    
    def _load_image_np(self, image_path: str):
        img = Image.open(image_path).convert("RGB")
        img = img.resize((224, 224))
        return np.array(img).astype(np.float32) / 255.0

    def _generate_noisy_images(self, img: np.ndarray, num_noise: int=10, sigma: float=0.02):
        noisy_images = []
        for _ in range(num_noise):
            noise = np.random.normal(0, sigma, img.shape)
            noisy = img + noise
            noisy = np.clip(noisy, 0, 1)
            noisy_images.append(noisy)
        return noisy_images

    def _compute_ssim(self, img1: np.ndarray, img2: np.ndarray):
        """Compute Structural Similarity Index (SSIM) between two images.

        Args:
            img1 (np.ndarray): First image.
            img2 (np.ndarray): Second image.

        Returns:
            float: SSIM value between the two images.
        """
        ssim_score = ssim(
            img1, img2,
            data_range=255,
            channel_axis=-1
        )
        return ssim_score

    def __call__(self):
        # Step 1: Load 50 image_paths
        image_paths = self._get_tomato_image_paths(self.root_dir)
        gradcam_explainer = GradCam()
        pss_all = []
        # Step 2: Loop for Each image_path
        for image_path in tqdm(image_paths, desc="Processing images"):
            # Step 3: Load image
            image_np = self._load_image_np(image_path)
            # Step 4: Add Gauusian Noise into image => 10 version image noise
            noisy_images = self._generate_noisy_images(image_np, num_noise=10, sigma=self.sigma)
            # Step 5: Explaination for noisy images
            saliency_maps = []
            for noisy_img in noisy_images:
                img_explained = gradcam_explainer(self.model, self.target_layer, noisy_img)
                saliency_maps.append(img_explained)
            # Step 6: pairwise SSIM
            K = len(saliency_maps)
            ssim_scores = []
            for i in range(K):
                for j in range(K):
                    if i != j:
                        score = self._compute_ssim(
                            saliency_maps[i],
                            saliency_maps[j]
                        )
                        ssim_scores.append(score)
            # Step 7: PSS for this Image
            pss_image = np.mean(ssim_scores)
            pss_all.append(pss_image)
        # Step 8: PSS ALL Average for 50 Images
        return np.mean(pss_all)


if __name__ == "__main__":
    import torch
    import numpy as np

    from model.vgg16 import model as vgg16
    from model.resnet50 import model as resnet50
    from model.mobilenetv3_small import model as mobilenetv3_small
    from model.mobileplantvit import model as mobileplantvit
    from model.shufflenet import model as shufflenetv2
    from model.squezzenet import model as squeezenet
    from model.densnet121 import model as densenet121

    models = {
        "VGG16": {
            "model": vgg16,
            "ckpt": "/media/icnlab/Data/Thang/plan_dieases/vit_xai/checkpoints/plantvillage/vgg16/run_20251019-171608/best_checkpoint.pth",
            "target_layer": [vgg16.features[26]]
        },
        "ResNet50": {
            "model": resnet50,
            "ckpt": "/media/icnlab/Data/Thang/plan_dieases/vit_xai/checkpoints/plantvillage/resnet50/run_20251019-084733/best_checkpoint.pth",
            "target_layer": [resnet50.layer4[-1]]
        },
        "MobileNetV3_Small": {
            "model": mobilenetv3_small,
            "ckpt": "/media/icnlab/Data/Thang/plan_dieases/vit_xai/checkpoints/plantvillage/mobilenetv3_small/run_20251021-151012/best_checkpoint.pth",
            "target_layer": [mobilenetv3_small.blocks[-1]]
        },
        "Mobileplantvit": {
            "model": mobileplantvit,
            "ckpt": "./media/icnlab/Data/Thang/plan_dieases/vit_xai/checkpoints/plantvillage/mobileplantvit/run_20260101-103938/best_checkpoint.pth",
            "target_layer": [mobileplantvit.block4[-1]]
        },
        "ShuffleNetV2": {
            "model": shufflenetv2,
            "ckpt": "/media/icnlab/Data/Thang/plan_dieases/vit_xai/checkpoints/plantvillage/shufflenetv2/run_20251022-132921/best_checkpoint.pth",
            "target_layer": [shufflenetv2.conv5]
        },
        "SqueezeNet": {
            "model": squeezenet,
            "ckpt": "/media/icnlab/Data/Thang/plan_dieases/vit_xai/checkpoints/plantvillage/squezzenetv2/run_20251021-171131/best_checkpoint.pth",
            "target_layer": [squeezenet.features[-1]]
        },
        "DenseNet121": {
            "model": densenet121,
            "ckpt": "/media/icnlab/Data/Thang/plan_dieases/vit_xai/checkpoints/plantvillage/densnet121/run_20251018-193243/best_checkpoint.pth",
            "target_layer": [densenet121.features.norm5]
        },
    }
    sigmas = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04,
          0.05, 0.06, 0.07, 0.08, 0.09, 0.1]

    pv_root = "/media/icnlab/Data/Thang/plan_dieases/vit_xai/data/PlantVillage"
    device = "cuda"
    all_pss_scores = {}

    for model_name, cfg in models.items():
        print(f"\n=== Evaluating PSS for {model_name} ===")

        model = cfg["model"]
        checkpoint = torch.load(cfg["ckpt"], map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        target_layers = cfg["target_layer"]

        model_pss = []

        for sigma in sigmas:
            pss_gradcam = PSS_GradCam(
                model=model,
                target_layer=target_layers,
                root_dir=pv_root,
                sigma=sigma
            )

            pss_score = pss_gradcam()
            model_pss.append(pss_score)

            print(f"[{model_name}] σ={sigma:.3f} → PSS={pss_score:.4f}")

        all_pss_scores[model_name] = np.array(model_pss)
    np.savez(
        "gradcam_pss_all_models.npz",
        sigmas=np.array(sigmas),
        **all_pss_scores
    )
