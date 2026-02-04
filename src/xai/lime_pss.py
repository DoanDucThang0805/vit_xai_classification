import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt
from lime import lime_image
from skimage.metrics import structural_similarity as ssim
from skimage.segmentation import mark_boundaries
from torchvision import transforms


class Lime:
    def __init__(self, model: nn.Module):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.process = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.model.to(self.device)
        self.model.eval()

    def _batch_predict(self, images):
        batch = torch.stack([self.process(Image.fromarray(np.uint8(img))) for img in images])
        batch = batch.to(self.device)
        with torch.no_grad():
            logits = self.model(batch)
            probs = torch.nn.functional.softmax(logits, dim=1)
        return probs.cpu().numpy()
    
    def __call__(
        self,
        image_np: np.ndarray,
        image_show: bool = False,
    ):      
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            image_np,
            self._batch_predict,
            top_labels=5,
            hide_color=0,
            num_samples=100
        )
        label_idx = explanation.top_labels[0]
        print("Predicted class index:", label_idx)
        temp, mask = explanation.get_image_and_mask(
            label_idx,
            positive_only=False,
            num_features=3,
            hide_rest=False
        )
        lime_image_np = mark_boundaries(temp / 255.0, mask)
        lime_image_np = (lime_image_np * 255).astype(np.uint8)
        if image_show:
            plt.imshow(lime_image_np)
            plt.axis('off')
            plt.show()
        return lime_image_np


class PSS_Lime:
    def __init__(
        self,
        model: nn.Module,
        root_dir: str,
        sigma: float
    ):
        self.model = model
        self.root_dir = root_dir
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

    def _generate_noisy_images(self, img: np.ndarray, num_noise: int=5, sigma: float=0.02):
        noisy_images = []
        for _ in range(num_noise):
            noise = np.random.normal(0, sigma, img.shape)
            noisy = img + noise
            noisy = np.clip(noisy, 0, 1)
            noisy_images.append(noisy)
        return noisy_images
    
    def _convert_to_uint8(self, img: np.ndarray):
        img_uint8 = (img * 255).astype(np.uint8)
        return img_uint8

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
        lime_explainer = Lime(self.model)
        image_paths = self._get_tomato_image_paths(self.root_dir)
        pss_all = []
        for image_path in tqdm(image_paths, desc="Processing Images"):
            image_np = self._load_image_np(image_path)
            noisy_images = self._generate_noisy_images(image_np, num_noise=5, sigma=self.sigma)
            saliency_maps = []
            for noisy_img in noisy_images:
                noisy_img_uint8 = self._convert_to_uint8(noisy_img)
                image_explained = lime_explainer(
                    noisy_img_uint8,
                    image_show=False
                )
                saliency_maps.append(image_explained)
            ssim_values = []
            K = len(saliency_maps)
            print("Computing SSIM for", K, "saliency maps")
            count = 0
            for i in range(K):
                for j in range(K):
                    if i != j:
                        count = count + 1
                        ssim_score = self._compute_ssim(
                            saliency_maps[i],
                            saliency_maps[j]
                        )
                        ssim_values.append(ssim_score)
            pss_image = np.mean(ssim_values)
            pss_all.append(pss_image)
        print(count)
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
        },
        # "ResNet50": {
        #     "model": resnet50,
        #     "ckpt": "/media/icnlab/Data/Thang/plan_dieases/vit_xai/checkpoints/plantvillage/resnet50/run_20251019-084733/best_checkpoint.pth",
        # },
        # "MobileNetV3_Small": {
        #     "model": mobilenetv3_small,
        #     "ckpt": "/media/icnlab/Data/Thang/plan_dieases/vit_xai/checkpoints/plantvillage/mobilenetv3_small/run_20251021-151012/best_checkpoint.pth",
        # },
        # "MobilePlantViT": {
        #     "model": mobileplantvit,
        #     "ckpt": "/media/icnlab/Data/Thang/plan_dieases/vit_xai/checkpoints/plantvillage/mobileplantvit/run_20260101-103938/best_checkpoint.pth",
        # },
        # "ShuffleNetV2": {
        #     "model": shufflenetv2,
        #     "ckpt": "/media/icnlab/Data/Thang/plan_dieases/vit_xai/checkpoints/plantvillage/shufflenetv2/run_20251022-132921/best_checkpoint.pth",
        # },
        # "SqueezeNet": {
        #     "model": squeezenet,
        #     "ckpt": "/media/icnlab/Data/Thang/plan_dieases/vit_xai/checkpoints/plantvillage/squezzenetv2/run_20251021-171131/best_checkpoint.pth",
        # },
        # "DenseNet121": {
        #     "model": densenet121,
        #     "ckpt": "/media/icnlab/Data/Thang/plan_dieases/vit_xai/checkpoints/plantvillage/densenet121/run_20251018-193243/best_checkpoint.pth",
        # },
    }
    sigmas = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1]

    pv_root = "/media/icnlab/Data/Thang/plan_dieases/vit_xai/data/PlantVillage"
    device = "cuda"

    all_pss_lime = {}

    for model_name, cfg in models.items():
        print(f"\n=== Evaluating PSS-LIME for {model_name} ===")

        model = cfg["model"]
        checkpoint = torch.load(cfg["ckpt"], map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        model_pss = []

        for sigma in sigmas:
            pss_lime = PSS_Lime(
                model=model,
                root_dir=pv_root,
                sigma=sigma
            )

            pss_value = pss_lime()
            model_pss.append(pss_value)

            print(f"[{model_name}] σ={sigma:.3f} → PSS-LIME={pss_value:.4f}")

        all_pss_lime[model_name] = np.array(model_pss)
        del model
        torch.cuda.empty_cache()
    np.savez(
        "lime_pss_all_models_vgg16.npz",
        sigmas=np.array(sigmas),
        **all_pss_lime
    )
