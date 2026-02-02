from typing import List
import numpy as np
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


class GradCam:
    def __init__(self):

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.device = "cuda" if torch.cuda.is_available() else "cpu"


    def _preprocess(self, image_path: str):
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0)
        return tensor.to(self.device), image

    def __call__(
        self,
        model: nn.Module,
        target_layers: List[nn.Module],
        image_path: str,
        image_show: bool = False
    ):
        model.to(self.device)
        model.eval()

        input_tensor = self._preprocess(image_path)[0]

        cam = GradCAM(
            model=model,
            target_layers=target_layers
        )

        with torch.inference_mode():
            logit = model(input_tensor)
            prob = torch.softmax(logit, dim=1)
            pred_class = prob.argmax(dim=1).item()
        
        targets = [ClassifierOutputTarget(pred_class)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        input_image = np.float32(self._preprocess(image_path)[1].resize((224, 224))) / 255.0
        grayscale_cam_map = grayscale_cam[0, :]
        gradcam_image_np = show_cam_on_image(input_image, grayscale_cam_map, use_rgb=True)
        if image_show:
            gradcam_image = Image.fromarray(gradcam_image_np)
            gradcam_image.show()
        return gradcam_image_np



if __name__ == "__main__":
    from model.vgg16 import model as vgg16
    checkpoint_path = "/media/icnlab/Data/Thang/plan_dieases/vit_xai/checkpoints/plantvillage/vgg16/run_20251019-171608/best_checkpoint.pth"
    image_path = "/media/icnlab/Data/Thang/plan_dieases/vit_xai/data/PlantVillage/Tomato_Septoria_leaf_spot/0a70601b-8511-4a56-9562-c95c46372874___Matt.S_CG 1032.JPG"
    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    model = vgg16
    model.load_state_dict(checkpoint['model_state_dict'])
    target_layers = [model.features[26]]
    gradcam = GradCam()
    gradcam_image = gradcam(
        model=model,
        image_path=image_path,
        target_layers=target_layers,
        image_show=True
    )
    print(gradcam_image)
    print(gradcam_image.dtype)
    print(gradcam_image.shape)
