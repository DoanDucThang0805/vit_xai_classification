from lime import lime_image
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
from torchvision import transforms
from skimage.segmentation import mark_boundaries


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

    def batch_predict(self, images):
        batch = torch.stack([self.process(Image.fromarray(np.uint8(img))) for img in images])
        batch = batch.to(self.device)
        with torch.no_grad():
            logits = self.model(batch)
            probs = torch.nn.functional.softmax(logits, dim=1)
        return probs.cpu().numpy()
    
    def __call__(
        self,
        image_path: str,
        image_show: bool = False,
    ):
        image = Image.open(image_path).convert("RGB").resize((224, 224))
        image_np = np.array(image)
        
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            image_np,
            self.batch_predict,
            top_labels=5,
            hide_color=0,
            num_samples=1000
        )
        label_idx = explanation.top_labels[0]
        print("Predicted class index:", label_idx)
        temp, mask = explanation.get_image_and_mask(
            label_idx,
            positive_only=False,
            num_features=10,
            hide_rest=False
        )
        lime_image_np = mark_boundaries(temp / 255.0, mask)
        if image_show:
            plt.imshow(lime_image_np)
            plt.axis('off')
            plt.show()
        return lime_image_np

if __name__ == "__main__":
    from model.vgg16 import model as vgg16
    checkpoint_path = "/media/icnlab/Data/Thang/plan_dieases/vit_xai/checkpoints/plantvillage/vgg16/run_20251019-171608/best_checkpoint.pth"
    image_path = "/media/icnlab/Data/Thang/plan_dieases/vit_xai/data/PlantVillage/Tomato_Septoria_leaf_spot/0a70601b-8511-4a56-9562-c95c46372874___Matt.S_CG 1032.JPG"
    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    model = vgg16
    model.load_state_dict(checkpoint['model_state_dict'])
    lime = Lime(model=model)
    image_path = "/media/icnlab/Data/Thang/plan_dieases/vit_xai/data/PlantVillage/Tomato_Septoria_leaf_spot/0a70601b-8511-4a56-9562-c95c46372874___Matt.S_CG 1032.JPG"
    lime_image = lime(image_path=image_path, image_show=True)
    print(lime_image)
    