from typing import List
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


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
    
    # --- METHOD MỚI ĐỂ PHỤC VỤ TÍNH TOÁN ---
    def get_details(self, model: nn.Module, target_layers: List[nn.Module], image_path: str):
        """
        Trả về heatmap thô và thông tin dự đoán để tính Confidence Drop
        """
        model.to(self.device)
        model.eval()

        input_tensor, _ = self._preprocess(image_path)

        cam = GradCAM(model=model, target_layers=target_layers)

        # 1. Dự đoán ban đầu (Lấy Score)
        with torch.inference_mode():
            logit = model(input_tensor)
            prob = torch.softmax(logit, dim=1)
            # Lấy class dự đoán và độ tin cậy của nó
            pred_conf, pred_class = torch.max(prob, dim=1)
            pred_class = pred_class.item()
            pred_conf = pred_conf.item()

        # 2. Tạo Heatmap
        targets = [ClassifierOutputTarget(pred_class)]
        # grayscale_cam là numpy array (1, 224, 224) giá trị từ 0 đến 1
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        
        # Trả về heatmap của ảnh đầu tiên trong batch
        heatmap = grayscale_cam[0, :] 
        
        return heatmap, pred_class, pred_conf, input_tensor
    

def calculate_confidence_drop(gradcam_instance, model, target_layers, image_path, top_k_percent=0.2):
    """
    Tính độ sụt giảm niềm tin VÀ kiểm tra xem mô hình có đổi class không.
    """
    # 1. Lấy thông tin từ Grad-CAM
    # pred_class ở đây là class gốc ban đầu
    heatmap, orig_pred_class, orig_conf, input_tensor = gradcam_instance.get_details(
        model, target_layers, image_path
    )
    
    # 2. Tạo mặt nạ (Mask)
    flat_heatmap = heatmap.flatten()
    threshold_idx = int(len(flat_heatmap) * (1 - top_k_percent))
    threshold_val = np.sort(flat_heatmap)[threshold_idx]
    
    mask = (heatmap < threshold_val).astype(np.float32)
    mask_tensor = torch.from_numpy(mask).to(gradcam_instance.device)
    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
    
    # 3. Tạo ảnh bị che
    masked_input = input_tensor * mask_tensor

    # 4. Dự đoán lại trên ảnh bị che
    model.eval()
    with torch.inference_mode():
        masked_logit = model(masked_input)
        masked_prob = torch.softmax(masked_logit, dim=1)
        
        # A. Lấy độ tin cậy của class GỐC (để tính Drop Score)
        masked_conf_of_orig_class = masked_prob[0, orig_pred_class].item()
        
        # B. Lấy class MỚI mà mô hình dự đoán (xem nó có đoán nhầm sang cái khác không)
        # new_conf là độ tin cậy của class mới, new_class là index của class mới
        new_conf, new_class_idx = torch.max(masked_prob, dim=1)
        masked_pred_class = new_class_idx.item()

    # 5. Tính độ sụt giảm (Drop %) dựa trên class gốc
    drop_score = (orig_conf - masked_conf_of_orig_class)
    drop_score = max(0, drop_score)
    
    # Trả về thêm: orig_pred_class (lớp cũ) và masked_pred_class (lớp mới)
    return drop_score, orig_conf, masked_conf_of_orig_class, masked_input, orig_pred_class, masked_pred_class


def visualize_drop(gradcam_instance, image_path, gradimg_path, masked_tensor, 
                   orig_conf, masked_conf, drop_score, 
                   orig_class_idx, masked_class_idx, class_names=None):
    
    grad_img = Image.open(gradimg_path).convert("RGB")
    grad_img_np = np.array(grad_img)
    
    # Lấy tên class nếu có danh sách, nếu không thì hiện số index
    orig_label = class_names[orig_class_idx] if class_names else f"Class {orig_class_idx}"
    masked_label = class_names[masked_class_idx] if class_names else f"Class {masked_class_idx}"

    # Lấy ảnh gốc
    _, original_pil = gradcam_instance._preprocess(image_path)
    
    # Xử lý ảnh masked
    masked_np = masked_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
    masked_np = (masked_np - masked_np.min()) / (masked_np.max() - masked_np.min())

    fig, axs = plt.subplots(1, 3, figsize=(12, 6))
    
    # Subplot 1: Ảnh gốc
    axs[0].imshow(original_pil)
    # axs[0].set_title(f"Original: {orig_label}\nConf: {orig_conf:.2%}", color='green', fontweight='bold')
    axs[0].axis('off')

    axs[1].imshow(grad_img_np)
    axs[1].axis("off")
    # Subplot 2: Ảnh bị che
    # Nếu class thay đổi thì tô màu đỏ đậm để cảnh báo
    title_color = 'red' if orig_class_idx != masked_class_idx else 'darkorange'
    
    axs[2].imshow(masked_np)
    # axs[3].set_title(f"Masked Prediction: {masked_label}\n(Orig Class Conf: {masked_conf:.2%})\nDrop: {drop_score:.2%}", 
    #                  color=title_color, fontweight='bold')
    axs[2].axis('off')
    
    plt.tight_layout()
    plt.savefig("/media/icnlab/Data/Thang/plan_dieases/vit_xai/src/xai/masked_img.png",
                bbox_inches='tight',
                pad_inches=0)
    plt.show()


if __name__ == "__main__":
    from model.mobileplantvit import model as vgg16
    checkpoint_path = "/media/icnlab/Data/Thang/plan_dieases/vit_xai/checkpoints/plantvillage/mobileplantvit/run_20260101-103938/best_checkpoint.pth"
    image_path = "/media/icnlab/Data/Thang/plan_dieases/vit_xai/data/PlantVillage/Tomato_Septoria_leaf_spot/0a70601b-8511-4a56-9562-c95c46372874___Matt.S_CG 1032.JPG"
    gradcam_img_path = "/media/icnlab/Data/Thang/plan_dieases/vit_xai/images/xai_images/mobileplantvit_gradcam.png"
    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    model = vgg16
    model.load_state_dict(checkpoint['model_state_dict'])
    target_layers = [model.patch_embedding.cbam]
    # 1. Định nghĩa danh sách tên bệnh chuẩn (Dựa trên dictionary bạn cung cấp)
    # Index từ 0 đến 9 khớp hoàn toàn với mô hình của bạn
    CLASS_NAMES = [
        'Bacterial Spot',                # 0: Tomato_Bacterial_spot
        'Early Blight',                  # 1: Tomato_Early_blight
        'Late Blight',                   # 2: Tomato_Late_blight
        'Leaf Mold',                     # 3: Tomato_Leaf_Mold
        'Septoria Leaf Spot',            # 4: Tomato_Septoria_leaf_spot
        'Spider Mites (Two-spotted)',    # 5: Tomato_Spider_mites_Two_spotted_spider_mite
        'Target Spot',                   # 6: Tomato__Target_Spot
        'Yellow Leaf Curl Virus',        # 7: Tomato__Tomato_YellowLeaf__Curl_Virus
        'Mosaic Virus',                  # 8: Tomato__Tomato_mosaic_virus
        'Healthy'                        # 9: Tomato_healthy
    ]

    # 2. Khởi tạo GradCam
    grad_cam = GradCam()

    # 3. ĐƯỜNG DẪN ẢNH TEST (Thay bằng ảnh thật của bạn)
    # Nên chọn một ảnh bệnh rõ ràng (ví dụ: Early Blight hoặc Leaf Mold) để thấy rõ hiệu quả

    # 4. Tính toán độ sụt giảm niềm tin
    # top_k_percent=0.2 nghĩa là che 20% vùng quan trọng nhất
    drop, orig_conf, masked_conf, masked_tensor, orig_cls, masked_cls = calculate_confidence_drop(
        grad_cam, 
        model, 
        target_layers, 
        image_path, 
        top_k_percent=0.3
    )

    # 5. In kết quả định lượng (Số liệu này dùng để điền vào bảng/viết trong bài)
    print("=" * 60)
    print(f"Original Prediction:  {CLASS_NAMES[orig_cls]:<30} (Conf: {orig_conf:.4f})")
    print(f"Masked Prediction:    {CLASS_NAMES[masked_cls]:<30} (Conf: {masked_conf:.4f})") 
    print("-" * 60)
    print(f"CONFIDENCE DROP:      {drop:.4f}")
    print("=" * 60)

    # 6. Vẽ hình minh họa (Figure) để đưa vào bài báo
    visualize_drop(
        grad_cam, 
        image_path,
        gradcam_img_path,
        masked_tensor, 
        orig_conf, 
        masked_conf, 
        drop, 
        orig_cls, 
        masked_cls, 
        class_names=CLASS_NAMES
    )