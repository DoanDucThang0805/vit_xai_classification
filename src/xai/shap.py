import io

import shap
from shap import Explanation
from shap.plots import colors
from shap import maskers, Explainer
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn

from model.vgg16 import model as vgg16


class Shap:
    def __init__(self, model: nn.Module):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).to(self.device).view(1, 3, 1, 1)
        self.model.to(self.device)
        self.model.eval()

    def _predictor(self, images: np.ndarray):
        # 1. Chuyển Numpy -> Tensor
        # SHAP gửi ảnh dạng (0, 255) hoặc (0, 1). Ta cần đảm bảo về (0, 1) float trước
        if np.max(images) > 1.0:
            images = images / 255.0
        tensor_img = torch.tensor(images).float().to(self.device)
        # 2. Chuyển dimension: (B, H, W, C) -> (B, C, H, W) cho PyTorch
        tensor_img = tensor_img.permute(0, 3, 1, 2)
        # 3. Normalize theo chuẩn ImageNet
        tensor_img = (tensor_img - self.mean) / self.std
        # 4. Dự đoán
        with torch.no_grad():
            logits = self.model(tensor_img)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()
    
    def image_plot_heatmap_only(
        self,
        shap_values,
        pixel_values=None,
        labels=None,
        true_labels=None,
        width=20,
        aspect=0.2,
        hspace=0.2,
        labelpad=None,
        cmap=colors.red_transparent_blue, # Sử dụng màu chuẩn SHAP
        vmax=None,
        show=False,
    ):
        """
        Phiên bản tùy biến của shap.image_plot:
        - KHÔNG hiển thị cột ảnh gốc đầu tiên.
        - Chỉ hiển thị Heatmap giải thích trên nền xám.
        """

        # --- 1. XỬ LÝ INPUT (GIỮ NGUYÊN LOGIC CỦA SHAP) ---
        if isinstance(shap_values, Explanation):
            shap_exp = shap_values
            if len(shap_exp.output_dims) == 1:
                shap_values = [shap_exp.values[..., i] for i in range(shap_exp.values.shape[-1])]
            elif len(shap_exp.output_dims) == 0:
                shap_values = [shap_exp.values]
            else:
                raise Exception("Number of outputs needs to have support added!!")
            
            if pixel_values is None:
                pixel_values = shap_exp.data
            if labels is None:
                labels = shap_exp.output_names

        if not isinstance(shap_values, list):
            shap_values = [shap_values]

        if len(shap_values[0].shape) == 3:
            shap_values = [v.reshape(1, *v.shape) for v in shap_values]
            pixel_values = pixel_values.reshape(1, *pixel_values.shape)

        if labels is not None:
            if isinstance(labels, list):
                labels = np.array(labels)
            labels = labels.reshape(-1, len(shap_values))

        label_kwargs = {} if labelpad is None else {"pad": labelpad}

        # --- 2. THIẾT LẬP LAYOUT (ĐÃ SỬA: GIẢM SỐ CỘT) ---
        x = pixel_values
        
        # SỬA: ncols = len(shap_values) thay vì len(shap_values) + 1
        # Bỏ cột dành cho ảnh gốc
        ncols = len(shap_values) 
        
        # Tính toán kích thước hình
        fig_size = np.array([3 * ncols, 2.5 * (x.shape[0] + 1)])
        if fig_size[0] > width:
            fig_size *= width / fig_size[0]
            
        # Tạo subplots
        fig, axes = plt.subplots(nrows=x.shape[0], ncols=ncols, figsize=fig_size, squeeze=False)

        # --- 3. VÒNG LẶP VẼ TỪNG HÀNG (SỬA LOGIC INDEX) ---
        for row in range(x.shape[0]):
            x_curr = x[row].copy()

            # Xử lý input shape
            if len(x_curr.shape) == 3 and x_curr.shape[2] == 1:
                x_curr = x_curr.reshape(x_curr.shape[:2])

            # --- TẠO ẢNH NỀN GRAYSCALE ---
            # Logic: Chuyển ảnh màu sang đen trắng để làm nền cho heatmap
            if len(x_curr.shape) == 3 and x_curr.shape[2] == 3:
                # RGB chuẩn
                x_curr_gray = 0.2989 * x_curr[:, :, 0] + 0.5870 * x_curr[:, :, 1] + 0.1140 * x_curr[:, :, 2]
            elif len(x_curr.shape) == 3:
                # Ảnh nhiều kênh (Feature maps, Satellite...) -> Lấy trung bình
                x_curr_gray = x_curr.mean(2) 
                # (Đã lược bỏ phần K-Means phức tạp để code chạy độc lập nhẹ nhàng hơn)
            else:
                x_curr_gray = x_curr

            # --- TÍNH VMAX (ĐỘ ĐẬM MÀU) ---
            if len(shap_values[0][row].shape) == 2:
                abs_vals = np.stack([np.abs(shap_values[i]) for i in range(len(shap_values))], 0).flatten()
            else:
                abs_vals = np.stack([np.abs(shap_values[i].sum(-1)) for i in range(len(shap_values))], 0).flatten()

            max_val = np.nanpercentile(abs_vals, 99.9) if vmax is None else vmax

            # --- VẼ HEATMAP (SỬA INDEX) ---
            for i in range(len(shap_values)):
                # SỬA: Dùng axes[row, i] thay vì axes[row, i+1] 
                # (Vì cột đầu tiên đã bị xóa, index lùi về 0)
                ax_curr = axes[row, i]

                # Set title
                if labels is not None:
                    if labels.shape[0] > 1 or row == 0:
                        ax_curr.set_title(labels[row, i], **label_kwargs)
                
                # Lấy giá trị SHAP (gộp kênh màu nếu cần)
                sv = shap_values[i][row] if len(shap_values[i][row].shape) == 2 else shap_values[i][row].sum(-1)
                
                # 1. Vẽ nền xám mờ (alpha=0.15 là chuẩn của thư viện)
                ax_curr.imshow(
                    x_curr_gray, cmap=plt.get_cmap("gray"), alpha=0.15, 
                    extent=(-1, sv.shape[1], sv.shape[0], -1)
                )
                
                # 2. Vẽ Heatmap đè lên
                im = ax_curr.imshow(sv, cmap=cmap, vmin=-max_val, vmax=max_val)
                
                # Tắt khung viền
                ax_curr.axis("off")

        # --- 4. HOÀN THIỆN COLORBAR ---
        if hspace == "auto":
            fig.tight_layout()
        else:
            fig.subplots_adjust(hspace=hspace)
            
        # Vẽ thanh màu
        # cb = fig.colorbar(
        #     im, ax=np.ravel(axes).tolist(), label="SHAP value", 
        #     orientation="horizontal", aspect=fig_size[0] / aspect
        # )
        # Vẽ thanh màu
        # cb = fig.colorbar(
        #     im, 
        #     ax=np.ravel(axes).tolist(), 
        #     label="SHAP value", 
        #     orientation="vertical",   # <--- Đổi thành vertical
        #     fraction=0.02,            # <--- Độ rộng của thanh màu (0.02 = 2% chiều rộng hình)
        #     pad=0.04                  # <--- Khoảng cách giữa hình và thanh màu
        # )
        # cb.outline.set_visible(False) # Tắt viền thanh màu

        if show:
            plt.show()
        # --- CODE MỚI: LẤY MA TRẬN ẢNH SÁT LỀ (CROP TIGHT) ---
        
        # 1. Tạo một buffer ảo trong RAM
        buf = io.BytesIO()
        
        # 2. Lưu figure vào buffer đó
        # bbox_inches='tight': Tự động cắt bỏ phần trắng thừa
        # pad_inches=0: Đặt khoảng cách đệm về 0 tuyệt đối
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
        
        # 3. Đưa con trỏ về đầu file buffer để đọc
        buf.seek(0)
        
        # 4. Đọc ảnh từ buffer bằng PIL
        img = Image.open(buf)
        
        # 5. Chuyển sang chuỗi NumPy (H, W, 4) - kênh 4 là Alpha (trong suốt)
        # Nếu muốn bỏ kênh Alpha (chỉ lấy RGB), dùng .convert('RGB')
        image_matrix = np.array(img.convert('RGB'))
        
        # Đóng figure và buffer để giải phóng RAM
        buf.close()

        return image_matrix

    def __call__(self, image_path: str, show: bool=False):
        image = Image.open(image_path).convert("RGB").resize((224, 224))
        input_image = np.clip(np.array(image), 0, 255)
        masker = maskers.Image("inpaint_telea", input_image.shape)
        explainer = Explainer(self._predictor, masker)
        shap_values = explainer(
            input_image[np.newaxis, ...],
            max_evals=10000,
            batch_size=50
        )
        probs = self._predictor(input_image[np.newaxis, ...])
        pred_class = np.argmax(probs, axis=1)[0]
        print(f"Predicted class: {pred_class}, Probability: {probs[0][pred_class]:.4f}")
        image_shap = self.image_plot_heatmap_only(
            shap_values=shap_values.values[..., pred_class],
            pixel_values=shap_values.data,
            show = show
        )
        return image_shap
    

if __name__ == "__main__":
    from model.vgg16 import model as vgg16
    model = vgg16
    checkpoint_path = "/media/icnlab/Data/Thang/plan_dieases/vit_xai/checkpoints/plantvillage/vgg16/run_20251019-171608/best_checkpoint.pth"
    image_path = "/media/icnlab/Data/Thang/plan_dieases/vit_xai/data/PlantVillage/Tomato_Septoria_leaf_spot/0a70601b-8511-4a56-9562-c95c46372874___Matt.S_CG 1032.JPG"
    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    model.load_state_dict(checkpoint['model_state_dict'])
    shap_explainer = Shap(model=model)
    shap_image = shap_explainer(image_path=image_path, show=False)
    print(shap_image.shape)
