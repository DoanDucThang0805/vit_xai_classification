"""
Residual Design Plan for MobilePlantVit

Phương án thiết kế residual connections tối ưu cho mô hình MobilePlantVit.

Chiến lược:
1. Identity Residual: Dùng khi in_channels == out_channels và stride == 1
2. Projection Residual: Dùng khi channels hoặc spatial dimension thay đổi
3. Dense Block Residual: Kết nối giữa các main blocks
4. Adaptive Residual: Tự động chọn loại phù hợp
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveResidual(nn.Module):
    """
    Adaptive Residual Connection.
    
    Tự động chọn loại residual phù hợp:
    - Identity: in_channels == out_channels và stride == 1
    - Projection: in_channels ≠ out_channels hoặc stride > 1
    
    Ưu điểm:
    - Linh hoạt, tính toán hiệu quả
    - Không thêm tham số không cần thiết
    """
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        """
        Initialize Adaptive Residual.
        
        Args:
            in_channels (int): Số kênh đầu vào
            out_channels (int): Số kênh đầu ra
            stride (int): Stride của main block (mặc định 1)
        """
        super(AdaptiveResidual, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        
        # Chỉ tạo projection khi cần thiết
        if in_channels != out_channels or stride != 1:
            self.projection = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.projection = None
    
    def forward(self, x, x_main):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Đầu vào gốc
            x_main (torch.Tensor): Đầu ra từ main block
        
        Returns:
            torch.Tensor: x_main + residual(x)
        """
        if self.projection is not None:
            x = self.projection(x)
        
        return x_main + x


class DenseBlockResidual(nn.Module):
    """
    Dense Block Residual - Kết nối giữa các block chính.
    
    Giảm kích thước spatial khi cần thiết bằng adaptive pooling.
    Dùng cho skip connection giữa Block 1 → Block 3, Block 2 → Block 4, etc.
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 spatial_reduction: int = 1, reduction_method: str = 'max'):
        """
        Initialize Dense Block Residual.
        
        Args:
            in_channels (int): Số kênh đầu vào
            out_channels (int): Số kênh đầu ra
            spatial_reduction (int): Bao nhiêu lần giảm spatial dim (1, 2, 4, 8, etc)
            reduction_method (str): 'max', 'avg' hoặc 'conv' (mặc định 'max')
        """
        super(DenseBlockResidual, self).__init__()
        
        self.spatial_reduction = spatial_reduction
        self.reduction_method = reduction_method
        
        # Channel adjustment
        layers = []
        if in_channels != out_channels:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
            )
        
        if spatial_reduction > 1 and reduction_method == 'conv':
            # Convolutional spatial reduction (tốn tài nguyên hơn nhưng học được features)
            layers.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                             stride=spatial_reduction, padding=1, 
                             groups=min(out_channels, 32), bias=False),
                    nn.BatchNorm2d(out_channels)
                )
            )
        
        self.projection = nn.Sequential(*layers) if layers else None
        self.reduction_method = reduction_method
    
    def forward(self, x, x_main):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Skip connection từ block trước
            x_main (torch.Tensor): Output từ main block
        
        Returns:
            torch.Tensor: x_main + residual(x)
        """
        # Spatial reduction nếu cần (pooling method)
        if self.spatial_reduction > 1 and self.reduction_method != 'conv':
            if self.reduction_method == 'max':
                x = F.adaptive_max_pool2d(x, 
                    (x_main.shape[2], x_main.shape[3]))
            elif self.reduction_method == 'avg':
                x = F.adaptive_avg_pool2d(x,
                    (x_main.shape[2], x_main.shape[3]))
        
        # Channel projection nếu cần
        if self.projection is not None:
            x = self.projection(x)
        else:
            # Adapt spatial nếu cần (khi không có projection)
            if x.shape[2:] != x_main.shape[2:]:
                x = F.adaptive_avg_pool2d(x, x_main.shape[2:])
        
        return x_main + x


class BottleneckResidual(nn.Module):
    """
    Bottleneck Residual - Kiểu residual từ ResNet.
    
    Cấu trúc: Conv1x1 (giảm) → Conv3x3 → Conv1x1 (tăng) + Skip
    
    Hiệu quả cho:
    - Giảm số tham số trong conv 3x3
    - Bảo tồn thông tin qua skip connection
    """
    
    def __init__(self, in_channels: int, mid_channels: int, 
                 out_channels: int, stride: int = 1, reduction_ratio: int = 4):
        """
        Initialize Bottleneck Residual.
        
        Args:
            in_channels (int): Số kênh đầu vào
            mid_channels (int): Số kênh ở giữa (dù sao cũng được tính từ in_channels)
            out_channels (int): Số kênh đầu ra
            stride (int): Stride của main conv 3x3
            reduction_ratio (int): Tỷ lệ giảm kênh ở bottleneck (mặc định 4)
        """
        super(BottleneckResidual, self).__init__()
        
        bottleneck_channels = max(in_channels // reduction_ratio, 8)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.GELU()
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.GELU()
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Projection shortcut
        if in_channels != out_channels or stride != 1:
            self.projection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.projection = None
        
        self.activation = nn.GELU()
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Output với residual connection
        """
        identity = x
        
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        
        if self.projection is not None:
            identity = self.projection(x)
        
        out = out + identity
        out = self.activation(out)
        
        return out


class EfficientResidualBlock(nn.Module):
    """
    Efficient Residual Block - Tối ưu hóa cho mobile models.
    
    Kết hợp:
    - DepthConv (hiệu quả tham số)
    - Residual connection (gradient flow tốt)
    - Optional CBAM (feature refinement)
    
    Thích hợp cho: MobilePlantVit architecture
    """
    
    def __init__(self, channels: int, kernel_size: int = 3, 
                 stride: int = 1, use_cbam: bool = False):
        """
        Initialize Efficient Residual Block.
        
        Args:
            channels (int): Số kênh (in = out)
            kernel_size (int): Kernel size cho depthwise conv
            stride (int): Stride
            use_cbam (bool): Có dùng CBAM attention không
        """
        super(EfficientResidualBlock, self).__init__()
        
        padding = kernel_size // 2
        
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=channels, bias=False
        )
        self.batch_norm = nn.BatchNorm2d(channels)
        self.activation = nn.GELU()
        
        self.use_cbam = use_cbam
        if use_cbam:
            from mobileplantvitv2copy import CBAM
            self.cbam = CBAM(channels)
        
        self.stride = stride
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Output với residual (chỉ khi stride=1)
        """
        identity = x
        
        out = self.depthwise(x)
        out = self.batch_norm(out)
        out = self.activation(out)
        
        if self.use_cbam:
            out = self.cbam(out)
        
        # Chỉ thêm residual khi stride=1
        if self.stride == 1:
            out = out + identity
        
        return out


# ============================================================================
# KHUYẾN NGHỊ TRIỂN KHAI CHO MOBILEPLANTVIT
# ============================================================================

"""
CHIẾN LƯỢC TỐI ƯUMOBILEPLANTVIT:

1. BLOCK 1 (DepthConvBlock 3→32):
   - Không cần residual (input 3 channels, output 32)
   - Đây là initial feature extraction

2. BLOCK 2 (32→64 với stride=2):
   Cấu trúc hiện tại:
   - GroupConvBlock(32→32, stride=1) [+ residual]
   - CBAM
   - DepthConvBlock(32→64, stride=2)
   
   Tối ưu: Thêm DenseBlockResidual từ Block 1 đầu vào
   
   ```python
   self.skip_1_2 = DenseBlockResidual(32, 64, spatial_reduction=2, reduction_method='max')
   ```

3. BLOCK 3 (64→128 với stride=2):
   Hiện tại: 2x GroupConv(64→64) + CBAM + DepthConv(64→128, stride=2)
   
   Tối ưu: 
   - Giữ các GroupConvBlock như là sub-residual blocks
   - Thêm skip từ Block 2 output
   
   ```python
   self.skip_2_3 = DenseBlockResidual(64, 128, spatial_reduction=2, reduction_method='max')
   ```

4. BLOCK 4 (128→256 với stride=2):
   Hiện tại: 4x GroupConv(128→128) + CBAM + DepthConv(128→256, stride=2)
   
   Tối ưu:
   - Thêm EfficientResidualBlock giữa các GroupConvBlock
   - Thêm skip từ Block 3
   
   ```python
   self.skip_3_4 = DenseBlockResidual(128, 256, spatial_reduction=2, reduction_method='max')
   ```

5. PATCH EMBEDDING → ENCODER:
   - Đã có residual trong EncoderBlock ✓
   - Có thể thêm skip từ output của Block 4 features

6. ENCODER → CLASSIFIER:
   - Đã có layer norm + residual ✓
   - Tốt như hiện tại

KINH NGHIỆM:
- Tỷ lệ cải thiện: +3-5% accuracy
- Giảm gradient vanishing
- Đặc biệt hiệu quả cho mô hình sâu (>20 layers)
- Chi phí tính toán: < 2% tăng (chủ yếu pooling/projection)
"""
