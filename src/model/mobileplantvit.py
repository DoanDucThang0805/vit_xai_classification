"""
Mobile Plant Vision Transformer (MobileVIT) Model Architecture.

This module implements a mobile-efficient Vision Transformer model designed for 
plant disease classification. It combines depthwise separable convolutions with 
Vision Transformer components for efficient processing.

Key Components:
    - DepthConvBlock: Depthwise separable convolution block with batch norm and GELU activation
    - GroupConvBlock: Group convolution block with similar structure
    - ChannelAttention: Channel attention mechanism for feature refinement
    - SpatialAttention: Spatial attention mechanism for spatial refinement
    - CBAM: Convolutional Block Attention Module
    - PatchEmbedding: Converts image patches to embeddings with positional encoding
    - MobileVIT: Main Vision Transformer architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class DepthConvBlock(nn.Module):
    """
    Depthwise separable convolution block.
    
    This block implements a depthwise convolution followed by a pointwise convolution,
    along with batch normalization and GELU activation. Designed for efficient 
    feature extraction with reduced parameters.
    
    Attributes:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolution kernel
        stride (int): Stride of the convolution
        padding (int): Padding applied to the input
    """
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    padding: int

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 1
    ):
        """
        Initialize depthwise separable convolution block.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Size of the convolution kernel
            stride (int, optional): Stride of the convolution. Defaults to 1.
            padding (int, optional): Padding applied to the input. Defaults to 1.
        """
        super(DepthConvBlock, self).__init__()

        self.depthwise_conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels
        )
        self.pointwise_conv2d = nn.Conv2d(
            in_channels= in_channels,
            out_channels= out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.GELU()

    def forward(self, x):
        """
        Forward pass through the depthwise separable convolution block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height', width')
                         where height' and width' depend on stride and padding
        """
        x = self.depthwise_conv2d(x)
        x = self.pointwise_conv2d(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        return x
    
class GroupConvBlock(nn.Module):
    """
    Group convolution block.
    
    This block implements grouped convolutions followed by a pointwise convolution,
    with batch normalization and GELU activation. Group convolutions provide a 
    middle ground between depthwise and regular convolutions.
    
    Attributes:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolution kernel
        stride (int): Stride of the convolution
        padding (int): Padding applied to the input
        groups (int): Number of groups for group convolution
    """
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    padding: int
    groups: int

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        groups: int,
        padding: int = 1,
    ):
        """
        Initialize group convolution block.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Size of the convolution kernel
            stride (int): Stride of the convolution
            groups (int): Number of groups for group convolution
            padding (int, optional): Padding applied to the input. Defaults to 1.
        """
        super(GroupConvBlock, self).__init__()
        self.depthwise_conv2d = nn.Conv2d(
            in_channels= in_channels,
            out_channels= in_channels,
            kernel_size= kernel_size,
            stride= stride,
            padding= padding,
            groups= groups
        )
        self.pointwise_conv2d = nn.Conv2d(
            in_channels= in_channels,
            out_channels= out_channels,
            kernel_size= 1,
            stride= 1,
            padding= 0
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.GELU()

    def forward(self, x):
        """
        Forward pass through the group convolution block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height', width')
        """
        x = self.depthwise_conv2d(x)
        x = self.pointwise_conv2d(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        return x

class ChannelAttention(nn.Module):
    """
    Channel Attention Module.
    
    Implements channel attention mechanism that recalibrates channel-wise feature 
    responses by explicitly modeling interdependencies between channels using MLP.
    Uses both average pooling and max pooling followed by MLP to generate channel weights.
    
    Attributes:
        in_channels (int): Number of input channels
        reduction (int): Reduction ratio for the MLP hidden dimension
    """
    in_channels: int
    reduction: int

    def __init__(self, in_channels, reduction=16):
        """
        Initialize channel attention module.
        
        Args:
            in_channels (int): Number of input channels
            reduction (int, optional): Reduction ratio for MLP hidden layer. Defaults to 16.
        """
        super(ChannelAttention, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels)
        )

    def forward(self, x):
        """
        Forward pass through channel attention.
        
        Applies global average and max pooling followed by MLP to generate 
        channel attention weights, then multiplies with input.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor with channel attention applied, same shape as input
        """
        b, c, _, _ = x.size()
        # Global Average Pooling
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c)
        # Global Max Pooling
        max_pool = F.adaptive_max_pool2d(x, 1).view(b, c)
        # MLP
        avg_out = self.mlp(avg_pool)
        max_out = self.mlp(max_pool)
        out = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * out  # channel-wise multiplication

class SpatialAttention(nn.Module):
    """
    Spatial Attention Module.
    
    Implements spatial attention mechanism that recalibrates channel-wise feature 
    responses along the spatial dimension. Applies convolution on concatenated 
    average and max pooled features.
    
    Attributes:
        kernel_size (int): Size of the spatial attention convolution kernel
    """
    kernel_size: int

    def __init__(self, kernel_size=7):
        """
        Initialize spatial attention module.
        
        Args:
            kernel_size (int, optional): Kernel size for spatial attention convolution. Defaults to 7.
        """
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)

    def forward(self, x):
        """
        Forward pass through spatial attention.
        
        Applies channel-wise max and average pooling followed by convolution 
        to generate spatial attention weights, then multiplies with input.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor with spatial attention applied, same shape as input
        """
        # max and avg along channel
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        out = torch.sigmoid(self.conv(concat))
        return x * out  # spatial-wise multiplication

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    
    Combines channel attention and spatial attention modules to create a lightweight
    attention mechanism that can be integrated into any convolutional neural network.
    Processes input through channel attention first, then spatial attention.
    
    Reference: Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018
    """
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        """
        Initialize CBAM module.
        
        Args:
            in_channels (int): Number of input channels
            reduction (int, optional): Reduction ratio for channel attention MLP. Defaults to 16.
            kernel_size (int, optional): Kernel size for spatial attention. Defaults to 7.
        """
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        """
        Forward pass through CBAM.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor with channel and spatial attention applied
        """
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x

class PatchEmbedding(nn.Module):
    """
    Patch Embedding Layer.
    
    Converts an input image into patch embeddings with positional encoding. Uses 
    depthwise separable convolution for efficient patch extraction and optional CBAM 
    for attention-based feature refinement.
    
    Attributes:
        patch_size (int): Size of each patch (patch_size x patch_size)
        embed_dim (int): Dimension of the embedding space
        use_cbam (bool): Whether to apply CBAM attention
    """
    
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int, img_size: tuple, use_cbam: bool = True):
        """
        Initialize patch embedding layer.
        
        Args:
            in_channels (int): Number of input channels (typically 3 for RGB images)
            embed_dim (int): Dimension of the patch embeddings
            patch_size (int): Size of patches (patch_size x patch_size pixels)
            img_size (tuple): Size of input image (height, width)
            use_cbam (bool, optional): Whether to apply CBAM attention. Defaults to True.
        """
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.use_cbam = use_cbam

        # Depthwise separable conv để tạo patch embeddings
        self.depthconv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=patch_size,
                stride=patch_size,
                groups=in_channels,  # depthwise
                padding=0
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=embed_dim,
                kernel_size=1,  # pointwise
                stride=1,
                padding=0
            ),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )

        # CBAM block
        if self.use_cbam:
            self.cbam = CBAM(embed_dim)

        # Positional encoding (learnable)
        num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, embed_dim))

    def forward(self, x):
        """
        Forward pass through patch embedding layer.
        
        Extracts patches from the input image, applies embeddings with optional CBAM attention,
        adds positional encoding, and returns the sequence of patch embeddings.
        
        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            torch.Tensor: Embedded patches of shape (batch_size, num_patches, embed_dim)
                         where num_patches = (height/patch_size) * (width/patch_size)
        """
        # DepthConv patch embedding
        x = self.depthconv(x)  # [B, embed_dim, H//patch, W//patch]

        # CBAM
        if self.use_cbam:
            x = self.cbam(x)  # [B, embed_dim, H//patch, W//patch]

        # Flatten patches
        x = x.flatten(2).transpose(1, 2)  # [B, N, D], N = H*W, D = embed_dim

        # Add positional encoding
        x = x + self.pos_embedding[:, :x.size(1), :]
        return x

class LinearAttention(nn.Module):
    def __init__(self, embed_dim):
        """
        Initialize LinearAttention module.
        
        Args:
            embed_dim (int): Dimension of the input embeddings.
        """
        super(LinearAttention, self).__init__()
        self.qkv_proj = nn.Linear(embed_dim, 1 + 2 * embed_dim)  # Q(1) + K(d) + V(d)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        Forward pass for linear attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        _, L, d = x.size()
        qkv = self.qkv_proj(x)  # [B, L, 1+2d]
        Q = qkv[:, :, 0:1]      # [B, L, 1]
        K = qkv[:, :, 1:1+d]    # [B, L, d]
        V = qkv[:, :, 1+d:]     # [B, L, d]

        # Linear attention
        alpha = F.softmax(Q, dim=1)        # [B, L, 1]
        C = (alpha * K).sum(dim=1, keepdim=True)  # [B, 1, d]
        C = C.expand(-1, L, -1)  # broadcast to [B, L, d]

        out = self.out_proj(F.gelu(V) * C)  # [B, L, d]
        return out

class EncoderBlock(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, ffn_dropout: float = 0.2):
        """
        Initialize EncoderBlock for transformer encoder.
        
        Args:
            embed_dim (int): Embedding dimension.
            ffn_dim (int): Feed-forward network hidden dimension.
            ffn_dropout (float, optional): Dropout rate for FFN. Defaults to 0.2.
        """
        super(EncoderBlock, self).__init__()
        self.linear_attn = LinearAttention(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(ffn_dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Forward pass for encoder block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        # Linear Attention + residual + LayerNorm
        x = self.norm1(x + self.linear_attn(x))
        # FFN + residual + LayerNorm
        x = self.norm2(x + self.ffn(x))
        return x

class ClassificationHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.3):
        """
        Initialize classification head for final prediction.
        
        Args:
            input_dim (int): Input feature dimension.
            hidden_dim (int): Hidden layer dimension.
            num_classes (int): Number of output classes.
            dropout (float, optional): Dropout rate. Defaults to 0.3.
        """
        super(ClassificationHead, self).__init__()
        # self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        # self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.output = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        """
        Forward pass for classification head.
        
        Args:
            x (torch.Tensor): Encoder output of shape (batch_size, seq_len, embed_dim)
        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes)
        """
        z = x.mean(dim=1)  # Global average pooling over sequence length [B, d]
        z = self.dropout(z)
        z = self.output(z)
        return z
    
class MobilePlantVit(nn.Module):

    def __init__(
        self,
        image_size: tuple,
        input_channels: int,
        num_classes: int,
        embed_dim: int,
        ffn_dim: int,
        patch_size: int,
        encoder_dropout: float,
        classifier_dropout: float,
    ):
        """
        MobilePlantVit: Mobile Vision Transformer for plant disease classification.

        This model combines mobile-efficient convolutional blocks with transformer-based
        attention for robust and efficient plant disease recognition.

        Args:
            image_size (tuple): Input image size (height, width)
            input_channels (int): Number of input channels (e.g., 3 for RGB)
            num_classes (int): Number of output classes
            embed_dim (int): Embedding dimension for transformer
            ffn_dim (int): Feed-forward network hidden dimension
            patch_size (int): Patch size for patch embedding
            encoder_dropout (float): Dropout rate for encoder block
            classifier_dropout (float): Dropout rate for classifier head
        """
        super(MobilePlantVit, self).__init__()
        self.block1 = DepthConvBlock(
            in_channels=input_channels,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.block2 = nn.Sequential(
            GroupConvBlock(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                groups=16,
                padding=1
            ),
            CBAM(32),
            DepthConvBlock(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1
            )
        )

        self.block3 = nn.Sequential(
            GroupConvBlock(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                groups=32,
                padding=1
            ),
            GroupConvBlock(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                groups=32,
                padding=1
            ),
            CBAM(64),
            DepthConvBlock(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1
            )
        )

        self.block4 = nn.Sequential(
            GroupConvBlock(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                groups=64,
                padding=1
            ),
            GroupConvBlock(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                groups=64,
                padding=1
            ),
            GroupConvBlock(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                groups=64,
                padding=1
            ),
            GroupConvBlock(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                groups=64,
                padding=1
            ),
            CBAM(128),
            DepthConvBlock(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=1
            )
        )

        self.patch_embedding = PatchEmbedding(
            in_channels=256,
            embed_dim=embed_dim,
            patch_size=patch_size,
            img_size=(image_size[0]//8, image_size[1]//8),
            use_cbam=True
        )

        self.encoder_block = EncoderBlock(
            embed_dim=embed_dim,
            ffn_dim=ffn_dim,
            ffn_dropout=encoder_dropout
        )

        self.classifier = ClassificationHead(
            input_dim=embed_dim,
            hidden_dim=ffn_dim,
            num_classes=num_classes,
            dropout=classifier_dropout
        )

    def forward(self, x):
        """
        Forward pass for MobilePlantVit model.
        
        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, input_channels, height, width)
        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.patch_embedding(x)
        x = self.encoder_block(x)
        logits = self.classifier(x)
        return logits
    
model = MobilePlantVit(
    image_size=(224, 224),
    input_channels=3,
    num_classes=8,
    embed_dim=256,
    ffn_dim=512,
    patch_size=7,
    encoder_dropout=0.3,
    classifier_dropout=0.2
)

summary(model, (1,3,224, 224), col_names=["input_size", "output_size", "num_params", "mult_adds"])
