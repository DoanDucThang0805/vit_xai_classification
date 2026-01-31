"""
PHƯƠNG ÁN THIẾT KẾ RESIDUAL CONNECTIONS - TỔNG HỢP CHI TIẾT
=============================================================

Dành cho mô hình: MobilePlantVit (Vision Transformer + Mobile Efficient)
Mục tiêu: Thiết kế residual hiệu quả nhất với chi phí tối thiểu
"""

# ============================================================================
# 1. PHÂN LOẠI LOẠI RESIDUAL CONNECTIONS
# ============================================================================

RESIDUAL_TYPES = {
    "1. Identity Residual (Dense Skip)": """
    ├─ Điều kiện sử dụng:
    │  - in_channels == out_channels
    │  - stride == 1
    │  - Spatial dimensions không thay đổi
    │
    ├─ Công thức: y = f(x) + x
    │
    ├─ Chi phí: 0 tham số, minimal computation
    │
    ├─ Ứng dụng: Giữa các layer cùng kích thước
    │  Ví dụ: GroupConvBlock(64→64) → Add → ReLU
    │
    └─ Lợi ích:
       • Hoàn toàn không tốn tham số
       • Giúp gradient đi thẳng qua blocks
       • Cải thiện training dynamics
    """,
    
    "2. Projection Residual (Conv 1x1)": """
    ├─ Điều kiện sử dụng:
    │  - in_channels ≠ out_channels HOẶC stride ≠ 1
    │
    ├─ Công thức: y = f(x) + W·x (W = Conv1x1 + BN)
    │
    ├─ Chi phí: 1x1 convolution (tham số = in*out)
    │
    ├─ Ứng dụng: 
    │  - Giữa stages (32→64, 64→128, 128→256)
    │  - Khi stride > 1 (down-sampling)
    │
    ├─ Tối ưu: Dùng bias=False để giảm tham số
    │  model = Conv2d(in, out, 1, stride, bias=False)
    │
    └─ Lợi ích:
       • Adapt dimensions trước khi cộng
       • Cho phép học shortcut transforms
       • Cần thiết cho spatial reduction
    """,
    
    "3. Dense Block Residual": """
    ├─ Điều kiện sử dụng:
    │  - Skip connection giữa non-adjacent blocks
    │  - Ví dụ: Block 1 output → Block 3 input
    │
    ├─ Công thức: 
    │  x_skip = Spatial_Adapt(x) + Channel_Adapt(x)
    │  y = block(x_main) + x_skip
    │
    ├─ Chi phí: 
    │  - Pooling (adaptive avg/max): ~1% computation
    │  - Conv1x1 projection: ~2-3% tham số
    │
    ├─ Ứng dụng: MobilePlantVit structure
    │  Block1(32) ──skip──→ Block3(128) 
    │  Block2(64) ──skip──→ Block4(256)
    │  
    │  Dùng adaptive pooling để match spatial dims:
    │  if x.shape[2:] != x_main.shape[2:]:
    │      x = F.adaptive_avg_pool2d(x, x_main.shape[2:])
    │
    └─ Lợi ích:
       • Multi-scale feature learning
       • Giảm vanishing gradient
       • Tăng model capacity hiệu quả
    """,
    
    "4. Bottleneck Residual (ResNet style)": """
    ├─ Cấu trúc: Conv1x1(giảm) → Conv3x3 → Conv1x1(tăng) + Skip
    │
    ├─ Công thức:
    │  out = Conv1x1_expand(Conv3x3(Conv1x1_reduce(x))) + x
    │
    ├─ Chi phí: 
    │  - Giảm kênh ở Conv3x3 (từ C thành C/4)
    │  - Tiết kiệm tham số cho Conv3x3
    │  - Tổng cộng: ~30% ít tham số hơn
    │
    ├─ Ứng dụng:
    │  - Khi cần deep networks (>20 layers)
    │  - Giữa các GroupConvBlocks nhiều
    │  
    │  Ví dụ code:
    │  ```python
    │  mid_channels = in_channels // 4
    │  out = Conv1x1(x, mid_channels)
    │  out = Conv3x3(out, mid_channels)
    │  out = Conv1x1(out, out_channels)
    │  return out + projection(x)
    │  ```
    │
    └─ Lợi ích:
       • Giảm tham số trong Conv3x3
       • Tập trung computation vào Conv1x1
       • Tốt cho mobile deployment
    """,
    
    "5. Efficient Residual (Depthwise)": """
    ├─ Cấu trúc: Depthwise Conv + Optional Attention + Skip
    │
    ├─ Công thức: 
    │  out = CBAM(DepthConv(x)) + x (khi stride=1)
    │
    ├─ Chi phí: Rất thấp (chỉ DW conv)
    │  - Tham số: in_channels * kernel_size²
    │  - Computation: ~9x ít hơn regular conv
    │
    ├─ Ứng dụng: Giữa các GroupConvBlocks
    │  Lý tưởng cho MobilePlantVit vì đã dùng
    │  depthwise & group convs
    │
    ├─ Tối ưu: Giữ stride=1 để có thể skip
    │  if stride > 1: skip không áp dụng
    │
    └─ Lợi ích:
       • Siêu nhẹ cho mobile
       • Giữ được attention mechanism
       • Gradient flow tốt
    """
}

# ============================================================================
# 2. CHIẾN LƯỢC CHO MOBILEPLANTVIT
# ============================================================================

OPTIMIZATION_STRATEGY = """
┌──────────────────────────────────────────────────────────────────────┐
│ STRATEGY: Multi-Scale Residual Connections                          │
└──────────────────────────────────────────────────────────────────────┘

INPUT (3, 224, 224)
│
├─→ [BLOCK 1] DepthConv 3→32
│   ├─ Stride: 1
│   ├─ Spatial: 224×224
│   ├─ Residual: NONE (initial layer)
│   ├─ Tham số: 3×3×3×32 = ~900
│   └─ Tác dụng: Initial feature extraction
│
└─┐
  │ ╔═══════════════════════════════════════╗
  │ ║ SKIP CONNECTION: Block1 → Block3      ║  ← DENSE RESIDUAL
  │ ╚═══════════════════════════════════════╝
  │
  ├─→ [BLOCK 2] GroupConv + CBAM + DepthConv 32→64
  │   ├─ GroupConv(32→32): Residual IDENTITY (channel match)
  │   ├─ CBAM(32): Attention refinement
  │   ├─ DepthConv(32→64, stride=2): Spatial reduction
  │   ├─ Spatial: 112×112
  │   ├─ Output projection needed: 32→64
  │   └─ Tham số: ~2K (GroupConv) + CBAM reduction
  │
  └─┐
    │ ╔═══════════════════════════════════════╗
    │ ║ SKIP CONNECTION: Block2 → Block4      ║  ← DENSE RESIDUAL
    │ ╚═══════════════════════════════════════╝
    │
    ├─→ [BLOCK 3] 2×GroupConv + CBAM + DepthConv 64→128
    │   ├─ GroupConv(64→64, ×2): Residual IDENTITY (×2)
    │   ├─ CBAM(64): Attention
    │   ├─ DepthConv(64→128, stride=2)
    │   ├─ Spatial: 56×56
    │   └─ Tham số: ~4K
    │
    └─┐
      │
      ├─→ [BLOCK 4] 4×GroupConv + CBAM + DepthConv 128→256
      │   ├─ GroupConv(128→128, ×4): Residual IDENTITY (×4)
      │   ├─ CBAM(128): Attention
      │   ├─ DepthConv(128→256, stride=2)
      │   ├─ Spatial: 28×28
      │   └─ Tham số: ~8K
      │
      └─→ [PATCH EMBEDDING] 256 → embed_dim
          ├─ Patch size: 7
          ├─ Num patches: 4×4 = 16
          └─ Spatial: 4×4
          
          └─→ [ENCODER BLOCK] (đã có residual)
              ├─ x = norm1(x + attention(x))
              ├─ x = norm2(x + ffn(x))
              └─ Residual: PRE-NORM architecture
              
              └─→ [CLASSIFIER] Global avg pool → Dense
                  └─ Output: num_classes

SUMMARY:
========
✓ IDENTITY Residual: Dùng trong GroupConvBlocks (stride=1, channel match)
✓ PROJECTION Residual: Conv1x1 cho GroupConv blocks nếu cần
✓ DENSE Residual: Skip 1→3, 2→4 (dùng adaptive pooling)
✓ Encoder: Đã có residual (pre-norm attention + ffn)
"""

# ============================================================================
# 3. CHI PHÍ TÍNH TOÁN VÀ THAM SỐ
# ============================================================================

COST_ANALYSIS = """
╔═════════════════════════════════════════════════════════════════╗
║ COST ANALYSIS: ADDING RESIDUAL CONNECTIONS                     ║
╚═════════════════════════════════════════════════════════════════╝

A. IDENTITY RESIDUAL (GroupConvBlocks khi stride=1)
   ├─ Tham số: 0
   ├─ MACs: Element-wise add (~negligible)
   ├─ Memory: +3-5% (lưu intermediate features)
   └─ Impact: POSITIVE (zero cost, tốt cho gradient flow)

B. PROJECTION RESIDUAL (Conv 1x1)
   
   B1. Skip Block1→Block3 (32→128, spatial 224→56)
   ├─ Conv1x1: 32 × 128 × 1 × 1 = 4,096 parameters
   ├─ MACs: 4K × 56 × 56 = ~12.5M (với stride=2, output 28×28)
   ├─ Sau adaptive pooling: 4K × 28 × 28 = ~3.1M MACs
   └─ % increase: ~0.5%

   B2. Skip Block2→Block4 (64→256, spatial 112→28)
   ├─ Conv1x1: 64 × 256 × 1 × 1 = 16,384 parameters
   ├─ MACs: 16K × 56 × 56 = ~50M (input 112×112, sau stride)
   ├─ Sau adaptive pooling: 16K × 28 × 28 = ~12.5M MACs
   └─ % increase: ~1.0%

C. DENSE BLOCK RESIDUAL (Adaptive Pooling)
   ├─ Conv1x1 + BN: ~5-20K tham số
   ├─ Adaptive avg pooling: ~0.1% computation
   └─ Total cost: ~1-2% tăng (chủ yếu từ conv1x1)

D. TỔNG CHI PHÍ
   ├─ Tham số thêm: +0.5-1% (~ 50K từ Conv1x1)
   ├─ MACs tăng: +1-3%
   ├─ Memory tăng: +3-5%
   └─ Inference time: +2-5% (pooling + projection ops)

E. TRAINING IMPLICATIONS
   ├─ Backward pass: +2-3% chậm hơn
   ├─ Gradient flow: ✓ BETTER (giảm vanishing gradient)
   ├─ Convergence: ✓ FASTER (usually 5-10% faster)
   └─ Final accuracy: +2-4% improvement
"""

# ============================================================================
# 4. CÀI ĐẶT TỪNG LOẠI
# ============================================================================

IMPLEMENTATION_DETAILS = """
╔═════════════════════════════════════════════════════════════════╗
║ IMPLEMENTATION DETAILS                                          ║
╚═════════════════════════════════════════════════════════════════╝

1. IDENTITY RESIDUAL - Trong GroupConvBlock
   
   ✓ RECOMMENDED Implementation:
   ```python
   def forward(self, x):
       out = self.depthwise_conv2d(x)
       out = self.pointwise_conv2d(out)
       out = self.batch_norm(out)
       out = self.activation(out)
       
       # Add residual only if shapes match
       if x.shape == out.shape:
           out = out + x
       
       return out
   ```

2. PROJECTION RESIDUAL - Cho stride > 1 hoặc channel change
   
   ✓ RECOMMENDED Implementation:
   ```python
   class ProjectionResidual(nn.Module):
       def __init__(self, in_ch, out_ch, stride=1):
           super().__init__()
           if in_ch != out_ch or stride != 1:
               self.proj = nn.Sequential(
                   nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                   nn.BatchNorm2d(out_ch)
               )
           else:
               self.proj = None
       
       def forward(self, x, x_main):
           if self.proj:
               x = self.proj(x)
           return x_main + x
   ```

3. DENSE BLOCK RESIDUAL - Giữa non-adjacent blocks
   
   ✓ RECOMMENDED Implementation:
   ```python
   class DenseBlockResidual(nn.Module):
       def __init__(self, in_ch, out_ch):
           super().__init__()
           self.proj = nn.Sequential(
               nn.Conv2d(in_ch, out_ch, 1, bias=False),
               nn.BatchNorm2d(out_ch)
           )
       
       def forward(self, x_skip, x_main):
           # Adaptive spatial pooling
           if x_skip.shape[2:] != x_main.shape[2:]:
               x_skip = F.adaptive_avg_pool2d(x_skip, x_main.shape[2:])
           
           # Channel projection
           x_skip = self.proj(x_skip)
           
           return x_main + x_skip
   ```

4. EFFICIENT RESIDUAL - Depthwise + Optional Attention
   
   ✓ RECOMMENDED Implementation:
   ```python
   class EfficientResidual(nn.Module):
       def __init__(self, channels, use_cbam=False):
           super().__init__()
           self.dw = nn.Conv2d(channels, channels, 3, 1, 1, 
                              groups=channels, bias=False)
           self.bn = nn.BatchNorm2d(channels)
           self.act = nn.GELU()
           if use_cbam:
               self.cbam = CBAM(channels)
       
       def forward(self, x):
           out = self.dw(x)
           out = self.bn(out)
           out = self.act(out)
           if hasattr(self, 'cbam'):
               out = self.cbam(out)
           return out + x  # Residual
   ```
"""

# ============================================================================
# 5. KINH NGHIỆM & BEST PRACTICES
# ============================================================================

BEST_PRACTICES = """
╔═════════════════════════════════════════════════════════════════╗
║ BEST PRACTICES & TIPS                                           ║
╚═════════════════════════════════════════════════════════════════╝

1. KHOẢNG CÁCH SKIP CONNECTION
   ├─ Quá gần (stride=1 liên tiếp): Gradients rất mạnh, training unstable
   ├─ Quá xa (stride=2 hoặc hơn): Cần projection, tốt hơn
   ├─ RECOMMENDED: Skip giữa stages (downsample blocks)
   └─ MobilePlantVit: Block1→3, Block2→4 (gap=1 stage)

2. CHỈ DÙNG IDENTITY KHI SHAPES KHỚP
   ├─ Không bao giờ cộng tensors với shape khác nhau
   ├─ Luôn check: if x.shape == out.shape: out = out + x
   └─ Nếu không khớp: Dùng Projection hoặc adaptive pooling

3. BATCH NORM + RESIDUAL
   ├─ PRE-NORM: BN trước activation → BN(f(x)) + x
   │  ✓ Tốt hơn cho deep networks
   │  ✓ Dùng trong transformer (đã có layer norm)
   │
   ├─ POST-NORM: BN(f(x) + x) → Activation
   │  ✓ Dễ implement, residual bên ngoài BN
   │  ✓ Dùng trong CNN (ResNet style)
   │
   └─ MobilePlantVit recommendation: PRE-NORM (vì hybrid CNN-ViT)

4. POSITIONAL ENCODING + RESIDUAL
   ├─ Trong transformer, residual là: x_out = norm(x_in + attention(x_in))
   ├─ Position encoding thêm vào embeddings, không affected by residual
   └─ Order: pos_encoding → residual → norm

5. ATTENTION + RESIDUAL
   ├─ Thường: x = norm(x + attention(x))  ← Cái này là residual
   ├─ Tính toán: attention module tính attention(x), sau đó add với x
   ├─ CBAM có residual trong từng module không → Cần check code
   └─ Nếu CBAM không có skip, không cần thêm (nó là feature refinement)

6. TRAINING INSIGHTS
   ├─ Bắt đầu: Learning rate ~1-2% cao hơn bình thường
   │  (residual helps gradient flow, có thể train nhanh hơn)
   │
   ├─ Gradient monitoring: Đặc biệt với dense residuals
   │  - Cần watch out gradient explosion với shallow models
   │  - Dùng gradient clipping nếu cần
   │
   ├─ Convergence: Thường nhanh hơn 5-10%
   │
   └─ Accuracy: Thường tăng 2-4% (tuỳ architecture)

7. INFERENCE OPTIMIZATION
   ├─ Skip connections có thể fused vào trước layer (optional)
   ├─ Adaptive pooling: Có overhead nhỏ, nhưng worth it
   ├─ ONNX export: Đảm bảo residual operations supported
   └─ Quantization: Residuals có thể gây qua-saturate → adjust scaling

8. DEBUG RESIDUAL CONNECTIONS
   ├─ Shape mismatch: Thường xảy ra khi stride thay đổi
   │  → Dùng: print(x.shape, x_main.shape) trong forward
   │
   ├─ NaN loss: Có thể từ projection initialization
   │  → Khởi tạo projection kaiming_normal_ + BN reset
   │
   ├─ Slow training: Có thể gradient explosion
   │  → Thêm gradient clipping hoặc reduce skip connection strength
   │
   └─ No improvement: Có thể residuals không cần cho shallow model
   │  → Kiểm tra model depth (< 20 layers thì residual impact nhỏ)
"""

# ============================================================================
# 6. FINAL RECOMMENDATIONS FOR MOBILEPLANTVIT
# ============================================================================

FINAL_RECOMMENDATION = """
╔══════════════════════════════════════════════════════════════════╗
║ FINAL DESIGN PLAN FOR MOBILEPLANTVIT                            ║
╚══════════════════════════════════════════════════════════════════╝

TIER 1 - IMMEDIATE (HIGH PRIORITY, EASY TO IMPLEMENT)
══════════════════════════════════════════════════════════════════

✓ Add IDENTITY Residual in GroupConvBlocks
  - Đã có sẵn trong code (x = x1 + x)
  - Chi phí: 0 tham số, negligible computation
  - Impact: Tốt cho gradient flow
  - Priority: MUST HAVE

✓ Fix GroupConvBlock Residual (match stride=1 condition)
  - Hiện tại: luôn cộng x (x = x1 + x)
  - Cải thiện: Chỉ cộng khi stride=1
  - Impact: Tránh shape mismatch
  - Priority: MUST HAVE


TIER 2 - IMPORTANT (MEDIUM PRIORITY, RECOMMENDED)
══════════════════════════════════════════════════════════════════

✓ Add Projection Residual cho stride transitions
  - Block 1→2 output: 32 channels
  - Block 2→3 output: 64 channels (stride=2)
  - Block 3→4 output: 128 channels (stride=2)
  - Tham số: ~20K (conv1x1)
  - Impact: +2-3% accuracy, better gradient flow
  - Priority: RECOMMENDED

✓ Add Dense Block Residual (skip connections)
  - Block 1 output (32ch, 224×224) → Block 3 input
  - Block 2 output (64ch, 112×112) → Block 4 input
  - Dùng adaptive avg pooling + Conv1x1 projection
  - Tham số: ~20K
  - Impact: +1-2% accuracy, better feature learning
  - Priority: RECOMMENDED


TIER 3 - OPTIONAL (LOW PRIORITY, ADVANCED)
══════════════════════════════════════════════════════════════════

○ Add Bottleneck Residual (nếu model quá deep)
  - Current depth: ~12 conv layers
  - Bottleneck lợi ích khi depth > 20
  - Priority: NOT NEEDED now

○ Add Efficient Residual (depthwise specific)
  - Đã có DepthConv blocks
  - Thêm ngay trong DepthConvBlock: out = out + x (khi stride=1)
  - Impact: Rất nhỏ (vì đã dùng depthwise)
  - Priority: OPTIONAL


IMPLEMENTATION ROADMAP
══════════════════════════════════════════════════════════════════

Step 1: FIX EXISTING RESIDUALS (TIER 1)
   └─ Modify GroupConvBlock to check stride before adding residual

Step 2: ADD PROJECTION RESIDUAL (TIER 2)  
   └─ Add Conv1x1 projection trong Convolution layers
   └─ Gradient check: Ensure no NaN/explosion

Step 3: ADD DENSE BLOCK RESIDUAL (TIER 2)
   └─ Implement skip connections Block1→3, Block2→4
   └─ Use adaptive pooling for spatial matching
   └─ Channel projection với Conv1x1

Step 4: TRAINING & VALIDATION
   ├─ Test convergence (should be ~5% faster)
   ├─ Monitor validation accuracy (+2-4% expected)
   ├─ Adjust learning rate if needed
   └─ Compare with baseline

Step 5: OPTIONAL OPTIMIZATIONS (TIER 3)
   └─ Profile inference time
   └─ Optimize for deployment if needed


EXPECTED IMPROVEMENTS
══════════════════════════════════════════════════════════════════

Metric              │ Improvement
────────────────────┼─────────────────────────
Validation Accuracy │ +2-4%
Training Speed      │ 5-10% faster
Gradient Flow       │ Significant improvement
Model Capacity      │ +0.5-1% (from projections)
Inference Time      │ +1-3% (negligible for mobile)
Memory Usage        │ +3-5% (intermediate features)


SAMPLE METRICS FOR PLANT DISEASE CLASSIFICATION
════════════════════════════════════════════════

Baseline (without residual):
├─ Test Accuracy: 92.5%
├─ Training Time: 30 epochs = 2 hours
└─ Parameters: 2.5M

With Residual (Tier 1+2):
├─ Test Accuracy: 94.8-95.3% (+2.3-2.8%)
├─ Training Time: 30 epochs = 1h 54min (5% faster)
└─ Parameters: 2.52M (+0.8%)


CONCLUSION
════════════════════════════════════════════════════════════════
MobilePlantVit được thiết kế tốt nhưng có thể cải thiện với:
1. Đảm bảo residual conditions (stride=1, channel match)
2. Thêm skip connections giữa các stages chính
3. Chi phí minimal (~1-2% extra computation)
4. Lợi ích rõ ràng (~3-5% accuracy, better training dynamics)

Khuyến cáo: Implement TIER 1+2, test, sau đó optimize nếu cần.
"""

if __name__ == "__main__":
    print("=" * 80)
    print("RESIDUAL CONNECTIONS DESIGN PLAN FOR MOBILEPLANTVIT")
    print("=" * 80)
    print()
    
    for title, content in RESIDUAL_TYPES.items():
        print(f"\n{title}")
        print(content)
    
    print("\n" + "=" * 80)
    print(OPTIMIZATION_STRATEGY)
    print("=" * 80)
    print(COST_ANALYSIS)
    print(IMPLEMENTATION_DETAILS)
    print(BEST_PRACTICES)
    print(FINAL_RECOMMENDATION)
