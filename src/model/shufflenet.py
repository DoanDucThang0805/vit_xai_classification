from torchvision.models import shufflenet_v2_x1_0
from torchinfo import summary

num_classes = 10

model = shufflenet_v2_x1_0(
    weights=None,          # hoặc "IMAGENET1K_V1" nếu muốn pretrained
    num_classes=num_classes
)

summary(
    model,
    input_size=(1, 3, 224, 224),
    col_names=["input_size", "output_size", "num_params", "mult_adds"],
)
