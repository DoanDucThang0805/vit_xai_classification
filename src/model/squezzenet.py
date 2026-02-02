import torch
from torchvision.models import squeezenet1_1
from torchinfo import summary

num_classes = 10

model = squeezenet1_1(
    weights=None,          # hoặc "IMAGENET1K_V1" nếu muốn pretrained
    num_classes=num_classes
)

summary(
    model,
    input_size=(1, 3, 224, 224),
    col_names=["input_size", "output_size", "num_params", "mult_adds"],
)
