import timm
from torchinfo import summary

num_classes = 8
model = timm.create_model(
    "shufflenet_v2_x1_0",  # phổ biến nhất
    pretrained=False,
    num_classes=num_classes
)


summary(
    model,
    input_size=(1, 3, 224, 224),
    col_names=["input_size", "output_size", "num_params", "mult_adds"],
)
