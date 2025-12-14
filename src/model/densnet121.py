import timm
from torchinfo import summary


num_classes = 8
model = timm.create_model(
    "densenet121",
    pretrained=False,
    num_classes=num_classes
)

summary(model, (1,3,224,224), col_names=["input_size", "output_size", "num_params", "mult_adds"])
