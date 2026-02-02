import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
from torchinfo import summary


NUM_CLASSES = 10

model = mobilenet_v3_small(
    pretrained=False
)

# 🔧 Replace classifier head
in_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)



# Summary
summary(model, (1,3,224,224), col_names=["input_size", "output_size", "num_params", "mult_adds"])
