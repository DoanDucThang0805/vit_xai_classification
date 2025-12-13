import torch.nn as nn
import torchvision.models as models
from torchinfo import summary


# Tải VGG16 pretrained
model = models.vgg16(pretrained=False)

# Thay đổi classifier cuối cùng cho 8 class
model.classifier[6] = nn.Linear(in_features=4096, out_features=8)

summary(model, (1,3,224, 224), col_names=["input_size", "output_size", "num_params", "mult_adds"])
