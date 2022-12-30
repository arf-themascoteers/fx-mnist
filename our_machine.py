import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet101, ResNet101_Weights


class OurMachine(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet101(weights = ResNet101_Weights.IMAGENET1K_V2)
        # number_input = self.resnet.fc.out_features
        # self.resnet.fc = nn.Sequential(
        #     nn.Linear(number_input, 256)
        # )
        #
        # for param in self.resnet.layer1.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        x = self.resnet(x)
        return F.log_softmax(x, dim=1)
