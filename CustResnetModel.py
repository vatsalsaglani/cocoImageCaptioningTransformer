import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torchvision import transforms, models

class ResNetCust2(nn.Module):

    def __init__(self, emb_dim):

        super().__init__()
        resnet = models.resnet50(pretrained=True)
        resnet_ = resnet
        weight = resnet_.conv1.weight.clone()
        resnet_.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            resnet_.conv1.weight[:, :3] = weight
            resnet_.conv1.weight[:, 3] = resnet.conv1.weight[:, 0]
        self.model = list(resnet_.children())[:-1]
        self.model = nn.Sequential(*self.model)

    def forward(self, images):

        images = self.model(images)
        return images