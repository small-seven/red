import torchvision
import torch.nn as nn

from torchvision import models
from packaging import version

from utils import get_parameter_number, get_named_paras, init_model
from model.model_utils import DataNorm
from config import Config, get_args

args = Config(get_args())


class ResNet18ForCIFAR(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet18ForCIFAR, self).__init__()
        # 使用预训练的ResNet18作为特征提取器
        if version.parse(torchvision.__version__) >= version.parse("0.13.0"):
            self.base_model = models.resnet18(weights="IMAGENET1K_V1")
        else:
            self.base_model = models.resnet18(pretrained=True)
        if not pretrained:
            self.base_model = init_model(self.base_model)

        self.data_norm = DataNorm(mean=args.data_mean, std=args.data_std)

        # 替换最后的全连接层以匹配CIFAR-10的类别数
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, args.num_classes)

        # 由于CIFAR-10的图像较小，我们移除第一个卷积层和最大池化层，并进行调整
        self.base_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.base_model.bn1 = nn.BatchNorm2d(64)
        self.base_model.maxpool = nn.Identity()  # 使用恒等层替换最大池化层

    def forward(self, x):
        x = self.data_norm(x)
        x = self.base_model(x)
        return x


if __name__ == '__main__':
    # 实例化模型
    model = ResNet18ForCIFAR(pretrained=True)

    print(model)
    get_named_paras(model.base_model)
    get_parameter_number(model.base_model)
