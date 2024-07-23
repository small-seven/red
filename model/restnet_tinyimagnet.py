import torchvision
import torch.nn as nn
import torchvision.models as models
from utils import get_parameter_number, get_named_paras
from packaging import version

from utils import get_parameter_number, get_named_paras, init_model
from model.model_utils import DataNorm
from config import Config, get_args

args = Config(get_args())


# define a revised ResNet18 for Tiny-ImageNet
class ResNet18ForTinyImageNet(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet18ForTinyImageNet, self).__init__()
        # use pretrained model
        if version.parse(torchvision.__version__) >= version.parse("0.13.0"):
            self.base_model = models.resnet18(weights="IMAGENET1K_V1")
        else:
            self.base_model = models.resnet18(pretrained=True)
        if not pretrained:
            self.base_model = init_model(self.base_model)

        self.data_norm = DataNorm(mean=args.data_mean, std=args.data_std)

        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, args.num_classes)

    def forward(self, x):
        x = self.data_norm(x)
        x = self.base_model(x)
        return x


if __name__ == '__main__':
    model = ResNet18ForTinyImageNet(pretrained=True)
    get_named_paras(model.base_model)
    get_parameter_number(model.base_model)
