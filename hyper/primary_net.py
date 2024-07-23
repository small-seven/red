import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from hyper.hypernetwork_modules import HyperNetwork
from hyper.resnet_blocks import ResNetBlock18, ResNetBlock50
from model.model_utils import DataNorm
from config import Config, get_args

args = Config(get_args())


class Embedding(nn.Module):

    def __init__(self, z_num, z_dim):
        super(Embedding, self).__init__()

        self.z_num = z_num
        self.z_dim = z_dim

        h, k = self.z_num

        for i in range(h * k):
            self.register_parameter(f'z_list_{i}', Parameter(torch.rand(self.z_dim)))

    def forward(self, hyper_net):
        ww = []
        h, k = self.z_num
        for i in range(h):
            w = []
            for j in range(k):
                w.append(hyper_net(eval(f'self.z_list_{i * k + j}')))
            ww.append(torch.cat(w, dim=1))
        return torch.cat(ww, dim=0)


# Hyper-ResNet18 Network Architecture
class PrimaryNetwork18(nn.Module):
    expansion = 1

    def __init__(self, z_dim=128):
        super(PrimaryNetwork18, self).__init__()
        self.data_norm = DataNorm(mean=args.data_mean, std=args.data_std)

        self.z_dim = z_dim
        self.hope = HyperNetwork(z_dim=self.z_dim)

        self.zs_size = [
            [1, 1], [1, 1], [1, 1], [1, 1],  # layer 1
            [2, 1], [2, 1], [2, 2], [2, 2], [2, 2],  # layer 2
            [4, 2], [4, 2], [4, 4], [4, 4], [4, 4],  # layer 3
            [8, 4], [8, 4], [8, 8], [8, 8], [8, 8]  # layer 4
        ]

        self.filter_size = [
            [64, 64], [64, 64], [64, 64], [64, 64],  # layer 1
            [128, 64], [128, 64], [128, 128], [128, 128], [128, 128],  # layer 2
            [256, 128], [256, 128], [256, 256], [256, 256], [256, 256],  # layer 3
            [512, 256], [512, 256], [512, 512], [512, 512], [512, 512]  # layer 4
        ]

        # first layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # four Layers
        self.res_net = nn.ModuleList()
        j = 0
        for i in range(8):
            if self.filter_size[j][0] == self.filter_size[j][1]:
                self.res_net.append(ResNetBlock18(None, self.filter_size[j], self.filter_size[j + 1]))
                j += 2
            else:
                self.res_net.append(
                    ResNetBlock18(self.filter_size[j], self.filter_size[j + 1], self.filter_size[j + 2]))
                j += 3

        self.zs = nn.ModuleList()
        for i in range(len(self.zs_size)):
            self.zs.append(Embedding(self.zs_size[i], self.z_dim))

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # self.global_avg = nn.AvgPool2d(4)
        self.final = nn.Linear(512 * self.expansion, args.num_classes)

    def forward(self, x):
        x = self.data_norm(x)
        out = F.relu(self.bn1(self.conv1(x)))
        j = 0
        for i in range(8):
            if i > 0 and i % 2 == 0:
                short_w = self.zs[j](self.hope)
                conv1_w = self.zs[j + 1](self.hope)
                conv2_w = self.zs[j + 2](self.hope)
                out = self.res_net[i](out, short_w, conv1_w, conv2_w)
                j += 3
            else:
                conv1_w = self.zs[j](self.hope)
                conv2_w = self.zs[j + 1](self.hope)
                out = self.res_net[i](out, None, conv1_w, conv2_w)
                j += 2

        out = self.avg_pool(out)
        out = self.final(out.view(-1, 512))

        return out


class PrimaryNetwork50(nn.Module):
    expansion = 4

    def __init__(self, z_dim=64, num_classes=10, data_bn=None):
        super(PrimaryNetwork50, self).__init__()
        self.data_bn = data_bn

        self.z_dim = z_dim
        self.hypernet = HyperNetwork(z_dim=self.z_dim)
        self.zs_size = [
            [1, 1], [1, 1], [4, 1], [4, 1],  # layer1.0 + shortcut
            [1, 4], [1, 1], [4, 1],  # layer1.1
            [1, 4], [1, 1], [4, 1],  # layer1.2
            [2, 4], [2, 2], [8, 2], [8, 4],  # layer2.0 + shortcut
            [2, 8], [2, 2], [8, 2],  # layer2.1
            [2, 8], [2, 2], [8, 2],  # layer2.2
            [2, 8], [2, 2], [8, 2],  # layer2.3
            [4, 8], [4, 4], [16, 4], [16, 8],  # layer3.0 + shortcut
            [4, 16], [4, 4], [16, 4],  # layer3.1
            [4, 16], [4, 4], [16, 4],  # layer3.2
            [4, 16], [4, 4], [16, 4],  # layer3.3
            [4, 16], [4, 4], [16, 4],  # layer3.4
            [4, 16], [4, 4], [16, 4],  # layer3.5
            [8, 16], [8, 8], [32, 8], [32, 16],  # layer4.0 + shortcut
            [8, 32], [8, 8], [32, 8],  # layer4.1
            [8, 32], [8, 8], [32, 8],  # layer4.2
        ]

        self.filter_size = [
            [[64, 64], [64, 64], [256, 64], [256, 64]],  # layer1.0 + shortcut
            [[64, 256], [64, 64], [256, 64]],  # layer1.1
            [[64, 256], [64, 64], [256, 64]],  # layer1.2
            [[128, 256], [128, 128], [512, 128], [512, 256]],  # layer2.0 + shortcut
            [[128, 512], [128, 128], [512, 128]],  # layer2.1
            [[128, 512], [128, 128], [512, 128]],  # layer2.2
            [[128, 512], [128, 128], [512, 128]],  # layer2.3
            [[256, 512], [256, 256], [1024, 256], [1024, 512]],  # layer3.0 + shortcut
            [[256, 1024], [256, 256], [1024, 256]],  # layer3.1
            [[256, 1024], [256, 256], [1024, 256]],  # layer3.2
            [[256, 1024], [256, 256], [1024, 256]],  # layer3.3
            [[256, 1024], [256, 256], [1024, 256]],  # layer3.4
            [[256, 1024], [256, 256], [1024, 256]],  # layer3.5
            [[512, 1024], [512, 512], [2048, 512], [2048, 1024]],  # layer4.0 + shortcut
            [[512, 2048], [512, 512], [2048, 512]],  # layer4.1
            [[512, 2048], [512, 512], [2048, 512]],  # layer4.2
        ]

        # first layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # four Layers
        self.res_net = nn.ModuleList()
        for i in range(len(self.filter_size)):
            if len(self.filter_size[i]) == 3:
                self.res_net.append(ResNetBlock50(
                    self.filter_size[i][0], self.filter_size[i][1],
                    self.filter_size[i][2], None
                ))
            else:
                self.res_net.append(ResNetBlock50(
                    self.filter_size[i][0], self.filter_size[i][1],
                    self.filter_size[i][2], self.filter_size[i][3]
                ))

        self.zs = nn.ModuleList()
        for i in range(len(self.zs_size)):
            self.zs.append(Embedding(self.zs_size[i], self.z_dim))

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # self.global_avg = nn.AvgPool2d(4)
        self.final = nn.Linear(512 * self.expansion, num_classes)

    def forward(self, x):
        x = self.data_bn(x)
        out = F.relu(self.bn1(self.conv1(x)))
        j = 0
        for i in range(len(self.filter_size)):
            conv1_w = self.zs[j](self.hypernet)
            conv2_w = self.zs[j + 1](self.hypernet)
            conv3_w = self.zs[j + 2](self.hypernet)
            short_w = None
            if len(self.filter_size[i]) == 4:
                short_w = self.zs[j + 3](self.hypernet)
                j += 1
            out = self.res_net[i](out, conv1_w, conv2_w, conv3_w, short_w)
            j += 3

        out = self.avg_pool(out)
        out = self.final(out.squeeze())

        return out


def PrimaryNetwork(depth=18, z_dim=128):
    if depth in [18, 50]:
        return eval(f'PrimaryNetwork{depth}')(z_dim)
    else:
        raise 'Hypernet depth error! (only 18 or 50 are supported)'
