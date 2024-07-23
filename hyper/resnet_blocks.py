import torch.nn as nn
import torch.nn.functional as F


class ResNetBlock18(nn.Module):

    def __init__(self, short_layer=None, layer1=(64, 64), layer2=(64, 64)):
        super(ResNetBlock18, self).__init__()
        self.short_layer = short_layer
        self.layer1 = layer1
        self.layer2 = layer2
        self.identity = nn.Sequential()

        if short_layer is not None:
            self.bn_short = nn.BatchNorm2d(short_layer[0])
        else:
            self.bn_short = None

        self.bn1 = nn.BatchNorm2d(layer1[0])
        self.bn2 = nn.BatchNorm2d(layer2[0])

    def forward(self, x, short_w=None, conv1_w=None, conv2_w=None):
        # torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
        if short_w is not None:
            short_w = short_w.mean(dim=(2, 3), keepdim=True)
            residual = self.bn_short(F.conv2d(x, short_w, stride=2, padding=0))

            out = F.relu(self.bn1(F.conv2d(x, conv1_w, stride=2, padding=1)), inplace=True)
        else:
            residual = self.identity(x)

            out = F.relu(self.bn1(F.conv2d(x, conv1_w, stride=1, padding=1)), inplace=True)

        out = self.bn2(F.conv2d(out, conv2_w, stride=1, padding=1))

        out += residual
        out = F.relu(out)

        return out


class ResNetBlock50(nn.Module):

    def __init__(self, layer1=(64, 64), layer2=(64, 64), layer3=(64, 64), short_layer=None):
        super(ResNetBlock50, self).__init__()
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.short_layer = short_layer
        self.identity = nn.Sequential()
        if short_layer is not None:
            self.bn_short = nn.BatchNorm2d(short_layer[0])
        else:
            self.bn_short = None

        self.bn1 = nn.BatchNorm2d(layer1[0])
        self.bn2 = nn.BatchNorm2d(layer2[0])
        self.bn3 = nn.BatchNorm2d(layer3[0])

    def forward(self, x, conv1_w=None, conv2_w=None, conv3_w=None, short_w=None):
        # torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
        # nn.Conv2d 是2D卷积层，而 F.conv2d 是2D卷积操作
        conv1_w = conv1_w.mean(dim=(2, 3), keepdim=True)
        out = F.relu(self.bn1(F.conv2d(x, conv1_w, stride=1, padding=0)))
        if short_w is not None:
            short_w = short_w.mean(dim=(2, 3), keepdim=True)
            if self.short_layer[1] == 64:
                residual = self.bn_short(F.conv2d(x, short_w, stride=1, padding=0))
                out = F.relu(self.bn2(F.conv2d(out, conv2_w, stride=1, padding=1)), inplace=True)
            else:
                residual = self.bn_short(F.conv2d(x, short_w, stride=2, padding=0))

                out = F.relu(self.bn2(F.conv2d(out, conv2_w, stride=2, padding=1)), inplace=True)
        else:
            residual = self.identity(x)

            out = F.relu(self.bn2(F.conv2d(out, conv2_w, stride=1, padding=1)), inplace=True)

        conv3_w = conv3_w.mean(dim=(2, 3), keepdim=True)
        out = self.bn3(F.conv2d(out, conv3_w, stride=1, padding=0))

        out += residual
        out = F.relu(out, inplace=True)

        return out
