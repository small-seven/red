import torch
import random
import torch.nn as nn
import torch.nn.functional as F


class Ensemble(nn.Module):
    def __init__(self, models):
        super(Ensemble, self).__init__()
        self.models = models
        assert len(self.models) > 0
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def forward(self, x):
        if len(self.models) > 1:
            outputs = 0
            for model in self.models:
                outputs += F.softmax(model(x), dim=-1)
                # print(F.softmax(model(x), dim=-1))
                # exit(0)
            output = outputs / len(self.models)
            output = torch.clamp(output, min=1e-40)
            return torch.log(output)
        else:
            return self.models[0](x)


class Red(nn.Module):
    def __init__(self, models):
        super(Red, self).__init__()
        self.models = models
        assert len(self.models) > 0
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def forward(self, x):
        if len(self.models) > 1:
            model = self.models[random.randint(0, len(self.models) - 1)]
            return model(x)
        else:
            return self.models[0](x)
