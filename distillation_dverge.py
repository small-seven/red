import torch
import torch.nn as nn
from packaging import version


class DistillationLoader:
    def __init__(self, source_data, target_data):
        # 保存数据集，而不是迭代器
        self.source_data = source_data
        self.target_data = target_data
        self.source_iter = iter(self.source_data)  # 创建迭代器
        self.target_iter = iter(self.target_data)  # 创建迭代器

    def __len__(self):
        # 假设 source_data 和 target_data 长度相同，返回其中一个的长度
        return len(self.source_data)

    def __iter__(self):
        # 返回当前实例本身，使其可以被迭代
        return self

    def __next__(self):
        try:  # 尝试从两个数据加载器中获取下一个批次的数据
            si, sl = next(self.source_iter)
            ti, tl = next(self.target_iter)
            return si, sl, ti, tl
        except StopIteration:
            # print("StopIteration: One of the iterators is exhausted.")
            self.reset()
            raise StopIteration

    def reset(self):
        # 重置迭代器，以便可以重新开始迭代
        self.source_iter = iter(self.source_data)
        self.target_iter = iter(self.target_data)


class Flatten(nn.Module):
    def forward(self, x):
        # 假设我们想要将除了第一个维度（批次大小）之外的所有维度展平
        return x.view(x.size(0), -1)


def gradient_wrt_input(model, inputs, targets, criterion=nn.CrossEntropyLoss()):
    inputs.requires_grad = True

    outputs = model(inputs)
    loss = criterion(outputs, targets)
    model.zero_grad()
    loss.backward()

    data_grad = inputs.grad.data
    return data_grad.clone().detach()


def gradient_wrt_feature(model, source_data, target_data, layer, before_relu, criterion=nn.MSELoss()):
    source_data.requires_grad = True

    out = model.module[1].get_features(x=model.module[0](source_data), layer=layer, before_relu=before_relu)

    target = model.module[1].get_features(x=model.module[0](target_data), layer=layer,
                                          before_relu=before_relu).data.clone().detach()

    loss = criterion(out, target)
    model.zero_grad()
    loss.backward()

    data_grad = source_data.grad.data
    return data_grad.clone().detach()


def get_module(partial_model, module, layer, min_layer, max_layer):
    mid_layer = min_layer + 2
    if min_layer <= layer < mid_layer:
        # layer == min_layer
        partial_model.append(module[0].conv1)
        partial_model.append(module[0].bn1)
        if layer == min_layer:
            return partial_model
        partial_model.append(module[0].relu)
        # layer == mid_layer - 1
        partial_model.append(module[0].conv2)
        partial_model.append(module[0].bn2)

        if layer == mid_layer - 1:
            return partial_model
    if mid_layer <= layer < max_layer:

        partial_model.append(module[0])
        # layer == mid_layer
        partial_model.append(module[1].conv1)
        partial_model.append(module[1].bn1)
        if layer == mid_layer:
            return partial_model
        partial_model.append(module[1].relu)
        # layer == max_layer - 1
        partial_model.append(module[1].conv2)
        partial_model.append(module[1].bn2)
        if layer == max_layer - 1:
            return partial_model


def get_layer(model, layer, partial_model):
    if 2 <= layer < 6:
        return get_module(partial_model, model.layer1, layer, 2, 6)
    partial_model.append(model.layer1)
    if 6 <= layer < 10:
        return get_module(partial_model, model.layer2, layer, 6, 10)
    partial_model.append(model.layer2)
    if 10 <= layer < 14:
        return get_module(partial_model, model.layer3, layer, 10, 14)
    partial_model.append(model.layer3)
    if 14 <= layer < 18:
        return get_module(partial_model, model.layer4, layer, 14, 18)
    partial_model.append(model.layer4)
    if layer == 18:
        partial_model.append(model.avgpool)
        partial_model.append(Flatten())
        partial_model.append(model.fc)
        return partial_model


def get_partial_model(models, layer):
    model = models.module
    base_model = model.base_model
    partial_model = torch.nn.Sequential(
        model.data_norm,
        base_model.conv1,
        base_model.bn1
    )
    if layer == 1:
        return partial_model
    partial_model.append(base_model.relu)
    partial_model.append(base_model.maxpool)

    if 2 <= layer <= 18:
        return get_layer(base_model, layer, partial_model)


def gradient_wrt_feature2(model, si, ti, layer, before_relu, criterion=nn.MSELoss()):
    si.requires_grad = True

    partial_model = get_partial_model(model, layer)
    partial_model.eval()
    so, to = partial_model(si), partial_model(ti).data.clone().detach()

    loss = criterion(so, to)
    partial_model.zero_grad()
    loss.backward()

    data_grad = si.grad.data
    return data_grad.clone().detach()


def Linf_PGD(model, dat, lbl, eps, alpha, steps, is_targeted=False, rand_start=True, momentum=False, mu=1,
             criterion=nn.CrossEntropyLoss()):
    x_nat = dat.clone().detach()
    x_adv = None
    if rand_start:
        x_adv = dat.clone().detach() + torch.FloatTensor(dat.shape).uniform_(-eps, eps).cuda()
    else:
        x_adv = dat.clone().detach()
    x_adv = torch.clamp(x_adv, 0., 1.)  # respect image bounds
    g = torch.zeros_like(x_adv)

    # Iteratively Perturb data
    for i in range(steps):
        # Calculate gradient w.r.t. data
        grad = gradient_wrt_input(model, x_adv, lbl, criterion)
        with torch.no_grad():
            if momentum:
                # Compute sample wise L1 norm of gradient
                flat_grad = grad.view(grad.shape[0], -1)
                l1_grad = torch.norm(flat_grad, 1, dim=1)
                grad = grad / torch.clamp(l1_grad, min=1e-12).view(grad.shape[0], 1, 1, 1)
                # Accumulate the gradient
                new_grad = mu * g + grad  # calc new grad with momentum term
                g = new_grad
            else:
                new_grad = grad
            # Get the sign of the gradient
            sign_data_grad = new_grad.sign()
            if is_targeted:
                x_adv = x_adv - alpha * sign_data_grad  # perturb the data to MINIMIZE loss on tgt class
            else:
                x_adv = x_adv + alpha * sign_data_grad  # perturb the data to MAXIMIZE loss on gt class
            # Clip the perturbations w.r.t. the original data so we still satisfy l_infinity
            # x_adv = torch.clamp(x_adv, x_nat-eps, x_nat+eps) # Tensor min/max not supported yet
            x_adv = torch.max(torch.min(x_adv, x_nat + eps), x_nat - eps)
            # Make sure we are still in bounds
            x_adv = torch.clamp(x_adv, 0., 1.)
    return x_adv.clone().detach()


def Linf_distillation(model, si, ti, eps, alpha, steps, layer, before_relu=True, mu=1, momentum=True,
                      rand_start=False):
    x_nat = si.clone().detach()
    if rand_start:
        x_adv = si.clone().detach() + torch.FloatTensor(si.shape).uniform_(-eps, eps).cuda()
    else:
        x_adv = si.clone().detach()
    x_adv = torch.clamp(x_adv, 0., 1.)  # respect image bounds
    g = torch.zeros_like(x_adv)

    # Iteratively Perturb data
    for i in range(steps):
        # Calculate gradient w.r.t. data
        grad = gradient_wrt_feature2(model, x_adv, ti, layer, before_relu)
        with torch.no_grad():
            if momentum:
                # Compute sample wise L1 norm of gradient
                flat_grad = grad.view(grad.shape[0], -1)
                l1_grad = torch.norm(flat_grad, 1, dim=1)
                grad = grad / torch.clamp(l1_grad, min=1e-12).view(grad.shape[0], 1, 1, 1)
                # Accumulate the gradient
                new_grad = mu * g + grad  # calc new grad with momentum term
                g = new_grad
            else:
                new_grad = grad
            x_adv = x_adv - alpha * new_grad.sign()  # perturb the data to MINIMIZE loss on tgt class
            # Clip the perturbations w.r.t. the original data so we still satisfy l_infinity
            # x_adv = torch.clamp(x_adv, x_nat-eps, x_nat+eps) # Tensor min/max not supported yet
            x_adv = torch.max(torch.min(x_adv, x_nat + eps), x_nat - eps)
            # Make sure we are still in bounds
            x_adv = torch.clamp(x_adv, 0., 1.)
    return x_adv.clone().detach()
