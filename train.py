import os
import random
import torch
import logging
import warnings
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from datetime import datetime

import torchattack
from model.resnet_cifar import ResNet18ForCIFAR
from model.ensemble import Ensemble, Red
from utils import logger, MultiAverageMeter, get_loaders, get_opt, \
    lr_schedule, save_ckpt, ensemble_requires_grad
from distillation_dverge import DistillationLoader, Linf_distillation
from config import Config, get_args

warnings.filterwarnings("ignore")


def PGD(args, models, x, y, criterion):
    x_adv = x.detach() + torch.FloatTensor(x.shape).uniform_(-args.eps, args.eps).to(args.device)
    x_adv = torch.clamp(x_adv, 0, 1)
    x_adv.requires_grad = True

    for _ in range(args.attack_steps):
        loss_grad = 0.
        for i, m in enumerate(models):
            loss = criterion(m(x_adv), y)
            grad = autograd.grad(loss, x_adv, create_graph=True)[0]
            grad = grad.flatten(start_dim=1)
            loss_grad += (torch.sum(grad ** 2, 1)).mean() * 2
        loss_grad /= args.num_models
        loss_grad.backward()
        sign_grad = x_adv.grad.data.sign()
        with torch.no_grad():
            x_adv.data = x_adv.data + args.alpha * sign_grad
            x_adv.data = torch.max(torch.min(x_adv.data, x + args.eps), x - args.eps)
            x_adv.data = torch.clamp(x_adv.data, min=0., max=1.)
    x_adv.grad = None
    return x_adv.detach()


def PGD2(args, models, x, y):
    model = Ensemble(models)
    atk = torchattack.PGD(model, eps=args.eps, alpha=args.alpha,
                          steps=10, random_start=True)
    x_adv = atk(x, y)
    return x_adv.detach()


def evaluate_pgd_red(args, models, test_loader):
    for i in range(args.num_models):
        models[i].eval()
    model = Red(models)
    atk = torchattack.PGD(model, eps=0.02, alpha=0.02 / 4., steps=10, random_start=True)
    meter = MultiAverageMeter()
    for step, (X, y) in enumerate(test_loader):
        X, y = X.to(args.device), y.to(args.device)
        X_adv = atk(X, y)
        with torch.no_grad():
            output = model(X_adv)
            loss = F.cross_entropy(output, y)
        meter.update('test_loss', loss.item(), y.size(0))
        # output.max(1)[1] 表示最大值的索引，而output.max(1)[0] 表示最大值的数值，所以这里应该用output.max(1)[1]
        meter.update('test_acc', (output.max(1)[1] == y).float().mean(), y.size(0))

    return meter.avg('test_acc')


def train_red(args, models, train_loader, test_loader, opt, criterion):
    best_loss = float("inf")
    elapsed_time = 0
    for epoch in range(args.start_epoch + 1, args.epochs + 1):
        start_time = time.time()
        train_loss, train_loss_ce, train_loss_sim, train_loss_smooth, train_n = 0, 0, 0, 0, 0
        lr = lr_schedule(epoch, args)
        opt.param_groups[0].update(lr=lr)
        logger.info(f'Epoch {epoch}, lr={lr:.6f}')

        for i in range(args.num_models):
            models[i].train()
            ensemble_requires_grad(models[i], True)

        for step, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            inputs.requires_grad = True
            grads, loss_ce = [], 0
            for j in range(args.num_models):
                logits = models[j](inputs)
                loss = criterion(logits, labels)
                grad = autograd.grad(loss, inputs, create_graph=True)[0]
                grad = grad.flatten(start_dim=1)
                grads.append(grad)
                loss_ce += loss

            loss_sim, loss_smooth, sim_cnt = 0, 0, 0

            for ii in range(len(models)):
                for j in range(ii + 1, len(models)):
                    # loss_sim += torch.abs(F.cosine_similarity(grads[ii], grads[j], eps=1e-06)).mean()
                    loss_sim += torch.abs(F.cosine_similarity(grads[ii], grads[j])).mean()
                    # loss_sim += F.cosine_similarity(grads[ii], grads[j]).mean()
                    sim_cnt += 1.
            loss_sim /= sim_cnt

            N = inputs.shape[0] // 2
            clean_inputs = inputs[:N].detach()
            adv_x = PGD2(args, models, inputs[N:], labels[N:]).detach()
            inputs_adv = torch.cat([clean_inputs, adv_x])
            inputs_adv.requires_grad = True

            if args.plus_adv:
                for j in range(args.num_models):
                    outputs = models[j](inputs_adv)
                    loss = criterion(outputs, labels)
                    grad = autograd.grad(loss, inputs_adv, create_graph=True)[0]
                    grad = grad.flatten(start_dim=1)
                    loss_smooth += (torch.sum(grad ** 2, 1)).mean() * 2
            else:
                # grads = []
                for j in range(args.num_models):
                    outputs = models[j](inputs)
                    loss = criterion(outputs, labels)
                    grad = autograd.grad(loss, inputs, create_graph=True)[0]
                    grad = grad.flatten(start_dim=1)
                    loss_smooth += (torch.sum(grad ** 2, 1)).mean() * 2

            loss_smooth /= args.num_models

            loss = loss_ce + args.lambda_1 * loss_sim + args.lambda_2 * loss_smooth

            train_loss += loss.item() * labels.size(0)
            train_loss_ce += loss_ce.item() * labels.size(0)
            train_loss_sim += loss_sim.item() * labels.size(0)
            train_loss_smooth += loss_smooth.item() * labels.size(0)
            train_n += labels.size(0)

            opt.zero_grad()
            loss.backward()
            if args.grad_clip:
                for i in range(args.num_models):
                    torch.nn.utils.clip_grad_norm_(models[i].parameters(), 1.0)
            opt.step()

            # if (step + 1) % args.log_interval == 0 or (step + 1) == len(train_loader):
            if (step + 1) == len(train_loader):
                logger.info(f"Training epoch {epoch} step {step + 1}/{len(train_loader)}, "
                            f"lr={opt.param_groups[0]['lr']:.4f}, loss_ce={train_loss_ce / train_n:.6f}, "
                            f"loss_sim={train_loss_sim / train_n:.6f}, "
                            f"loss_smooth={train_loss_smooth / train_n:.6f}, "
                            f"loss={train_loss / train_n:.6f}")

        elapsed_time += int(time.time() - start_time)
        if train_loss < best_loss:
            best_loss = train_loss
            saved_name = f'{args.model}{args.depth}_{args.dataset}_{args.train_mode}_' \
                         f'n{args.num_models}_s{args.seed}_la{args.lambda_1}_lb{args.lambda_2}'
            if args.grad_clip:
                saved_name += "_grad_clip"
            save_ckpt(args, train_loss, models, epoch, opt, saved_name, logger)

    hours = elapsed_time // 3600
    minutes = (elapsed_time - hours * 3600) // 60
    seconds = elapsed_time - hours * 3600 - minutes * 60
    logger.info(f'Total Elapsed Time: {hours:02d}:{minutes:02d}:{seconds:02d}')


def main(args):
    # log preparing
    try:
        os.mkdir(args.log_dir)
    except FileExistsError:
        pass
    # if not os.path.exists(args.log_dir):
    #     os.mkdir(args.log_dir)
    logfile = os.path.join(
        args.log_dir,
        f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} {args.model}{args.depth}_{args.dataset}_{args.train_mode}.log')
    file_handler = logging.FileHandler(logfile)
    file_handler.setFormatter(logging.Formatter('%(levelname)-8s %(asctime)-12s %(message)s'))
    logger.addHandler(file_handler)
    logger.info(args.__dict__)

    # data preparing
    args.test_batch_size = 100
    train_loader, test_loader = get_loaders(args)

    # model preparing
    if f'{args.model}{args.depth}' == 'resnet18':
        if args.train_mode == "vanilla":
            model = nn.DataParallel(ResNet18ForCIFAR(pretrained=args.pretrained).to(args.device))
        else:
            model = []
            for i in range(args.num_models):
                sub_model = nn.DataParallel(ResNet18ForCIFAR(pretrained=args.pretrained).to(args.device))
                model.append(sub_model)
    else:
        raise ValueError(args.model)

    # criterion preparing
    criterion = nn.CrossEntropyLoss()

    # training
    opt = get_opt(model, args)
    train_red(args, model, train_loader, test_loader, opt, criterion)


if __name__ == '__main__':
    arguments = Config(get_args())
    # attack parameter setting
    arguments.alpha = arguments.eps / 4.
    main(arguments)
