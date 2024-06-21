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
from hyper.primary_net import PrimaryNetwork
from model.ensemble import Ensemble, Red
from utils import logger, MultiAverageMeter, get_loaders, get_opt, \
    lr_schedule, save_ckpt, ensemble_requires_grad
from config import Config, get_args

warnings.filterwarnings("ignore")


def PGD2(args, models, x, y):
    model = Ensemble(models)
    atk = torchattack.PGD(model, eps=args.eps, alpha=args.alpha,
                          steps=args.attack_steps, random_start=True)
    x_adv = atk(x, y)
    return x_adv.detach()


def train_red(args, models, train_loader, test_loader, opt, criterion):
    logger.info("hyper: any parameters same to original TRS && pretrained.")
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
    model = []
    for i in range(args.num_models):
        sub_model = nn.DataParallel(PrimaryNetwork(depth=args.depth, z_dim=128).to(args.device))
        model.append(sub_model)

    # criterion preparing
    criterion = nn.CrossEntropyLoss()

    # optimizer
    opt = get_opt(model, args)

    # training
    train_red(args, model, train_loader, test_loader, opt, criterion)


if __name__ == '__main__':
    arguments = Config(get_args())
    # attack parameter setting
    arguments.alpha = arguments.eps / 4.
    main(arguments)
