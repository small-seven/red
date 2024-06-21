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


def evaluate_pgd_ens(args, models, test_loader):
    for i in range(args.num_models):
        models[i].eval()
    model = Ensemble(models)
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


def train_vanilla(args, model, train_loader, opt, criterion):
    best_loss = float("inf")
    for epoch in range(args.start_epoch + 1, args.epochs + 1):
        train_loss, train_acc, train_n = 0, 0, 0
        lr = lr_schedule(epoch, args)
        opt.param_groups[0].update(lr=lr)
        logger.info(f'Epoch {epoch}, lr={lr:.6f}')

        # PGD
        def train_step(X, y):
            model.eval()
            atk = torchattack.PGD(model, eps=args.eps, alpha=args.alpha,
                                  steps=args.attack_steps, random_start=True)
            X_adv = atk(X, y)
            model.train()
            output = model(X_adv)
            loss_fun = criterion(output, y)

            loss_fun.backward()
            accuracy = (output.max(1)[1] == y).float().mean()

            return loss_fun, accuracy

        for step, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            loss, acc = train_step(inputs, labels)

            train_loss += loss.item() * labels.size(0)
            train_acc += acc.item() * labels.size(0)
            train_n += labels.size(0)

            # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            opt.step()
            opt.zero_grad()

            # if (step + 1) % args.log_interval == 0 or (step + 1) == len(train_loader):
            if (step + 1) == len(train_loader):
                logger.info(f"Training epoch {epoch} step {step + 1}/{len(train_loader)}, "
                            f"lr={opt.param_groups[0]['lr']:.4f}, "
                            f"loss={train_loss / train_n:.6f}, "
                            f"acc={train_acc / train_n:.4f}")

        if train_loss < best_loss:
            best_loss = train_loss
            saved_name = f'{args.model}{args.depth}_{args.dataset}_{args.train_mode}_' \
                         f'b{args.batch_size}'
            save_ckpt(args, train_loss, model, epoch, opt, saved_name, logger)


def train_gal(args, models, train_loader, test_loader, opt, criterion):
    best_loss = float("inf")
    elapsed_time = 0
    for epoch in range(args.start_epoch + 1, args.epochs + 1):
        start_time = time.time()
        train_loss, train_loss_ce, train_loss_gal, train_n = 0, 0, 0, 0
        lr = lr_schedule(epoch, args)
        opt.param_groups[0].update(lr=lr)
        logger.info(f'Epoch {epoch}/{args.epochs}, lr={lr:.6f}')

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

            cos_sim = []
            for ii in range(len(models)):
                for j in range(ii + 1, len(models)):
                    cos_sim.append(F.cosine_similarity(grads[ii], grads[j], dim=-1))
            cos_sim = torch.stack(cos_sim, dim=-1)
            loss_gal = torch.log(cos_sim.exp().sum(dim=-1) + 1e-20).mean()

            loss = loss_ce + args.lambda_1 * loss_gal

            train_loss += loss.item() * labels.size(0)
            train_loss_ce += loss_ce.item() * labels.size(0)
            train_loss_gal += loss_gal.item() * labels.size(0)
            train_n += labels.size(0)

            opt.zero_grad()
            loss.backward()
            opt.step()

            # if (step + 1) % args.log_interval == 0 or (step + 1) == len(train_loader):
            if (step + 1) == len(train_loader):
                logger.info(f"Training epoch {epoch} step {step + 1}/{len(train_loader)}, "
                            f"lr={opt.param_groups[0]['lr']:.4f}, "
                            f"loss_ce={train_loss_ce / train_n:.6f}, "
                            f"loss_gal={train_loss_gal / train_n:.6f}, "
                            f"loss={train_loss / train_n:.6f}")

        elapsed_time += int(time.time() - start_time)
        if train_loss < best_loss:
            best_loss = train_loss
            saved_name = f'{args.model}{args.depth}_{args.dataset}_{args.train_mode}_' \
                         f'n{args.num_models}_s{args.seed}_la{args.lambda_1}'
            save_ckpt(args, train_loss, models, epoch, opt, saved_name, logger)

    hours = elapsed_time // 3600
    minutes = (elapsed_time - hours * 3600) // 60
    seconds = elapsed_time - hours * 3600 - minutes * 60
    logger.info(f'Total Elapsed Time: {hours:02d}:{minutes:02d}:{seconds:02d}')


def train_adp(args, models, train_loader, test_loader, opt, criterion):
    best_loss = float("inf")
    elapsed_time = 0
    for epoch in range(args.start_epoch + 1, args.epochs + 1):
        start_time = time.time()
        train_loss, train_loss_entropy, train_loss_ce, train_loss_det, train_n = 0, 0, 0, 0, 0
        lr = lr_schedule(epoch, args)
        opt.param_groups[0].update(lr=lr)
        logger.info(f'Epoch {epoch}, lr={lr:.6f}')

        for i in range(args.num_models):
            models[i].train()
            ensemble_requires_grad(models[i], True)

        for step, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(args.device), labels.to(args.device)

            inputs.requires_grad = True

            y_true = torch.zeros(inputs.size(0), args.num_classes).to(args.device)
            y_true.scatter_(1, labels.view(-1, 1), 1)

            loss_ce = 0
            mask_non_y_pred = []
            ensemble_probs = 0

            for j in range(args.num_models):
                outputs = models[j](inputs)
                loss_ce += criterion(outputs, labels)

                # for log_det
                y_pred = F.softmax(outputs, dim=-1)
                # batch_size X (num_class X num_models), 2-D
                bool_R_y_true = torch.eq(
                    torch.ones_like(y_true) - y_true, torch.ones_like(y_true)
                )
                # batch_size X (num_class-1) X num_models, 1-D
                mask_non_y_pred.append(
                    torch.masked_select(y_pred, bool_R_y_true).reshape(-1, args.num_classes - 1)
                )

                # for ensemble entropy
                ensemble_probs += y_pred

            ensemble_probs = ensemble_probs / len(models)
            ensemble_entropy = torch.sum(
                -torch.mul(ensemble_probs, torch.log(torch.Tensor(ensemble_probs + 1e-20).to(args.device))),
                dim=-1
            ).mean()

            mask_non_y_pred = torch.stack(mask_non_y_pred, dim=1)
            assert mask_non_y_pred.shape == (inputs.size(0), len(models), args.num_classes - 1)
            mask_non_y_pred = mask_non_y_pred / torch.norm(mask_non_y_pred, p=2, dim=-1,
                                                           keepdim=True)  # batch_size X num_model X (num_class-1), 3-D
            matrix = torch.matmul(
                mask_non_y_pred, mask_non_y_pred.permute(0, 2, 1)
            )  # batch_size X num_model X num_model, 3-D
            log_det = torch.logdet(
                matrix + 1e-6 * torch.eye(len(models), device=matrix.device).unsqueeze(0)
            ).mean()  # batch_size X 1, 1-D
            scaling = 0.1  # without scaling, get loss=nan
            loss = scaling * loss_ce - scaling * args.lambda_1 * ensemble_entropy - scaling * args.lambda_2 * log_det

            train_loss += loss.item() * labels.size(0)
            train_loss_ce += loss_ce.item() * labels.size(0)
            train_loss_entropy += ensemble_entropy.item() * labels.size(0)
            train_loss_det += log_det.item() * labels.size(0)
            train_n += labels.size(0)

            opt.zero_grad()
            loss.backward()
            opt.step()

            # if (step + 1) % args.log_interval == 0 or (step + 1) == len(train_loader):
            if (step + 1) == len(train_loader):
                logger.info(f"Training epoch {epoch} step {step + 1}/{len(train_loader)}, "
                            f"lr={opt.param_groups[0]['lr']:.4f}, "
                            f"loss_ce={train_loss_ce / train_n:.6f}, "
                            f"loss_entropy={train_loss_entropy / train_n:.6f}, "
                            f"loss_det={train_loss_det / train_n:.6f},"
                            f"loss={train_loss / train_n:.6f}")

        elapsed_time += int(time.time() - start_time)
        if train_loss < best_loss:
            best_loss = train_loss
            saved_name = f'{args.model}{args.depth}_{args.dataset}_{args.train_mode}_' \
                         f'n{args.num_models}_s{args.seed}_la{args.lambda_1}_lb{args.lambda_2}'
            save_ckpt(args, train_loss, models, epoch, opt, saved_name, logger)

    hours = elapsed_time // 3600
    minutes = (elapsed_time - hours * 3600) // 60
    seconds = elapsed_time - hours * 3600 - minutes * 60
    logger.info(f'Total Elapsed Time: {hours:02d}:{minutes:02d}:{seconds:02d}')


def train_dverge(args, models, train_loader, criterion):
    train_loader = DistillationLoader(train_loader, train_loader)
    distill_cfg = {'eps': args.distill_eps,
                   'alpha': args.distill_alpha,
                   'steps': args.distill_steps,
                   'layer': args.distill_layer,
                   'rand_start': args.distill_rand_start,
                   'before_relu': True,
                   'momentum': args.distill_momentum
                   }

    opts = []
    for i in range(args.num_models):
        opt = get_opt(models[i], args)
        opts.append(opt)

    best_loss = float("inf")
    elapsed_time = 0
    for epoch in range(args.start_epoch + 1, args.epochs + 1):
        start_time = time.time()
        train_losses, train_ns = [], []
        lr = lr_schedule(epoch, args)
        for opt in opts:
            opt.param_groups[0].update(lr=lr)
        if args.distill_layer is None:
            distill_cfg['layer'] = random.randint(1, args.depth)
        for i in range(args.num_models):
            models[i].train()
            ensemble_requires_grad(models[i], True)
            train_losses.append(0)
            train_ns.append(0)
        for step, (si, sl, ti, tl) in enumerate(train_loader):
            # logger.info(f"{step}")
            si, sl = si.to(args.device), sl.to(args.device)
            ti, tl = ti.to(args.device), tl.to(args.device)

            distilled_data_list = []
            for i in range(args.num_models):
                distilled_data_list.append(Linf_distillation(models[i], si, ti, **distill_cfg))

            for j in range(args.num_models):
                loss = 0
                for k, distilled_data in enumerate(distilled_data_list):
                    if j == k: continue
                    outputs = models[j](distilled_data)
                    loss += criterion(outputs, sl)

                train_losses[j] += loss.item() * tl.size(0)
                train_ns[j] += tl.size(0)

                # # loss 太大了，手动调小一点
                # loss *= 0.1

                opts[j].zero_grad()
                loss.backward()
                # loss为NaN，试试把梯度截取
                torch.nn.utils.clip_grad_norm_(models[j].parameters(), 1.0)
                opts[j].step()

            # if (step + 1) % args.log_interval == 0 or (step + 1) == len(train_loader):
            if (step + 1) == len(train_loader):
                info_msg = f"E {epoch} {step + 1}/{len(train_loader)} lr {opts[0].param_groups[0]['lr']:.4f}, "
                total_loss = 0
                for i in range(args.num_models):
                    info_msg += f"sub {i}: loss={train_losses[i] / train_ns[i]:.4f}, "
                    total_loss += train_losses[i]
                info_msg += f"total={total_loss / train_ns[0]:.4f}"
                logger.info(info_msg)

        train_loss = 0
        for i in range(args.num_models):
            train_loss += train_losses[i]

        elapsed_time += int(time.time() - start_time)
        if train_loss < best_loss:
            best_loss = train_loss
            saved_name = f'{args.model}{args.depth}_{args.dataset}_{args.train_mode}_' \
                         f'n{args.num_models}_s{args.seed}'
            save_ckpt(args, train_loss, models, epoch, opts, saved_name, logger)

    hours = elapsed_time // 3600
    minutes = (elapsed_time - hours * 3600) // 60
    seconds = elapsed_time - hours * 3600 - minutes * 60
    logger.info(f'Total Elapsed Time: {hours:02d}:{minutes:02d}:{seconds:02d}')


def train_trs(args, models, train_loader, test_loader, opt, criterion):
    logger.info("trs: any parameters same to original TRS && no pretrained.")
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
                    loss_sim += torch.abs(F.cosine_similarity(grads[ii], grads[j])).mean()
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
            save_ckpt(args, train_loss, models, epoch, opt, saved_name, logger)

    hours = elapsed_time // 3600
    minutes = (elapsed_time - hours * 3600) // 60
    seconds = elapsed_time - hours * 3600 - minutes * 60
    logger.info(f'Total Elapsed Time: {hours:02d}:{minutes:02d}:{seconds:02d}')


def train_red(args, models, train_loader, test_loader, opt, criterion):
    logger.info("red: parameters same to TRS && pretrained && grad_clip.")
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
    if args.train_mode in ['red', 'vanilla', 'adp', 'gal', 'trs']:
        # optimizer preparing
        opt = get_opt(model, args)
        eval(f"train_{args.train_mode}")(args, model, train_loader, test_loader, opt, criterion)
    elif args.train_mode == 'dverge':
        train_dverge(args, model, train_loader, criterion)
    else:
        raise ValueError(args.train_mode)


if __name__ == '__main__':
    arguments = Config(get_args())
    # attack parameter setting
    arguments.alpha = arguments.eps / 4.
    main(arguments)
