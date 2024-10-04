import os
import time
import torch
import logging
import warnings
import torch.nn.functional as F
import torch.nn as nn
# torch.cuda.current_device() # 本地：不加这句会报RuntimeError: CUDA unknown error错误

from datetime import datetime

import torchattack
from utils import get_loaders, MultiAverageMeter, logger
from model.resnet_cifar import ResNet18ForCIFAR
from hyper.primary_net import PrimaryNetwork
from model.ensemble import Ensemble, Red
from config import get_args, Config

warnings.filterwarnings("ignore")


def equipped_atk_func(model, test_loader, atk_func, atk_name, log_str):
    acc = 1.9999
    try:
        acc = evaluate_attack(model, test_loader, atk_func, atk_name)
    except torch.cuda.OutOfMemoryError:
        logger.warning(f'{atk_name} Out Of Memory!')
    if log_str == "":
        log_str += f"{acc * 100:.2f}"
    else:
        log_str += f"\t{acc * 100:.2f}"
    return log_str


def evaluate_attack(model, test_loader, atk, atk_name):
    model.eval()
    meter = MultiAverageMeter()
    start_time = time.time()
    for i, (X, y) in enumerate(test_loader):
        X, y = X.to('cuda'), y.to('cuda')

        X_adv = atk(X, y)  # torchattack

        with torch.no_grad():
            output = model(X_adv)
            loss = F.cross_entropy(output, y)
            meter.update('test_loss', loss.item(), y.size(0))
            meter.update('test_acc', (output.max(1)[1] == y).float().mean(), y.size(0))

    elapsed_time = int(time.time() - start_time)
    hours = elapsed_time // 3600
    minutes = (elapsed_time - 3600 * hours) // 60
    seconds = elapsed_time - 3600 * hours - 60 * minutes
    logger.info(f'Attack_type: [{atk_name}],\t'
                f'loss:{float(meter.avg("test_loss")): .4f},\t'
                f'acc: {float(meter.avg("test_acc")) * 100: .2f},\t'
                f'Elapsed Time: {hours:02d}:{minutes:02d}:{seconds:02d}')
    return float(meter.avg("test_acc"))


def main(args):
    # log preparing
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    logfile = os.path.join(
        args.log_dir,
        f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} {args.model}{args.depth}_{args.dataset}_{args.train_mode}_eval.log')
    file_handler = logging.FileHandler(logfile)
    file_handler.setFormatter(logging.Formatter('%(levelname)-8s %(asctime)-12s %(message)s'))
    logger.addHandler(file_handler)
    logger.info(args.__dict__)
    logger.info("2024-05-21: new test setting!")

    # if args.dataset == "TinyImageNet":
    #     raise ValueError(args.attack_type)

    # data preparing
    args.test_batch_size = 100
    test_loader = get_loaders(args)

    # model preparing & load the checkpoint
    load_name = f"{args.model}{args.depth}_{args.dataset}"
    # args.load_dir = os.path.join(args.load_dir, f"{args.dataset}_hpc")
    # args.load_dir = f'{args.load_dir}/{args.dataset}'
    logger.info(args.load_dir)
    if f'{args.model}{args.depth}' == 'resnet18':

        load_name += f"_{args.train_mode}_n{args.num_models}_s{args.seed}_la{args.lambda_1}_lb{args.lambda_2}"
        if args.train_mode == "hyper":
            load_name += f"_z{args.embedding}"
        if args.at:
            load_name += f"_at"
        model = []
        for i in range(args.num_models):
            if args.train_mode == 'hyper':
                sub_model = PrimaryNetwork(depth=args.depth, z_dim=args.embedding)
            else:
                sub_model = ResNet18ForCIFAR(pretrained=False)
            if args.grad_clip and (args.train_mode == 'red'):
                loadName = f"{load_name}_grad_clip_{i}.pth"
            else:
                loadName = f"{load_name}_{i}.pth"
            logger.info(loadName)
            ckpt = torch.load(os.path.join(args.load_dir, loadName))
            sub_model.load_state_dict(ckpt['model'])
            sub_model = torch.nn.DataParallel(sub_model).to(args.device)
            sub_model.eval()
            model.append(sub_model)
        if args.red_test:
            model = Red(model)
        else:
            model = Ensemble(model)
    else:
        raise ValueError(args.model)

    if not args.eval:
        raise ValueError(args.eval)

    print_srt = ""
    # if args.attack_type not in ['nat', 'fgsm', 'mifgsm', 'pgd', 'deepfool', 'cwl2', 'autoattack', 'all']:
    #     raise ValueError(args.attack_type)
    if args.attack_type in ['nat', 'all', 'white_part']:
        atk = torchattack.VANILA(model)
        print_srt = equipped_atk_func(model, test_loader, atk, 'vanilla', print_srt)
    if args.attack_type in ['fgsm', 'all', 'white_part']:
        atk = torchattack.FGSM(model, eps=args.eps)
        print_srt = equipped_atk_func(model, test_loader, atk, 'fgsm', print_srt)
    if args.attack_type in ['mifgsm', 'all', 'white_part']:
        atk = torchattack.MIFGSM(model, eps=args.eps, alpha=args.eps / 4., steps=10, decay=1.0)
        print_srt = equipped_atk_func(model, test_loader, atk, 'mifgsm', print_srt)
    if args.attack_type in ['bim', 'all']:
        atk = torchattack.BIM(model, eps=args.eps, alpha=args.eps / 4., steps=10)
        print_srt = equipped_atk_func(model, test_loader, atk, 'bim', print_srt)
    if args.attack_type in ['pgd', 'all', 'white_part']:
        atk = torchattack.PGD(model, eps=args.eps, alpha=args.eps / 4., steps=10, random_start=True)
        print_srt = equipped_atk_func(model, test_loader, atk, 'pgd', print_srt)
    if args.attack_type in ['deepfool', "all"]:
        atk = torchattack.DeepFool(model, steps=50, overshoot=0.02)
        print_srt = equipped_atk_func(model, test_loader, atk, 'deepfool', print_srt)
    if args.attack_type in ['cwl2', 'all']:
        atk = torchattack.CW(model, c=0.1, kappa=0, steps=50, lr=0.01)
        print_srt = equipped_atk_func(model, test_loader, atk, 'cwl2', print_srt)
    if args.attack_type in ['autoattack', 'all', 'white_part']:
        atk = torchattack.AutoAttack(model, norm='Linf', eps=args.eps, version='standard',
                                     n_classes=args.num_classes, n_queries=200)
        print_srt = equipped_atk_func(model, test_loader, atk, 'autoattack', print_srt)

    # black and other attacks
    if args.attack_type in ['black_all', 'onepixel']:
        atk = torchattack.OnePixel(model, pixels=1, steps=10, popsize=10, inf_batch=128)
        print_srt = equipped_atk_func(model, test_loader, atk, 'onepixel', print_srt)
    if args.attack_type in ['black_all', 'square']:
        atk = torchattack.Square(model, norm='Linf', eps=args.eps, n_queries=200, n_restarts=1,
                                 p_init=.8, loss='margin', resc_schedule=True, seed=0, verbose=False)
        print_srt = equipped_atk_func(model, test_loader, atk, 'square', print_srt)
    if args.attack_type in ['black_all', 'pixle']:
        atk = torchattack.Pixle(model, x_dimensions=(0.1, 0.2), restarts=10, max_iterations=50)
        print_srt = equipped_atk_func(model, test_loader, atk, 'pixle', print_srt)
    if args.attack_type in ['black_all', 'di2fgsm']:
        atk = torchattack.DIFGSM(model, eps=args.eps, alpha=args.eps / 4., steps=10, decay=0.0,
                                 resize_rate=0.9, diversity_prob=0.5, random_start=False)
        print_srt = equipped_atk_func(model, test_loader, atk, 'di2fgsm', print_srt)
    if args.attack_type in ['black_all', 'eotpgd']:
        atk = torchattack.EOTPGD(model, eps=args.eps, alpha=args.eps / 4., steps=10, eot_iter=2)
        print_srt = equipped_atk_func(model, test_loader, atk, 'eotpgd', print_srt)
    if args.attack_type in ['black_all', 'sparsefool']:
        logger.info(f'sparsefool: bs={args.test_batch_size}')
        atk = torchattack.SparseFool(model, steps=10, lam=3, overshoot=0.02)
        print_srt = equipped_atk_func(model, test_loader, atk, 'sparsefool', print_srt)
    if args.attack_type in ['black_all', 'apgd']:
        atk = torchattack.APGD(model, norm='Linf', eps=args.eps, steps=10, n_restarts=1, seed=0, loss='ce', eot_iter=1,
                               rho=.75, verbose=False)
        print_srt = equipped_atk_func(model, test_loader, atk, 'apgd', print_srt)

    print(print_srt)


if __name__ == "__main__":
    arguments = Config(get_args())
    main(arguments)
