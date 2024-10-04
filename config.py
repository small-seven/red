import os
import torch
import argparse
import random
import numpy as np
import torch.backends.cudnn as cudnn


def simple_fix_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def fix_random_seed(seed):
    random.seed(seed)  # random
    np.random.seed(seed)  # numpy
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)  # torch cpu
    torch.cuda.manual_seed(seed)  # torch gpu
    torch.cuda.manual_seed_all(seed)  # torch all gpus
    cudnn.benchmark = False  #
    cudnn.deterministic = True


cfg_datasets = {
    'CIFAR10': dict(
        data_mean=(0.4914, 0.4822, 0.4465),
        data_std=(0.2471, 0.2435, 0.2616),
        num_classes=10,
    ),
    'SVHN': dict(
        data_mean=(0.4377, 0.4438, 0.4728),
        data_std=(0.1980, 0.2010, 0.1970),
        num_classes=10,
    ),
    'TinyImageNet': dict(
        data_mean=(0.485, 0.456, 0.406),
        data_std=(0.229, 0.224, 0.225),
        num_classes=200,
    ),
    'CIFAR100': dict(
        data_mean=(0.5071, 0.4867, 0.4408),
        data_std=(0.2675, 0.2565, 0.2761),
        num_classes=100,
    ),
    'ImageNet': dict(
        data_mean=(0.485, 0.456, 0.406),
        data_std=(0.229, 0.224, 0.225),
        num_classes=1000,
    ),
    'ImageNette': dict(
        data_mean=(0.485, 0.456, 0.406),
        data_std=(0.229, 0.224, 0.225),
        num_classes=10,
    ),

}


class Config(object):
    # random seed
    seed = 123

    # model setting
    model = 'resnet'
    depth = 18
    proj_dir = '/home/hhgong/code/vit_rar'

    # data setting
    dataset = 'CIFAR10'
    data_dir = '../data'

    # log setting
    log_dir = './logs'
    log_interval = 100
    saved_dir = './checkpoint'

    # training setting
    train_mode = 'red'  # choices=['vanilla', 'gal', 'adp', 'dverge', 'trs', 'red']
    num_models = 3
    train_batch_size = 256
    epochs = 120
    start_epoch = 0
    lambda_1 = 0.0
    lambda_2 = 0.0
    pretrained = False
    plus_adv = False  # for TRS and RED

    # opt setting
    lr_max = 0.1
    weight_decay = 1e-4
    momentum = 0.9
    grad_clip = False

    # attack setting
    eps = 0.03
    attack_steps = 7

    # test
    load_dir = "./checkpoint"
    attack_type = "all"  # choices=['nat', 'fgsm', 'mifgsm', 'pgd', 'deepfool', 'cwl2', 'autoattack', 'all']
    eval = False
    red_test = False

    # gpu
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    mul_gpus = False
    num_workers = 8
    pin_memory = False

    def __init__(self, args=None):
        if args is not None:
            names = self.__dict__
            for arg in vars(args):
                if arg == 'device':
                    names[arg] = f'cuda:{getattr(args, arg)}'
                else:
                    names[arg] = getattr(args, arg)

        # dataset setting
        self.data_mean = cfg_datasets[self.dataset]['data_mean']
        self.data_std = cfg_datasets[self.dataset]['data_std']
        self.num_classes = cfg_datasets[self.dataset]['num_classes']

        self.alpha = self.eps / 4.

        simple_fix_random_seed(args.seed)


def get_args():
    parser = argparse.ArgumentParser()
    # random seed
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    # model
    parser.add_argument('--model', type=str, default='resnet')
    parser.add_argument('--depth', type=int, default=18)
    parser.add_argument('--num_models', default=3, type=int)
    parser.add_argument('--proj_dir', type=str, default='/home/code/vit_rar')
    parser.add_argument('--continuing', action='store_true', help="plus adv training")
    # dataset
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        choices=['CIFAR10', 'CIFAR100', 'SVHN', 'TinyImageNet'])
    # training
    parser.add_argument('--train_mode', type=str, default='vanilla',
                        choices=['red', 'vanilla', 'adp', 'gal', 'dverge', 'trs', 'hyper'])
    parser.add_argument('--train_batch_size', type=int, default=256)  # choices=[64, 128, 256, 512]
    parser.add_argument('--epochs', default=120, type=int)
    parser.add_argument('--pretrained', action='store_true', help="pretrained or not")

    # attack setting
    parser.add_argument('--eps', default=0.03, type=float)
    # parser.add_argument('--alpha', default=2, type=int)
    parser.add_argument('--attack_steps', default=7, type=int)

    # loss hyperparameter
    # gal: lambda_1 = 0.5
    # adp: lambda_1 = 2.0, lambda_2 = 0.5
    # trs: lambda_1 = 10.0, lambda_2 = 10.0
    # red: lambda_1 = 0.2, lambda_2 = 10.0
    parser.add_argument('--lambda_1', default=2.0, type=float)
    parser.add_argument('--lambda_2', default=0.5, type=float)
    # TRS specific hyperparameter
    parser.add_argument('--plus_adv', action='store_true', help="TRS/RED hyperparameter")
    parser.add_argument('--grad_clip', action='store_true', help="gradident clipping")
    parser.add_argument('--at', action='store_true', help="plus adv training")
    parser.add_argument('--embedding', default=128, type=int,
                        help="hypernetwork parameters")

    # DVERGE specific hyperparameter
    parser.add_argument('--distill_eps', default=0.07, type=float,
                        help='perturbation budget for distillation')
    parser.add_argument('--distill_alpha', default=0.007, type=float,
                        help='step size for distillation')
    parser.add_argument('--distill_steps', default=7, type=int,
                        help='number of steps for distillation')
    parser.add_argument('--distill_layer', default=None, type=int,
                        help='which layer is used for distillation, only useful when distill-fixed-layer is True')
    parser.add_argument('--distill_rand_start', default=False, action="store_true",
                        help='whether use random start for distillation')
    parser.add_argument('--distill_no_momentum', action="store_false", dest='distill_momentum',
                        help='whether use momentum for distillation')
    # gpu
    parser.add_argument('--mul_gpus', action="store_true")

    # test
    parser.add_argument('--load_dir', type=str, default='./checkpoint')
    parser.add_argument('--eval', action="store_true")
    parser.add_argument('--red_test', action="store_true")
    parser.add_argument('--attack_type', type=str, default="all")
    # parser.add_argument('--attack_type', type=str, default="nat",
    #                     choices=['nat', 'fgsm', 'mifgsm', 'pgd', 'deepfool', 'cwl2', 'autoattack', 'all'])

    args = parser.parse_args()

    return args
