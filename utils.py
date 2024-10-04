import sys
import logging
import warnings
import os
import torch

import torch.nn as nn
import torch.utils.data as data

from PIL import Image
from torchvision import datasets, transforms
from collections import defaultdict

logging.basicConfig(
    format='%(levelname)-8s %(asctime)-12s %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if os.environ.get('AUTOLIRPA_DEBUG', 0) else logging.INFO)

warnings.simplefilter("once")


class MultiAverageMeter(object):
    """Computes and stores the average and current value for multiple metrics"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum_meter = defaultdict(float)
        self.lasts = defaultdict(float)
        self.counts_meter = defaultdict(int)

    def update(self, key, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.item()
        self.lasts[key] = val
        self.sum_meter[key] += val * n
        self.counts_meter[key] += n

    def last(self, key):
        return self.lasts[key]

    def avg(self, key):
        if self.counts_meter[key] == 0:
            return 0.0
        else:
            return self.sum_meter[key] / self.counts_meter[key]

    def __repr__(self):
        s = ""
        for k in self.sum_meter:
            s += "{}={:.4f} ".format(k, self.avg(k))
        return s.strip()


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print({'Total': total_num, 'Trainable': trainable_num})


def get_named_paras(net):
    for name, parameters in net.named_parameters():
        print(name, ': ', parameters.size())


def parseClasses(file):
    classes = []
    filenames = []
    with open(file) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    for x in range(0, len(lines)):
        tokens = lines[x].split()
        classes.append(tokens[1])
        filenames.append(tokens[0])
    return filenames, classes


class TinyImgNetDataset(data.Dataset):
    """Dataset wrapping images and ground truths."""

    def __init__(self, img_path, gt_path, class_to_idx=None, transform=None):
        self.img_path = img_path
        self.transform = transform
        self.gt_path = gt_path
        self.class_to_idx = class_to_idx
        self.classidx = []
        self.imgs, self.classnames = parseClasses(gt_path)
        for classname in self.classnames:
            self.classidx.append(self.class_to_idx[classname])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, y) where y is the label of the image.
        """
        img = None
        with open(os.path.join(self.img_path, self.imgs[index]), 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
        y = self.classidx[index]
        return img, y

    def __len__(self):
        return len(self.imgs)


def get_loaders(args):
    if 'CIFAR' in args.dataset:
        # transforms
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        # train set
        train_dataset = eval(f'datasets.{args.dataset}')(
            os.path.join(args.data_dir, args.dataset), train=True,
            transform=train_transform, download=True
        )
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=args.train_batch_size, shuffle=True,
            pin_memory=args.pin_memory, num_workers=args.num_workers,
        )
        # test set
        test_dataset = eval(f'datasets.{args.dataset}')(
            os.path.join(args.data_dir, args.dataset), train=False,
            transform=test_transform, download=True
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=args.test_batch_size, shuffle=False,
            pin_memory=args.pin_memory, num_workers=args.num_workers,
        )
    # elif 'ImageNet' in args.dataset:
    #     train_transform = transforms.Compose([
    #         transforms.RandomResizedCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #     ])
    #     test_transform = transforms.Compose([
    #         transforms.Resize((224, 224)),
    #         transforms.ToTensor(),
    #     ])
    #     train_dataset = datasets.ImageFolder(
    #         root=f'{args.data_dir}/{args.dataset}/train', transform=train_transform)
    #     test_dataset = datasets.ImageFolder(
    #         root=f'{args.data_dir}/{args.dataset}/val', transform=test_transform)
    elif args.dataset == "TinyImageNet":
        # transforms
        train_transform = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        # train set
        train_dataset = datasets.ImageFolder(root=f'{args.data_dir}/{args.dataset}/train', transform=train_transform)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=args.train_batch_size, shuffle=True,
            pin_memory=args.pin_memory, num_workers=args.num_workers,
        )
        # test set
        test_dir = os.path.join(f'{args.data_dir}/{args.dataset}', 'val', 'images')
        target_file = os.path.join(f'{args.data_dir}/{args.dataset}', 'val', 'val_annotations.txt')
        test_dataset = TinyImgNetDataset(
            test_dir, target_file, class_to_idx=train_loader.dataset.class_to_idx.copy(),
            transform=test_transform
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=args.test_batch_size, shuffle=False,
            pin_memory=args.pin_memory, num_workers=args.num_workers,
        )
    else:
        raise ValueError(args.dataset)

    if args.eval:
        return test_loader
    else:
        return train_loader, test_loader


def get_opt2(model, args):

    param = list(model[0].parameters())
    for i in range(1, args.num_models):
        param.extend(list(model[i].parameters()))
    return torch.optim.Adam(param, lr=args.lr_max, weight_decay=args.weight_decay, eps=1e-7)


def get_opt(model, args):

    param = list(model[0].parameters())
    for i in range(1, args.num_models):
        param.extend(list(model[i].parameters()))
    return torch.optim.SGD(param, lr=args.lr_max, momentum=args.momentum,
                               weight_decay=args.weight_decay)


def lr_schedule(t, args):
    if t < args.epochs * 2 / 3.:
        return args.lr_max
    elif t < args.epochs * 11 / 12.:
        return args.lr_max * 0.1
    else:
        return args.lr_max * 0.01


def save_ckpt(args, train_loss, model, epoch, opt, saved_name, logger):
    if not os.path.exists(args.saved_dir):
        os.mkdir(args.saved_dir)
    if args.train_mode == "vanilla":
        saved_path = os.path.join(args.saved_dir, f'{saved_name}.pth')
        state = {
            'model': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
            'epoch': epoch,
            # 'opt': opt,
            # 'train_loss': train_loss
        }
        torch.save(state, saved_path)
        logger.info(f'| Saving best model to {saved_path} ...\n')
    else:
        assert args.num_models > 0
        for i in range(args.num_models):
            saved_path = os.path.join(
                args.saved_dir, f'{saved_name}_{i}.pth'
            )
            state = {
                'model': model[i].module.state_dict() if isinstance(model[i], torch.nn.DataParallel) else model[
                    i].state_dict(),
                'epoch': epoch,
                # 'opt': opt[i] if isinstance(opt, list) else opt,
                # 'train_loss': train_loss
            }
            torch.save(state, saved_path)
            if i < args.num_models - 1:
                logger.info(f'| Saving best model to {saved_path} ...')
            else:
                logger.info(f'| Saving best model to {saved_path} ...\n')


def init_model(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    return model


def ensemble_requires_grad(model: torch.nn.Module, requires_grad: bool) -> None:
    for param in model.parameters():
        param.requires_grad_(requires_grad)
