import logging
import os
import time

import numpy as np
import pandas as pd
import tabulate
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable

from darts.cnn.genotypes import Genotype
from darts.cnn.model import NetworkCIFAR
# import darts.cnn.utils as darts_utils
from darts.cnn.utils import count_parameters_in_MB, save, AvgrageMeter, accuracy, Cutout
from darts.darts_config import *
from misc.random_string import random_generator


class DARTSTrainer:
    def __init__(self,
                 data_path: str,
                 model_save_path: str,
                 genotype: Genotype,
                 dataset: str = 'cifar10',
                 report_freq: int = 50,
                 eval_policy: str = 'best',
                 gpu_id: int = 0,
                 epochs: int = 50,
                 cutout: bool = False,
                 train_portion: float = 0.7,
                 save_interval: int = 10,
                 auxiliary_tower: bool = True,
                 hash_string: str = None,
                 ):
        """
        Train a DARTS architecture on a benchmark dataset, given the Genotype
        """
        self.data_path = data_path
        self.genotype = genotype
        self.save_interval = save_interval
        self.auxiliary_tower = auxiliary_tower
        self.report_freq = report_freq
        if not torch.cuda.is_available():
            raise ValueError("No GPU is available!")
        self.gpu_id = gpu_id
        torch.cuda.set_device(self.gpu_id)
        self.epochs = epochs
        self.cutout = cutout
        self.train_portion = train_portion
        self.eval_policy = eval_policy

        assert dataset in ['cifar10', 'cifar100', 'imagenet'], "dataset " + str(dataset) + " is not recognised!"
        # Prepare the model for training
        if dataset == 'cifar10':
            self.model = NetworkCIFAR(INIT_CHANNELS, 10, LAYERS, self.auxiliary_tower, genotype)
            self.model.cuda()
            train_transform, valid_transform = data_transforms_cifar('cifar10', self.cutout, CUTOUT_LENGTH)
            self.train_data = dset.CIFAR10(root=self.data_path, train=True, download=True, transform=train_transform)
        elif dataset == 'cifar100':
            self.model = NetworkCIFAR(INIT_CHANNELS, 100, LAYERS, self.auxiliary_tower, genotype)
            train_transform, valid_transform = data_transforms_cifar('cifar100', self.cutout, CUTOUT_LENGTH)
            self.train_data = dset.CIFAR100(root=self.data_path, train=True, download=True, transform=train_transform)
        else:
            raise NotImplementedError("Not implemented yet!")
        self.dataset = dataset

        # Initialise the val/train acc/loss
        self.training_stats = pd.DataFrame(np.nan,
                                           columns=['epoch', 'lr', 'train_acc', 'val_acc', 'val_acc_top5', 'time'],
                                           index=np.arange(self.epochs))
        self.stats, self.eval_stats = {}, {}
        self.n_params = count_parameters_in_MB(self.model)
        self.trained = False
        # generate a unique key representation of the arch.
        self.key = hash_string if hash_string is not None else random_generator(7)
        self.model_save_path = os.path.join(model_save_path, self.key)
        # if not os.path.exists(self.model_save_path): os.makedirs(self.model_save_path)

    def train(self):
        """Actually train the model"""
        print('--Current architecture hash string: ', self.key, '--')
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            LR,
            momentum=MOMENTUM,
            weight_decay=WD
        )

        num_train = len(self.train_data)
        indices = list(range(num_train))
        split = int(np.floor(self.train_portion * num_train))

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs, eta_min=1e-6)
        train_queue = torch.utils.data.DataLoader(
            self.train_data, batch_size=BATCH_SIZE,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
            pin_memory=True, num_workers=0)

        valid_queue = torch.utils.data.DataLoader(
            self.train_data, batch_size=BATCH_SIZE,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
            pin_memory=True, num_workers=0)

        for e in range(self.epochs):
            self.model.drop_path_prob = DROPPATH_PROB * e / self.epochs
            start = time.time()
            train_acc, train_obj = train_step(train_queue, self.model, criterion, optimizer, self.gpu_id)
            valid_acc, valid5_acc, valid_obj = valid_step(valid_queue, self.model, criterion, self.gpu_id)
            scheduler.step()
            end = time.time()
            values = [e, scheduler.get_lr()[0], train_acc, valid_acc, valid5_acc, end - start]
            self.training_stats.iloc[e, :] = values
            # Save model if it is the best val acc encountered so far
            # if valid_acc >= np.max(self.training_stats.loc[:e, 'val_acc']):
            #     save(self.model, self.model_save_path + '/weights.pt')
            table = tabulate.tabulate([values], headers=self.training_stats.columns, tablefmt='simple', floatfmt='8.4f')
            if e % 40 == 0:
                table = table.split('\n')
                table = '\n'.join([table[1]] + table)
            else:
                table = table.split('\n')[2]
            print(table)
        total_time = np.sum(self.training_stats.time)
        self.stats = {'train_stats': self.training_stats, 'model_size': self.n_params, 'time': total_time,
                      'genotype': self.genotype, 'hash': self.key}
        self.trained = True

    def retrieve(self, which='val'):
        if self.trained is False:
            logging.error('The requested architecture has not been trained')
            return None
        if which == 'val':
            data = self.training_stats.val_acc
        elif which == 'train':
            data = self.training_stats.train_acc
        elif which == 'val5':
            data = self.training_stats.val_acc_top5
        else:  # Return full statistics
            raise ValueError("unknown parameter " + which)

        if self.eval_policy == 'best':
            return np.max(data), self.stats
        elif self.eval_policy == 'last':
            return data[-1], self.stats
        elif self.eval_policy == 'last5':
            return np.mean(data[-5:]), self.stats


class DARTSEvaluater(DARTSTrainer):

    def __init__(self, data_path: str,
                 model_save_path: str,
                 genotype: Genotype,
                 dataset: str = 'cifar10',
                 report_freq: int = 50,
                 eval_policy: str = 'last5',
                 gpu_id: int = 0,
                 epochs: int = 50,
                 cutout: bool = False,
                 save_interval: int = 10,
                 auxiliary_tower: bool = False,
                 hash_string: str = None, ):
        """Evaluate a DARTS-style architecture on benchmark"""
        super(DARTSEvaluater, self).__init__(data_path, model_save_path, genotype, dataset, report_freq, eval_policy,
                                             gpu_id, epochs, cutout, 0.5, save_interval, auxiliary_tower, hash_string)
        if dataset == 'cifar10':
            self.model = NetworkCIFAR(EVAL_INIT_CHANNEL, 10, EVAL_LAYERS, self.auxiliary_tower, genotype)
            self.model.cuda()
            train_transform, valid_transform = data_transforms_cifar('cifar10', self.cutout, CUTOUT_LENGTH)
            self.train_data = dset.CIFAR10(root=self.data_path, train=True, download=True, transform=train_transform)
            self.val_data = dset.CIFAR10(root=self.data_path, train=False, download=True, transform=valid_transform)
        elif dataset == 'cifar100':
            self.model = NetworkCIFAR(EVAL_INIT_CHANNEL, 100, EVAL_LAYERS, self.auxiliary_tower, genotype)
            self.model.cuda()
            train_transform, valid_transform = data_transforms_cifar('cifar100', self.cutout, CUTOUT_LENGTH)
            self.train_data = dset.CIFAR100(root=self.data_path, train=True, download=True, transform=train_transform)
            self.val_data = dset.CIFAR100(root=self.data_path, train=False, download=True, transform=valid_transform)
        else:
            raise NotImplementedError

    def train(self):
        """Actually train the model"""
        logging.info("Start training.")
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            0.1,
            momentum=MOMENTUM,
            weight_decay=WD,
            # nesterov=True,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs, eta_min=1e-6)
        train_queue = torch.utils.data.DataLoader(
            self.train_data, batch_size=EVAL_BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)
        # worker set to 0 not to spawn children processes

        valid_queue = torch.utils.data.DataLoader(
            self.val_data, batch_size=EVAL_BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=2)

        for e in range(self.epochs):
            self.model.drop_path_prob = DROPPATH_PROB * e / self.epochs
            start = time.time()
            train_acc, train_obj = train_step(train_queue, self.model, criterion, optimizer, self.gpu_id)
            valid_acc, valid5_acc, valid_obj = valid_step(valid_queue, self.model, criterion, self.gpu_id)
            scheduler.step()
            end = time.time()
            values = [e, scheduler.get_lr()[0], train_acc, valid_acc, valid5_acc, end - start]
            self.training_stats.iloc[e, :] = values
            table = tabulate.tabulate([values], headers=self.training_stats.columns, tablefmt='simple', floatfmt='8.4f')
            if e % 40 == 0:
                table = table.split('\n')
                table = '\n'.join([table[1]] + table)
            else:
                table = table.split('\n')[2]
            print(table)
        if not os.path.exists(self.model_save_path): os.makedirs(self.model_save_path)
        save(self.model, self.model_save_path + '/eval_weights.pt')
        total_time = np.sum(self.training_stats.time)
        self.stats = {'eval_stats': self.training_stats, 'model_size': self.n_params, 'eval_time': total_time,
                      'genotype': self.genotype, 'hash': self.key}
        # pickle.dump(eval_stats, open(self.model_save_path + "/eval_stats.pickle"))
        self.trained = True


def train_step(train_queue, model, criterion, optimizer, auxiliary, gpu_id=0):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    model.train()

    device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() \
                              else 'cpu')

    for step, (input, target) in enumerate(train_queue):
        input = Variable(input).to(device)
        target = Variable(target).to(device)

        optimizer.zero_grad()
        logits, logits_aux = model(input)
        loss = criterion(logits, target)
        if auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += AUXILIARY_WEIGHT * loss_aux
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), GRAD_CLIP)
        # model.parameters().clip_grad_norm_(GRAD_CLIP)
        optimizer.step()

        prec1, prec5 = accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % REPORT_FREQ == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def valid_step(valid_queue, model, criterion, gpu_id=0):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    model.eval()

    device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() \
                              else 'cpu')

    for step, (input, target) in enumerate(valid_queue):
        with torch.no_grad():
            input = Variable(input).to(device)
            target = Variable(target).to(device)

            logits, _ = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % REPORT_FREQ == 0:
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, top5.avg, objs.avg


def data_transforms_cifar(which, cutout, cutout_length):
    if which == 'cifar10':
        CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
        CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
    elif which == 'cifar100':
        CIFAR_MEAN = [0.5071, 0.4867, 0.4408]
        CIFAR_STD = [0.2675, 0.2565, 0.2761]
    else:
        raise NotImplementedError

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if cutout:
        train_transform.transforms.append(Cutout(cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


if __name__ == '__main__':
    from darts.cnn.genotypes import DARTS_V2

    t = DARTSTrainer('/media/xwan/SSD/PythonProjects/graphbayesnas/data/cifar10',
                     '/media/xwan/SSD/PythonProjects/graphbayesnas/results/darts/', DARTS_V2, epochs=10)
    t.train()
    print(t.retrieve())
