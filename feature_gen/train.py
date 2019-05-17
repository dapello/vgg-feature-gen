import argparse
import os
import shutil
import time
import random

import numpy as np
import h5py as h5
from scipy.io import loadmat

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

# local packages
import models
from datasets import construct_data_loaders

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("vgg")
                     and callable(models.__dict__[name]))

data_sets = ['CIFAR10', 'CIFAR100', 'EMNIST', 'MNIST', 'HvM64', 'HvM64_V0', 'HvM64_V3', 'HvM64_V6', 'HvM64.r', 'HvM64_V0.r', 'HvM64_V3.r', 'HvM64_V6.r']

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--archclass', '-ac', metavar='ARCHCLASS', default='vgg_s')
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg16')
parser.add_argument('--dataset', '-d', metavar='ARCH', default='CIFAR10',
                    choices=data_sets,
                    help='Dataset, choose from: ' + ' | '.join(data_sets) +
                    ' (default: CIFAR10)')
parser.add_argument('--invert', default=0, type=int, metavar='N',
                    help='switch starting order of classes from the dataset to use')
parser.add_argument('--classes', default=10, type=int, metavar='N',
                    help='number of classes from the dataset to use')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--seed', dest='seed', default=0, type=int,
                    help='random number generator seed')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batchnorm', default=0, type=int,
                    help='If true, construct networks with batchnorm.')
parser.add_argument('--dropout', default=0.0, type=float,
                    help='probability of dropout')
parser.add_argument('--dataaug', action='store_true',
                    help='whether or not to train with data augmentation')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--optimizer', metavar='OPT', default='SGD',
                    choices=['SGD', 'ADAM'],
                    help='Optimizer choices: ' + ' | '.join(['SGD', 'ADAM']) +
                    ' (default: SGD)')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--freeze-upto', '-fu', dest='freeze_upto', default='none', type=str,
                    help='layer from the end of the network to freeze weights at.')
parser.add_argument('--freeze-after', '-fa', dest='freeze_after', default='none', type=str,
                    help='layer from the end of the network to freeze weights at.')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
#parser.add_argument('--repepoch', dest='replacement_epoch', default='', type=str, metavar='PATH',
#                    help='path to replacement checkpoint (default: none)')
#parser.add_argument('--replayer', dest='replace_layer', default='', type=str,
#                    help='layer to replace in model')
#parser.add_argument('--repweights', dest='replace_weights', action='store_true',
#                    help='whether or not to replace a layers weights in the model.')
#parser.add_argument('-s', '--sample-features', dest='sample_features', action='store_true',
#                    help='sample model features on unshuffled training and validation set')
#parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
#                    help='evaluate model on validation set')
#parser.add_argument('--reinit-test', dest='reinit_test', action='store_true',
#                    help='evaluate model on validation set with each layer reinitialized to epoch 0')
#parser.add_argument('--pretrained', dest='pretrained', action='store_true',
#                    help='use pre-trained model')
#parser.add_argument('-gpu', '--use-cuda', dest='use_cuda', action='store_true',
#                    help='use cuda -- deprecated, always uses GPU+cuda')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
#parser.add_argument('--feature-dir', dest='feature_dir',
#                    help='The directory used to save the extracted features',
#                    default='feature_temp', type=str)
parser.add_argument('--results-dir', dest='results_dir',
                    help='The directory used to save results ie from reinit test',
                    default='result_temp', type=str)


best_prec1 = 0


def main():
    global args, samplePoints, best_prec1
    args = parser.parse_args()
    print('running main.py with args:', args)

    set_seed(args.seed) 
    construct_filesystem(args)

    model = models.__dict__[args.archclass](args.arch, classes=args.classes, batchnorm=args.batchnorm, dropout=args.dropout)
    model.cuda()

    print('>>> Model Parameters: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            # args.start_epoch = checkpoint['epoch']

            best_prec1 = checkpoint['best_prec1']
          
            #if args.replace_weights:
            #    model.load_state_dict(checkpoint['state_dict'])
            #    
            #    layer = 'features.' + args.replace_layer
            #    replacement_epoch = args.replacement_epoch
            #    model_i = models.__dict__[args.archclass](args.arch, classes=args.classes, batchnorm=args.batchnorm, dropout=args.dropout)

            #    checkpoint_i = torch.load(replacement_epoch)

            #    model_i.load_state_dict(checkpoint_i['state_dict'])
            #    w_i = model_i.state_dict()[layer].data.clone()
            #    model.state_dict()[layer].data.copy_(w_i)

            #    print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
            #    print("=> replaced {} weights with checkpoint (epoch {})".format(layer, checkpoint_i['epoch']))

            ## if freeze_upto isn't 0, freeze weights for layers up to freeze_upto from the last layer
            #elif args.freeze_upto != 'none':
            #    model_dict = model.state_dict()
            #    pretrained_dict = {}

            #    for name, L in model.named_parameters():
            #        if args.freeze_upto in name:
            #            break
            #        else:
            #            print('params {} loaded.'.format(name))
            #            pretrained_dict[name] = L

            #    model_dict.update(pretrained_dict)
            #    model.load_state_dict(model_dict)

            #    for name, L in model.named_parameters():
            #        if args.freeze_upto in name:
            #            break
            #        else:
            #            print('params {} frozen.'.format(name))
            #            L.requires_grad = False

            #else:
            model.load_state_dict(checkpoint['state_dict'])

            print("=> loaded checkpoint (epoch {})"
                  .format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

        print('freeze after:', args.freeze_after)

    #if args.freeze_after != 'none':
    #    freeze = False
    #    for name, L in model.named_parameters():
    #        if args.freeze_after in name:
    #            freeze = True
    #        
    #        if freeze:
    #            print('params {} frozen.'.format(name))
    #            L.requires_grad = False
    #        else:
    #            print('params {} not frozen.'.format(name))


    cudnn.benchmark = True
    
    train_loader, val_loader = construct_data_loaders(args)

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()

    model_parameters = filter(lambda p: p.requires_grad,model.parameters())
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model_parameters, args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'ADAM':
        optimizer = torch.optim.Adam(model_parameters, lr=0.001, weight_decay=args.weight_decay)
    
    # save initial state of the network as epoch 0
    save_checkpoint({
        'epoch': 0,
        'state_dict': model.state_dict(),
        'best_prec1': 0,
    }, False, filename=os.path.join(args.save_dir, 'checkpoint_{}.tar'.format(0)))

    for epoch in range(args.start_epoch, args.epochs):
        print('beginning epoch: ', epoch)
        
        # reshuffling procedure for subsampled CIFAR100
        if (args.dataset == 'CIFAR100'):
            if not (int(args.classes) == 100):
                train_loader, val_loader = construct_data_loaders(args)

        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1, loss_avg = validate(val_loader, model, criterion, epoch=epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(args.save_dir, 'checkpoint_{}.tar'.format(epoch + 1)))


def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True).long()
        input_var = torch.autograd.Variable(input).cuda()

        #target_var = torch.autograd.Variable(target)
        target_var = torch.autograd.Variable(target).long()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, criterion, epoch=None):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True).long()
        input_var = torch.autograd.Variable(input, volatile=True).cuda()
        target_var = torch.autograd.Variable(target, volatile=True).long()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg, losses.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lr = args.lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def construct_filesystem(args):
    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Check the results_dir exists or not
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

def set_seed(seed):
    # repeatability, damnit
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

if __name__ == '__main__':
    main()
