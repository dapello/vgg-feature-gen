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
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import vgg


model_names = sorted(name for name in vgg.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("vgg")
                     and callable(vgg.__dict__[name]))

data_sets = ['CIFAR10', 'CIFAR100', 'MNIST', 'HvM64']

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
parser.add_argument('--dataaug', default=1, type=int,
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
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-s', '--sample-features', dest='sample_features', action='store_true',
                    help='sample model features on unshuffled training and validation set')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('-gpu', '--use-cuda', dest='use_cuda', action='store_true',
                    help='use cuda')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--feature-dir', dest='feature_dir',
                    help='The directory used to save the extracted features',
                    default='feature_temp', type=str)


best_prec1 = 0

# outputs acts as a global buffer to save extracted features to before dumping to .h5 file.
outputs = {}
outputs['inputs'] = []
outputs['labels'] = []
 

def main():
    global args, outputs, samplePoints, best_prec1
    args = parser.parse_args()

    # repeatability, damnit
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed(args.seed)
    
    print('cuda',args.use_cuda)

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Check the feature_dir exists or not
    if not os.path.exists(args.feature_dir):
        os.makedirs(args.feature_dir)

    if args.pretrained:
        model = torchvision.models.vgg16(pretrained=True)

    else:
        model = vgg.__dict__[args.archclass](args.arch, classes=args.classes, batchnorm=args.batchnorm, dropout=args.dropout)

    print('>>> Model Parameters: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    # model.features = torch.nn.DataParallel(model.features)


    if args.use_cuda:
        model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.use_cuda:
                checkpoint = torch.load(args.resume)
            else:
                checkpoint = torch.load(args.resume, map_location='cpu')
            # args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
           
            # if freeze_upto isn't 0, freeze weights for layers up to freeze_upto from the last layer
            if args.freeze_upto != 'none':
                model_dict = model.state_dict()
                pretrained_dict = {}

                for name, L in model.named_parameters():
                    if args.freeze_upto in name:
                        break
                    else:
                        print('params {} loaded.'.format(name))
                        pretrained_dict[name] = L

                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)

                for name, L in model.named_parameters():
                    if args.freeze_upto in name:
                        break
                    else:
                        print('params {} frozen.'.format(name))
                        L.requires_grad = False

            else:
                model.load_state_dict(checkpoint['state_dict'])

            print("=> loaded checkpoint (epoch {})"
                  .format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    
    train_loader, val_loader = construct_data_loaders(args)


    # define loss function (criterion) and pptimizer
    if args.use_cuda:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()#.cuda()
    
    if args.half:
        model.half()
        criterion.half()

    model_parameters = filter(lambda p: p.requires_grad,model.parameters())
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model_parameters, args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'ADAM':
        optimizer = torch.optim.Adam(model_parameters, lr=0.001, weight_decay=args.weight_decay)


    if args.sample_features:
        for i, L in enumerate(model.features):
            # if 'ReLU' not in str(L) and "Dropout" not in str(L):
            name = 'features_'+str(i)+"_"+str(L)
            extractor = Extractor(name)
            L.register_forward_hook(extractor.extract)
            print('applied forward hook to extract features from:{}'.format(name))

        for i, L in enumerate(model.classifier):
            if "Dropout" not in str(L):
                name = 'classifier_'+str(i)+"_"+str(L)
                extractor = Extractor(name)
                L.register_forward_hook(extractor.extract)
                print('applied forward hook to extract features from:{}'.format(name))
        
        sample(train_loader, model, criterion, args.start_epoch, 'train')
        #sample(val_loader, model, criterion, args.start_epoch, 'val')
        return

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    # save initial state of the network as epoch 0
    save_checkpoint({
        'epoch': 0,
        'state_dict': model.state_dict(),
        'best_prec1': 0,
    }, False, filename=os.path.join(args.save_dir, 'checkpoint_{}.tar'.format(0)))

    for epoch in range(args.start_epoch, args.epochs):
        print('beginning epoch: ', epoch)
        if (args.dataset == 'CIFAR100'):
            if not (int(args.classes) == 100):
                train_loader, val_loader = construct_data_loaders(args)

        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)
        clear(outputs)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch=epoch)
        clear(outputs)

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
        outputs['inputs'].append(input.detach().cpu().numpy())
        outputs['labels'].append(target.detach().cpu().numpy())

        # measure data loading time
        data_time.update(time.time() - end)

        if args.use_cuda:
            #target = target.cuda(async=True)
            target = target.cuda(async=True).long()
            input_var = torch.autograd.Variable(input).cuda()
        else:
            target = target#.cuda(async=True)
            input_var = torch.autograd.Variable(input)#.cuda()

        #target_var = torch.autograd.Variable(target)
        target_var = torch.autograd.Variable(target).long()
        if args.half:
            input_var = input_var.half()

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
        outputs['inputs'].append(input.detach().cpu().numpy())
        outputs['labels'].append(target.detach().cpu().numpy())

        if args.use_cuda:
            target = target.cuda(async=True).long()
            input_var = torch.autograd.Variable(input, volatile=True).cuda()
        else:
            target = target#.cuda(async=True)
            input_var = torch.autograd.Variable(input, volatile=True)#.cuda()
        target_var = torch.autograd.Variable(target, volatile=True).long()
        if args.half:
            input_var = input_var.half()

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

    return top1.avg

def sample(loader, model, criterion, epoch, image_set):
    """ 
        Run one epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to eval mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(loader):
        outputs['inputs'].append(input.detach().cpu().numpy())
        outputs['labels'].append(target.detach().cpu().numpy())

        if args.use_cuda:
            target = target.cuda(async=True).long()
            input_var = torch.autograd.Variable(input, volatile=True).cuda()
        else:
            target = target#.cuda(async=True)
            input_var = torch.autograd.Variable(input, volatile=True)#.cuda()
        
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
    
        save_template = '{image_set}/{image_set}-ep_{epoch}-step_{i}-acc_{top1.avg:.2f}-'.format(
                image_set=image_set, 
                epoch=epoch, 
                i=i, 
                top1=top1
        )

        save_features(outputs, save_template) 
        clear(outputs)

        if i % args.print_freq == 0:
            pass
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))

class Extractor(object):
    def __init__(self, name):
        self.name = name
        outputs[self.name]=[]

    def extract(self, module, input, output):
        outputs[self.name].append(output.detach().cpu().numpy())

def construct_data_loaders(args):
    print("constructing dataset loaders: ", args.dataset)
    print('Data augmentation:', args.dataaug)
    if args.dataset == 'CIFAR10':
        if args.dataaug:
            train_transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])
        else:
            print('no data augmentation')
            train_transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])

        val_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])

        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform)

    elif args.dataset == "CIFAR100":
        if int(args.classes) == 100:
            if args.dataaug:
                train_transform=transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                ])
            else:
                print('no data augmentation')
                train_transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                ])
            
            val_transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])

            train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
            val_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=val_transform)
        else:
            if args.dataaug:
                train_transform=transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                ])
            else:
                print('no data augmentation')
                train_transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                ])
            
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100(
                        root='./data', 
                        download=True,
                        train=True,
                        transform=train_transform),
                batch_size=50000, 
                shuffle=False, 
                num_workers=4, 
                pin_memory=False
            )
            
            for (images,labels) in train_loader:
                # load images
                pass 
            
            train_dataset = torch.utils.data.TensorDataset(*sub_cifar100(
                images,
                labels,
                args.classes,
                500,
                invert=args.invert
                ))
            
            val_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100(
                        root='./data', 
                        download=True,
                        train=False,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                ])),
                batch_size=10000, 
                shuffle=False, 
                num_workers=4, 
                pin_memory=False
            )
            
            for (images,labels) in val_loader:
                # load images
                pass 
            
            val_dataset = torch.utils.data.TensorDataset(*sub_cifar100(
                images,
                labels,
                args.classes,
                100,
                invert=args.invert
                ))
    elif args.dataset == "MNIST":
        ds = datasets.MNIST
        # custom transforms
    elif args.dataset == "HvM64":
        trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([224,224]),
            transforms.ToTensor(),
            transforms.Normalize((0.44688916, 0.44688916, 0.44688916),(0.17869806, 0.17869806, 0.17869806))
        ])

        data = loadmat('data/HvM_128px.mat')
        #V = 'V0'
        #X = torch.Tensor(data['imgs'][data['obj_version']==V].transpose(0,3,1,2))
        X = torch.Tensor(data['imgs'].transpose(0,3,1,2))
        X = torch.stack([trans(x) for x in X])
        #Y = data['obj_id'][data['obj_version']==V]
        Y = data['obj_id']

        # convert Y to a unique integer for a give label
        labels = list(np.unique(Y))
        Y = torch.Tensor([labels.index(Y[i]) for i in range(len(Y))])
        
        train_dataset = torch.utils.data.TensorDataset(X, Y)
        val_dataset = torch.utils.data.TensorDataset(X, Y)
        del X, Y
    else:
        print('No valid dataset specified!')
        return

    shuffle = True
    if args.sample_features:
        shuffle = False

    # shuffle and augment training images during training
    print("Creating train_loader. shuffling: {}".format(shuffle))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=shuffle,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    return train_loader, val_loader

def sub_cifar100(images, labels, numcat, numexamp, invert=0):
    print('numcat:', numcat)
    flatten = lambda l: [item for sublist in l for item in sublist]
   
    # invert reverses starting order for image catagories
    if invert:
        label_set = torch.tensor(flatten([list(range((9-i), 100, 10)) for i in range(int(numcat/10))]))
    else:
        label_set = torch.tensor(flatten([list(range(i, 100, 10)) for i in range(int(numcat/10))]))
    print('label set:', label_set)
    sub_images = torch.zeros(numcat,numexamp,3,32,32)
    sub_labels = torch.zeros(numcat,numexamp)
    for i in range(len(label_set)):
        sub_images[i] = images[labels==label_set[i]]
        sub_labels[i] = labels[labels==label_set[i]]
        
    sub_images = sub_images.reshape(
        sub_images.shape[0]*sub_images.shape[1],
        sub_images.shape[2],
        sub_images.shape[3],
        sub_images.shape[4]
    )
    
    sub_labels = sub_labels.reshape(sub_labels.shape[0]*sub_labels.shape[1])
    sub_unique = list(np.unique(sub_labels))
    for i in range(len(sub_labels)):
        sub_labels[i] = sub_unique.index(sub_labels[i])

    print('>>sub_images shape:', sub_images.shape)
    return sub_images, sub_labels


def save_features(outputs, path):
    outputs = process_outputs(outputs)
    print(path)
    for key in outputs:
        data = outputs[key]
        fullPath = os.path.join(args.feature_dir, path + key + '-featnum_{}-.h5').replace(' ', '_').format(data.shape[1])
        if not os.path.exists(os.path.dirname(fullPath)):
            os.makedirs(os.path.dirname(fullPath))

        print('saving data shape:{} \nfor key:{} \nat path:{}'.format(data.shape, key, fullPath))
        f = h5.File(fullPath, 'w')
        f.create_dataset('obj_arr', data=data)
        f.close()

def process_outputs(outputs):
    dataDict = {}
    for key in outputs:
        data = np.array(outputs[key]).astype('float16')
        data = np.reshape(data, (data.shape[0]*data.shape[1], -1))
        print(key, data.shape)
        dataDict[key] = data

    return dataDict
    
def clear(outputs):
    for key in outputs:
        outputs[key] = []
    
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


if __name__ == '__main__':
    main()
