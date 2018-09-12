import argparse
import os
import shutil
import time

import numpy as np
import h5py as h5

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import vgg

np.random.seed(0)
torch.manual_seed(0)

model_names = sorted(name for name in vgg.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("vgg")
                     and callable(vgg.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg19',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: vgg19)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-s', '--sample-train-features', dest='sample_train_features', action='store_true',
                    help='sample model features on unshuffled training set')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--use-cuda', dest='use_cuda', default=False,
                    help='use cuda', type=bool)
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
 

samplePoints = np.concatenate([
    np.arange(0,10), 
    np.arange(10,25,2), 
    np.arange(25,50,4), 
    np.arange(50,150,10), 
    np.arange(150,300,20)
])

def main():
    global args, outputs, samplePoints, best_prec1
    args = parser.parse_args()

    print('cuda',args.use_cuda)

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Check the feature_dir exists or not
    if not os.path.exists(args.feature_dir):
        os.makedirs(args.feature_dir)

    model = vgg.__dict__[args.arch]()

    # model.features = torch.nn.DataParallel(model.features)
    

    for i, L in enumerate(model.features):
        if 'ReLU' not in str(L) and "Dropout" not in str(L):
            name = 'features_'+str(i)+"_"+str(L)
       	    extractor = Extractor(name)
       	    L.register_forward_hook(extractor.extract)
       	    print('applied forward hook to extract features from:{}'.format(name))

    for i, L in enumerate(model.classifier):
        if 'ReLU' not in str(L) and "Dropout" not in str(L):
            name = 'classifier_'+str(i)+"_"+str(L)
       	    extractor = Extractor(name)
       	    L.register_forward_hook(extractor.extract)
       	    print('applied forward hook to extract features from:{}'.format(name))

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
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    feature_extract_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root='./data', train=True, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and pptimizer
    if args.use_cuda:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()#.cuda()

    if args.half:
        model.half()
        criterion.half()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.sample_train_features:
        sample_features(feature_extract_loader, model, criterion, checkpoint['epoch'])
        return

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
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
        }, is_best, filename=os.path.join(args.save_dir, 'checkpoint_{}.tar'.format(epoch)))


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
            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input).cuda()
        else:
            target = target#.cuda(async=True)
            input_var = torch.autograd.Variable(input)#.cuda()

        target_var = torch.autograd.Variable(target)
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
            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input, volatile=True).cuda()
        else:
            target = target#.cuda(async=True)
            input_var = torch.autograd.Variable(input, volatile=True)#.cuda()
        target_var = torch.autograd.Variable(target, volatile=True)

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

def sample_features(feature_extract_loader, model, criterion, epoch):
    """ 
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to eval mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(feature_extract_loader):
        outputs['inputs'].append(input.detach().cpu().numpy())
        outputs['labels'].append(target.detach().cpu().numpy())

        if args.use_cuda:
            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input).cuda()
        else:
            target = target#.cuda(async=True)
            input_var = torch.autograd.Variable(input)#.cuda()

        target_var = torch.autograd.Variable(target)
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
    

        save_features(outputs,'train_ep_{}_step_{}_'.format(epoch, i)) 
        clear(outputs)

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(feature_extract_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))

class Extractor(object):
    def __init__(self, name):
        self.name = name
        outputs[self.name]=[]

    def extract(self, module, input, output):
        outputs[self.name].append(output.detach().cpu().numpy())

class Downsampler(object):
    def __init__(self, **kwargs):
        self.keys = []
        self.perms_dict = {}
        self.samples=5000

    def downsample(self, layer):
        np.random.seed(0)
        print('original layer shape: {}'.format(layer.shape))
        fsize = layer.shape[-1]
        if fsize > self.samples:
            if fsize in self.keys:
                print('using perm for layer size: ',fsize)
                perm = self.perms_dict[fsize]
            else:
                print('creating new perm for layer size: ',fsize)
                perm = np.sort(np.random.permutation(fsize)[:self.samples])
                self.keys.append(fsize)
                self.perms_dict[fsize] = perm
            layer = layer[:,perm]
        print('new layer shape: {}'.format(layer.shape)) 
        return layer

def save_features(outputs, path):
    outputs = process_outputs(outputs)
    for key in outputs:
        fullPath = os.path.join(args.feature_dir, path + key + '.h5').replace(' ', '_').replace('(','').replace(')','')
        f = h5.File(fullPath, 'w')
        print('saving output: {}'.format(key))
        data = outputs[key]
        print(data.shape)
        f.create_dataset('obj_arr', data=data)
        f.close()

def process_outputs(outputs):
    # labels = np.reshape(outputs['labels'],-1)
    # numLabels = np.unique(labels).shape[0]
    # count, _ = np.histogram(labels, bins=numLabels)
    # minCount = count.min()-1
    dataDict = {}
    for key in outputs:
        data = np.array(outputs[key]).astype('float16')
        data = np.reshape(data, (data.shape[0]*data.shape[1], -1))
        print(key, data.shape)
        # data = ds.downsample(data)
        # minCount clips the arrays to the shortest number of labelled examples in the epoch.
        # dataByLabel = np.array([data[labels==label][:minCount] for label in np.unique(labels)])
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
    ds = Downsampler()
    main()
