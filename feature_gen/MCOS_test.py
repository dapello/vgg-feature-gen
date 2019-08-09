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
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn

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
parser.add_argument('--dataset', '-d', metavar='ARCH', default='CIFAR100',
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
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
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
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--repepoch', dest='replacement_epoch', default='', type=str, metavar='PATH',
                    help='path to replacement checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--feature-dir', dest='feature_dir',
                    help='The directory used to save the extracted features',
                    default='feature_temp', type=str)
parser.add_argument('--SVD-base-path', dest='SVD_base_path',
                    help='The directory containing the SVD compressed representations',
                    default='feature_temp', type=str)
parser.add_argument('--results-dir', dest='results_dir',
                    help='The directory used to save results ie from reinit test',
                    default='result_temp', type=str)


best_prec1 = 0

# outputs acts as a global buffer to save extracted features to before dumping to .h5 file.
outputs = {}
outputs['inputs'] = []
outputs['labels'] = []

def main():
    global args, outputs, samplePoints, best_prec1
    args = parser.parse_args()
    print('running main.py with args:', args)

    set_seed(args.seed) 
    construct_filesystem(args)

    if args.pretrained:
        model = torchvision.models.vgg16(pretrained=True)
    else:
        model = models.__dict__[args.archclass](args.arch, classes=args.classes, batchnorm=args.batchnorm, dropout=args.dropout)

    model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            # args.start_epoch = checkpoint['epoch']

            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])

            print("=> loaded checkpoint (epoch {})"
                  .format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    train_loader, val_loader = construct_data_loaders(args, sample=True)
    
    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # initialize original, trained model, get it's best score
    best_prec1, best_loss_avg, best_class_acc = sample(train_loader, model, criterion, args.start_epoch, 'Full')

    k_map = {
            'features_1_ReLU': 65536,
            'features_3_ReLU': 65536,
            'features_6_ReLU': 32768,
            'features_8_ReLU': 32768,
            'features_11_ReLU': 16384,
            'features_13_ReLU': 16384,
            'features_15_ReLU': 16384,
            'features_18_ReLU': 8192,
            'features_20_ReLU': 8192,
            'features_22_ReLU': 8192,
            'features_25_ReLU': 2048,
            'features_27_ReLU': 2048,
            'features_29_ReLU': 2048,
            'features_30_MaxPool2d': 512, 
            'classifier_1_ReLU': 512, 
            'classifier_3_ReLU': 512        
    }

    # next, loop over feature layers in network
    for part_of_model, part_name in [[model.features, 'features'],[model.classifier, 'classifier']]:
        for i, L in enumerate(part_of_model):
            # generate original naming scheme
            name = part_name+'_'+str(i)+"_"+str(L)
            name_to_match = name.split('(')[0]
            print('name_to_match ',name_to_match )
            # if name_to_match in ['features_29_ReLU','features_30_MaxPool2d', 'classifier_1_ReLU', 'classifier_3_ReLU']:
            #if name_to_match in ['features_30_MaxPool2d', 'classifier_1_ReLU', 'classifier_3_ReLU']:
            if name_to_match in ['features_1_ReLU', 'features_15_ReLU']:
                print('caught',name_to_match, i)

               ## bisection like search for dim est
               # start = maxK # max rank of approximation
               # range_ = [1.0,2.0]
               # hist = [0.0]
               # #scores = [100.0]
               # dist = 100.0 

                k = k_map[name_to_match]
                ## create first component
                U = np.random.randn(k,1)
                Q,R = np.linalg.qr(U)
                print('Q.shape', Q.shape)
                
                build_OS_results = []
                sample_n = k

                for component in np.arange(0,sample_n):
                    print('build component {} of {}'.format(component, name_to_match))
                    
                    # create model with bottleneck
                    model_ = bottleneck_model(model, i, Q, loc=part_name)
                    model_.cuda()
                    freeze_trained_weights(model_)

                    model_parameters = filter(lambda p: p.requires_grad,model_.parameters())
                    if args.optimizer == 'SGD':
                        optimizer = torch.optim.SGD(model_parameters, args.lr,
                               momentum=args.momentum,
                               weight_decay=args.weight_decay)

                    # train subspace
                    for epoch in range(0, args.epochs):
                        print('beginning epoch: ', epoch)

                        train_bottleneck_component(train_loader, model_, criterion, optimizer, epoch, component, l=1.0)
                        # train_bottleneck(train_loader, model_, criterion, optimizer, epoch, l=1.0)

                        # evaluate on validation set
                        #prec1, loss_avg = validate_bottleneck(val_loader, model_, criterion, epoch=epoch)

                    ## ensure Q is orthonormalized and then check final validation acc
                    Q = model_.estimator.weight.cpu().detach().numpy().T
                    Q,R = np.linalg.qr(Q)
                    model_ = bottleneck_model(model, i, Q, loc=part_name)
                    model_.cuda()
                    prec1, loss_avg, class_acc = sample(train_loader, model_, criterion, args.start_epoch, k)
                    #prec1, loss_avg = validate_bottleneck(val_loader, model_, criterion, epoch=epoch)
                

                    dist = 100*(best_prec1[0].item() - prec1[0].item())/best_prec1[0].item()
                    
                    print('dist:',dist)

                    build_OS_results.append([
                        name_to_match, 
                        k, 
                        component, 
                        prec1[0].item(), 
                        loss_avg[0].item(), 
                        dist, 
                        best_prec1[0].item(), 
                        str(class_acc),
                        str(best_class_acc),
                        #U.shape[1]
                    ])

                    if dist < 1:
                        pathname = os.path.join(args.results_dir,'MCOS_W_{}_{}.h5'.format(sample_n, name_to_match)) 
                        np.save(pathname, Q)
                        break

                    # and a new random direction
                    U = np.random.randn(512,1)
        
                    # concatenate new random direction
                    Q_ = np.concatenate([Q,U],axis=1)
                    
                    # and make it orthogonal to original Q
                    Q,R = np.linalg.qr(Q_)
                
                pathname = os.path.join(args.results_dir,'MCOS_test_{}_{}.h5'.format(sample_n, name_to_match)) 
                print('saving results at ',pathname)
                f = h5.File(pathname, 'w')
                f.create_dataset('build_OS_result', data=np.array(build_OS_results))
                f.close()
                print('results saved')

class bottleneck_model(nn.Module):
    ''' 
    Expects a VGG like network split into features1, features2, classifier1, and classifier2,
    and an estimator (ie U.T@U, from the SVD of the representations). loc specifies if the estimator
    is to be put into the feature layers or classifier.
    '''
    def __init__(self, model, i, U, loc='features'):
        super(bottleneck_model, self).__init__()
        if loc=='features':
            self.features1 = nn.Sequential(*list(model.features.children())[:i+1])
            self.features2 = nn.Sequential(*list(model.features.children())[i+1:])
            self.classifier1 = nn.Sequential(*list(model.classifier.children()))
            self.classifier2 = nn.Sequential()
        elif loc=='classifier':
            self.features1 = nn.Sequential(*list(model.features.children()))
            self.features2 = nn.Sequential()
            self.classifier1 = nn.Sequential(*list(model.classifier.children())[:i+1])
            self.classifier2 = nn.Sequential(*list(model.classifier.children())[i+1:])
        print(U.shape)
        self.estimator = nn.Linear(in_features=U.shape[0], out_features=U.shape[1], bias=False)
        self.estimator.weight = torch.nn.parameter.Parameter(data=torch.Tensor(U.T))
        
        self.loc = loc
        
    def forward(self, x): 
        x = self.features1(x)
        
        # if estimator goes in between features, this route is called
        if self.loc == 'features':
            size = x.size()
#             print(size)
            x_ = x.view(x.size(0), -1)
#             print(x.size())
            xW = self.estimator(x_)
            xWWt = F.linear(xW, self.estimator.weight.t())
            x = xWWt
            x = x.view(size)
            
        x = self.features2(x)
        x = x.view(x.size(0), -1) 
        x = self.classifier1(x)
        
        if self.loc == 'classifier':
            x_ = x
            xW = self.estimator(x_)
            xWWt = F.linear(xW, self.estimator.weight.t())
            x = xWWt
        
        x = self.classifier2(x)
        
        return x#, [x_, xWWt]

def freeze_trained_weights(model):
    for name, L in model.named_parameters():
        if 'estimator' not in name:
            print('params {} frozen.'.format(name))
            L.requires_grad = False

def train_bottleneck_component(train_loader, model, criterion, optimizer, epoch, component, l=1):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    norm_losses = AverageMeter()
    orth_losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        target = target.cuda(async=True).long()
        input_var = torch.autograd.Variable(input).cuda()

        #target_var = torch.autograd.Variable(target)
        target_var = torch.autograd.Variable(target).long()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        norm_loss = l*compute_norm_loss(model.estimator.weight)
        
        #orth_loss = l*ortho_cost_1(model.estimator.weight)
        orth_loss = l*FIP_cost(model.estimator.weight)
        
        joint_loss = loss + norm_loss + orth_loss
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        joint_loss.backward()
        
        # zero grad for all but the current component
        for p in model.parameters():
            if p.requires_grad:
                for row in range(p.grad.shape[0]):
                    if row != component:
                        p.grad[row] = 0
        
        optimizer.step()

        output = output.float()
        loss = loss.float()
        
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.data[0], input.size(0))
        norm_losses.update(norm_loss.data[0], input.size(0))
        orth_losses.update(orth_loss, input.size(0))
        top1.update(prec1[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Norm Loss {norm_loss.val:.4f} ({norm_loss.avg:.4f})\t'
                  'Orth Loss {orth_loss.val:.4f} ({orth_loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), loss=losses, norm_loss=norm_losses, 
                      orth_loss=orth_losses, top1=top1))

compute_norm_loss = lambda W : torch.abs(1-W.norm(dim=1)).sum()
ortho_cost_1 = lambda W : torch.abs(torch.matmul(W,W.t()) - torch.eye(W.size(0))).sum()

cos = torch.nn.CosineSimilarity(dim=0)

def FIP_cost(W):
    cost = 0
    for m in range(len(W)):
        for n in range(len(W)):
            if m != n:
                cost_mn = cos(W[m],W[n]).abs()
                cost+=cost_mn
    return cost


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

    targets = []
    outputs = []
    for i, (input, target) in enumerate(loader):
        #outputs['inputs'].append(input.detach().cpu().numpy())
        #outputs['labels'].append(target.detach().cpu().numpy())

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
        
        # collect targets and outputs
        targets.append(target.detach().cpu().numpy())
        outputs.append(output.detach().cpu().numpy())

        save_template = '{image_set}/{image_set}-ep_{epoch}-step_{i}-acc_{top1.avg:.2f}-'.format(
                image_set=image_set, 
                epoch=epoch, 
                i=i, 
                top1=top1
        )

        #save_features(outputs, save_template)
        #clear(outputs)

        if i % args.print_freq == 0:
            pass
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))

    # compute per class loss
    targets = np.concatenate(targets)
    outputs = np.concatenate(outputs)



    class_acc = []

    for label in np.unique(targets):
        label_mask = [target==label for target in targets]
        target_i = torch.tensor(targets[label_mask])
        output_i = torch.tensor(outputs[label_mask])
        accuracy_i = accuracy(output_i, target_i)[0].item()
        class_acc.append(accuracy_i)

    return top1.avg, losses.avg, class_acc

def catch(filepath, target, ind=1, verbose=False):
    filepath_ = filepath.replace('/','-')
    parts = filepath_.split('-')
    match = [part for part in parts if target in part]
    if len(match) == 1:
        return match[0].split('_')[ind]
    else:
        if verbose:
            print('target {} not found in filepath {}'.format(target,filepath))
        return None
    
def get_U(base):
    U_paths = [os.path.join(base, path) for path in os.listdir(base) if 'U_' in path]
    Ks = [U_path.split('/')[-1] for U_path in U_paths]
    Ks = [k.split('.')[0] for k in Ks]
    Ks = [np.int(k.split('_')[1]) for k in Ks]
    maxK = max(Ks)
    U_path = [U_path for U_path in U_paths if str(maxK) in U_path][0]
    U = h5.File(U_path, 'r')['obj_arr'].value
    return U, maxK

def get_SVD_sets(base):
    dset_paths = [os.path.join(base, path) for path in os.listdir(base) if 'dimest' in path]
    dimests = [catch(dataset,'dimest') for dataset in dset_paths]
    return dimests, dset_paths

def create_loader(path, batch_size):
    images = h5.File(path, 'r')['obj_arr'].value
    
    label_path = os.path.join(path.split('dimest')[0], 'labels.h5')
    labels = h5.File(label_path, 'r')['obj_arr'].value

    X = torch.Tensor(images)
    Y = torch.Tensor(labels)

    train_dataset = torch.utils.data.TensorDataset(X, Y)
#     val_dataset = torch.utils.data.TensorDataset(X, Y)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader

class Extractor(object):
    def __init__(self, name):
        self.name = name
        outputs[self.name]=[]

    def extract(self, module, input, output):
        outputs[self.name].append(output.detach().cpu().numpy())

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

def get_layers(tree):
    children = list(tree.children())
    if len(children) > 0:
        return [get_layers(child) for child in children]
    else:
        return tree
    
def flatten(aList):
    t = []
    for i in aList:
        if not isinstance(i, list):
             t.append(i)
        else:
             t.extend(flatten(i))
    return t

def construct_filesystem(args):
    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Check the feature_dir exists or not
    if not os.path.exists(args.feature_dir):
        os.makedirs(args.feature_dir)

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
