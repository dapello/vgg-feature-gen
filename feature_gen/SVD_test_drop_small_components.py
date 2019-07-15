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
parser.add_argument('--batchnorm', default=0, type=int,
                    help='If true, construct networks with batchnorm.')
parser.add_argument('--dropout', default=0.0, type=float,
                    help='probability of dropout')
parser.add_argument('--dataaug', action='store_true',
                    help='whether or not to train with data augmentation')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
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

    # this is where the SVD compressed representations live
    SVD_base_path = args.SVD_base_path
    SVD_rep_paths = os.listdir(SVD_base_path)
    #layer_names = ['features_1_ReLU','features_8_ReLU','features_16_MaxPool2d', 'classifier_1_ReLU']
    #layer_names.reverse()
    print('SVD_rep_paths', SVD_rep_paths) 
    
    results = []

    # initialize original, trained model, get it's best score
    best_prec1, best_loss_avg = sample(train_loader, model, criterion, args.start_epoch, 'Full')

    # next, loop over feature layers in network
    for part_of_model, part_name in [[model.features, 'features'],[model.classifier, 'classifier']]:
        for i, L in enumerate(part_of_model):
            # generate original naming scheme
            name = part_name+'_'+str(i)+"_"+str(L)
            name_to_match = name.split('(')[0]
            print('name_to_match ',name_to_match )
            # check if the SVD_rep file exists
            #if name_to_match in SVD_rep_paths:
            #if name_to_match in ['features_29_ReLU','features_30_MaxPool2d', 'classifier_1_ReLU', 'classifier_3_ReLU']:
            if name_to_match in ['classifier_1_ReLU', 'classifier_3_ReLU']:
            #if name_to_match in ['features_27_ReLU','features_25_ReLU','features_23_MaxPool2d','features_11_ReLU','features_8_ReLU']:
            # if name_to_match in layer_names:
                print('caught',name_to_match, i)

                ## fetch U, k
                SVD_layer_path = os.path.join(SVD_base_path, name_to_match)
                U, maxK = get_U(SVD_layer_path)
                print('max K for layer:', maxK)
                # pull out as args?
                start = maxK # max rank of approximation
                range_ = [1.0,2.0]
                hist = [0.0]
                #scores = [100.0]
                dist = 100.0 

                k = maxK
                
                # need to get max prec1 first, define compare. . 
                limit = 15 
                counter = 0

                print('Entering while loop')
                while not (range_[0]<dist<range_[1]):
                    print('k=',k)
                    counter +=1
                    if counter > limit or k > maxK:
                        break
                    # create model with rank k approximator inserted at layer i
                    model_ = VGG_pc_bottleneck(model, i, U[:k], loc=part_name)
                    model_.cuda()

                    # sample model loss at rank k approximation for layer
                    prec1, loss_avg = sample(train_loader, model_, criterion, args.start_epoch, k)
                    dist = 100*(best_prec1[0].item() - prec1[0].item())/best_prec1[0].item()
                    
                    print('dist:',dist)

                    hist.append(k)
                    #scores.append(prec1[0].item())
                    results.append([name_to_match, k, prec1[0].item(), loss_avg[0].item(), str(list(model_.children())), dist, best_prec1[0].item(), U.shape[1]])
                
                    if range_[0]>dist:
                        k -= np.int(np.abs(hist[-1] - hist[-2])/2)
                    if range_[1]<dist:
                        k += np.int(np.abs(hist[-1] - hist[-2])/2)

                ## scan dropping every nth component from U[:k]. save n to correlate with EV
                drop_n = lambda U, n, k : np.concatenate([U[:n,:],U[n+1:k,:]])
                
                drop_component_results = []
                sample_n = 100
                for n in np.arange(0,sample_n):
                    # n is the row to drop -- sample_n rows from 0 to ~k
                    # n = int(n*(k/sample_n))
                    #if n > k:
                    #    break

                    print('drop component {} of {}'.format(n, name_to_match))

                    model_ = VGG_pc_bottleneck(model, i, drop_n(U,n,k), loc=part_name)
                    model_.cuda()

                    # sample model loss at rank k approximation for layer
                    prec1, loss_avg = sample(train_loader, model_, criterion, args.start_epoch, k)
                    dist = 100*(best_prec1[0].item() - prec1[0].item())/best_prec1[0].item()
                    
                    print('dist:',dist)

                    drop_component_results.append([name_to_match, k, n, prec1[0].item(), loss_avg[0].item(), str(list(model_.children())), dist, best_prec1[0].item(), U.shape[1]])


                drop_last_results = []
                for n in np.arange(1,sample_n):
                    # n is the row to drop -- sample_n rows from 0 to ~k
                    # n = int(n*(k/sample_n))
                    #if n > k:
                    #    break

                    print('drop component {} of {}'.format(n, name_to_match))

                    model_ = VGG_pc_bottleneck(model, i, U[:n], loc=part_name)
                    model_.cuda()

                    # sample model loss at rank k approximation for layer
                    prec1, loss_avg = sample(train_loader, model_, criterion, args.start_epoch, k)
                    dist = 100*(best_prec1[0].item() - prec1[0].item())/best_prec1[0].item()
                    
                    print('dist:',dist)

                    drop_last_results.append([name_to_match, k, n, prec1[0].item(), loss_avg[0].item(), str(list(model_.children())), dist, best_prec1[0].item(), U.shape[1]])

                
                pathname = os.path.join(args.results_dir,'SVD_test_play_allsmall100_{}.h5'.format(name_to_match)) 
                print('saving results at ',pathname)
                f = h5.File(pathname, 'w')
                # f.create_dataset('search_results', data=np.array(results))
                f.create_dataset('drop_component_results', data=np.array(drop_component_results))
                f.create_dataset('drop_last_results', data=np.array(drop_last_results))
                f.close()
                print('results saved')

class VGG_pc_bottleneck(nn.Module):
    ''' 
    Expects a VGG like network split into features1, features2, classifier1, and classifier2,
    and an estimator (ie U.T@U, from the SVD of the representations). loc specifies if the estimator
    is to be put into the feature layers or classifier.
    '''
    def __init__(self, model, i, U, loc='features'):
        super(VGG_pc_bottleneck, self).__init__()
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
        
        if U.shape[1]>U.shape[0]:
            U = U.T

        print('working U shape:', U.shape)

        self.estimator = nn.Sequential(
            nn.Linear(in_features=U.shape[0], out_features=U.shape[1], bias=False),
            nn.Linear(in_features=U.shape[1], out_features=U.shape[0], bias=False)
        )
        self.init_estimator(U)
        self.loc = loc

        for m in self.modules():
            m.requires_grad = False

    def init_estimator(self, U):
        for m in self.estimator:
            if m.weight.shape[0] == U.shape[0]:
                m.weight = torch.nn.parameter.Parameter(data=torch.Tensor(U))
            if m.weight.shape[1] == U.shape[0]:
                m.weight = torch.nn.parameter.Parameter(data=torch.Tensor(U.T))
        
    def forward(self, x): 
        x = self.features1(x)
        
        # if estimator goes in between features, this route is called
        if self.loc == 'features':
            size = x.size()
            x = x.view(x.size(0), -1)
            x = self.estimator(x)
            x = x.view(size)
            
        x = self.features2(x)
        x = x.view(x.size(0), -1) 
        x = self.classifier1(x)
       
        # else, estimator goes into the classifier, this route is called 
        if self.loc == 'classifier':
            x = self.estimator(x)
        
        x = self.classifier2(x)
        
        return x

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

    return top1.avg, losses.avg

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
