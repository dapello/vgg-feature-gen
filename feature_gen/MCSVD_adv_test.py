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

from art.attacks.fast_gradient import FastGradientMethod
from art.classifiers.pytorch import PyTorchClassifier

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
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
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
    best_prec1, best_loss_avg, best_class_acc = sample(train_loader, model, criterion, args.start_epoch, 'Full')

    og_x_tests, og_x_test_advs, og_y_tests = gen_adv_inputs(model, val_loader, criterion, args)
    og_mean_results = adv_test(model, og_x_tests, og_x_test_advs, og_y_tests, criterion, args)
    print('adv test on original model')
    print(og_mean_results)

    # next, loop over feature layers in network
    for part_of_model, part_name in [[model.features, 'features'],[model.classifier, 'classifier']]:
        for i, L in enumerate(part_of_model):
            # generate original naming scheme
            name = part_name+'_'+str(i)+"_"+str(L)
            name_to_match = name.split('(')[0]
            print('name_to_match ',name_to_match )
            # check if the SVD_rep file exists
            #if name_to_match in SVD_rep_paths:
            #if name_to_match in ['features_1_ReLU','features_3_ReLU','features_13_ReLU','features_15_ReLU','features_23_MaxPool2d']:
            #if name_to_match in ['features_6_ReLU','features_8_ReLU','features_11_ReLU','features_18_ReLU','features_20_ReLU','features_22_ReLU','features_25_ReLU','features_27_ReLU','features_29_ReLU','features_4_MaxPool2d','features_9_MaxPool2d','features_16_MaxPool2d']:
            if name_to_match in ['features_1_ReLU']:
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
                k = 2000
                
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
                    prec1, loss_avg, class_acc = sample(train_loader, model_, criterion, args.start_epoch, k)
                    dist = 100*(best_prec1[0].item() - prec1[0].item())/best_prec1[0].item()
                    
                    print('dist:',dist)

                    hist.append(k)
                    #scores.append(prec1[0].item())
                    results.append([
                        name_to_match, 
                        k, 
                        prec1[0].item(), 
                        loss_avg[0].item(), 
                        dist, 
                        best_prec1[0].item(), 
                        str(class_acc),
                        str(best_class_acc),
                        U.shape[1]
                    ])

                    if range_[0]>dist:
                        k -= np.int(np.abs(hist[-1] - hist[-2])/2)
                    if range_[1]<dist:
                        k += np.int(np.abs(hist[-1] - hist[-2])/2)

                # try original adv images on bottlenecked model
                og_mean_results_bottleneck = adv_test(model_, og_x_tests, og_x_test_advs, og_y_tests, criterion, args)
                print('model adv images on bottlenecked model')
                print(og_mean_results_bottleneck)

                # and generate adv examples for bottlenecked model
                x_tests, x_test_advs, y_tests = gen_adv_inputs(model_, val_loader, criterion, args)
                mean_results_bottleneck = adv_test(model_, x_tests, x_test_advs, y_tests, criterion, args)
                print('bottlenecked model adv images on bottlenecked model')
                print(mean_results_bottleneck)

                pathname = os.path.join(args.results_dir,'MCSVD_adv_test_{}.h5'.format(name_to_match)) 
                print('saving results at ',pathname)
                f = h5.File(pathname, 'w')
                f.create_dataset('search_results', data=np.array(results))
                f.create_dataset('adv_test', data=np.array([og_mean_results, og_mean_results_bottleneck, mean_results_bottleneck]))
                #f.create_dataset('drop_last_results', data=np.array(drop_last_results))
                #f.close()
                #print('results saved')

def gen_adv_inputs(model, loader, criterion, args, epsilon=0.2):
    """take a model and dataloader and make a set of adversarial images using ART's fast gradient method."""
    x_tests, x_test_advs, y_tests = [], [], []
    for i, (x_test, y_test) in enumerate(loader):
        print('Computing adversarial batch: {}'.format(i+1))
        x_test = x_test.detach().cpu().numpy()
        y_test = y_test.detach().cpu().numpy()
        print('shape:', x_test.shape)
        if i == 0:
            # in first draw, specify the adversarial generator 
            model_parameters = filter(lambda p: p.requires_grad,model.parameters())
            optimizer = torch.optim.SGD(model_parameters, args.lr,
                    momentum=args.momentum, weight_decay=args.weight_decay)

            classifier = PyTorchClassifier(model=model, loss=criterion, optimizer=optimizer, input_shape=tuple(x_test.shape[1:]), nb_classes=args.classes) 
            adv_crafter = FastGradientMethod(classifier, eps=epsilon)

        x_test_adv = adv_crafter.generate(x=x_test)

        x_tests.append(x_test)
        y_tests.append(y_test)
        x_test_advs.append(x_test_adv)
    
    x_tests = np.array(x_tests)
    x_test_advss = np.array(x_test_advs)
    y_tests = np.array(y_tests)

    return x_tests, x_test_advs, y_tests

def adv_test(model, x_tests, x_test_advs, y_tests, criterion, args):
    """Run original images and adv images on provided model"""
    #x_tests = x_tests.reshape(-1, args.batch_size, x_tests.shape[1:])
    #x_test_advs = x_test_advs.reshape(-1, args.batch_size, x_test_advs.shape[1:])
    #y_tests = y_tests.reshape(-1, args.batch_size, y_tests.shape[1:])

    model_parameters = filter(lambda p: p.requires_grad,model.parameters())
    optimizer = torch.optim.SGD(model_parameters, args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay)
    
    classifier = PyTorchClassifier(model=model, loss=criterion, optimizer=optimizer, input_shape=tuple(x_tests.shape[2:]), nb_classes=args.classes) 

    results = []

    for x_test, x_test_adv, y_test in zip(x_tests, x_test_advs, y_tests):
        predictions = classifier.predict(x_test)
        accuracy = np.sum(np.argmax(predictions, axis=1) == y_test) / len(y_test)
        #print('Accuracy before attack: {}%'.format(accuracy * 100))
        
        predictions = classifier.predict(x_test_adv)
        adv_accuracy = np.sum(np.argmax(predictions, axis=1) == y_test) / len(y_test)
        #print('Accuracy after attack: {}%'.format(adv_accuracy * 100))

        results.append([accuracy, adv_accuracy])

    results = np.array(results)
    means = results.mean(axis=0)

    return means

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
