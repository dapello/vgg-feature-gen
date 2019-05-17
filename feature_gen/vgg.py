'''
Modified from https://github.com/pytorch/vision.git
'''
import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

__all__ = ['VGG', 'vgg', 'vgg_s', 'fc', 'resnet']

IN_CHANNELS = 3
CLASSES = 10

class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features, linear_in=512, dropout=0.0, custom_weight_init=None, classes=CLASSES):
        super(VGG, self).__init__()
        self.features = features
        print('dropout = {}'.format(dropout))
        self.classifier = nn.Sequential(
            nn.Linear(linear_in, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, classes),
        )

       # if custom_weight_init == 'orthogonal':
       #     for m in self.modules():
       #         print(m)
       #         if isinstance(m, nn.Conv2d):
       #             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
       #             m.weight.data.normal_(0, math.sqrt(2. / n))
       #             m.bias.data.zero_()
       # elif custom_weight_init == 'original':
       #      # Initialize weights
        for m in self.modules():
            #print(m)
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class VGG_s(nn.Module):
    '''
    VGG model with shallow classifier 
    '''
    def __init__(self, features, linear_in=16384, dropout=0.0, custom_weight_init=False, classes=CLASSES):
        super(VGG_s, self).__init__()
        self.features = features
        print('dropout = {}'.format(dropout))
        self.classifier = nn.Sequential(
            nn.Linear(linear_in, classes), # 65536 is number of inputs
        )

        #if custom_weight_init:
             # Initialize weights
        for m in self.modules():
            #print(m)
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class FC(nn.Module):
    '''
    fully connected model 
    '''
    def __init__(self, features, linear_in=256, dropout=0.0, classes=CLASSES):
        super(FC, self).__init__()
        self.features = features
        print('dropout = {}'.format(dropout))
        self.classifier = nn.Sequential(
            nn.Linear(linear_in, classes), 
        )

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.features(x)
        x = self.classifier(x)
        return x

## resnet construction
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        )

        self.shortcut = nn.Sequential()
        self.ReLU = nn.ReLU(inplace=True)
        
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                #nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.conv(x)
        out += self.shortcut(x)
        out = self.ReLU(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False),
        )

        self.shortcut = nn.Sequential()
        self.ReLU = nn.ReLU(inplace=True)
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                #nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.conv(x)
        out += self.shortcut(x)
        out = self.ReLU(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        #self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AvgPool2d(4)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        #out = F.relu(self.bn1(self.conv1(x)))
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

#class BasicBlock(nn.Module):
#    expansion = 1
#
#    def __init__(self, in_planes, planes, stride=1):
#        super(BasicBlock, self).__init__()
#        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#        #self.bn1 = nn.BatchNorm2d(planes)
#        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#        #self.bn2 = nn.BatchNorm2d(planes)
#
#        self.shortcut = nn.Sequential()
#        if stride != 1 or in_planes != self.expansion*planes:
#            self.shortcut = nn.Sequential(
#                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
#                #nn.BatchNorm2d(self.expansion*planes)
#            )
#
#    def forward(self, x):
#        #out = F.relu(self.bn1(self.conv1(x)))
#        out = F.relu(self.conv1(x))
#        #out = self.bn2(self.conv2(out))
#        out = self.conv2(out)
#        out += self.shortcut(x)
#        out = F.relu(out)
#        return out
#
#
#class Bottleneck(nn.Module):
#    expansion = 4
#
#    def __init__(self, in_planes, planes, stride=1):
#        super(Bottleneck, self).__init__()
#        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
#        #self.bn1 = nn.BatchNorm2d(planes)
#        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#        #self.bn2 = nn.BatchNorm2d(planes)
#        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
#        #self.bn3 = nn.BatchNorm2d(self.expansion*planes)
#
#        self.shortcut = nn.Sequential()
#        if stride != 1 or in_planes != self.expansion*planes:
#            self.shortcut = nn.Sequential(
#                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
#                #nn.BatchNorm2d(self.expansion*planes)
#            )
#
#    def forward(self, x):
#        #out = F.relu(self.bn1(self.conv1(x)))
#        out = F.relu(self.conv1(x))
#        #out = F.relu(self.bn2(self.conv2(out)))
#        out = F.relu(self.conv2(out))
#        #out = self.bn3(self.conv3(out))
#        out = self.conv3(out)
#        out += self.shortcut(x)
#        out = F.relu(out)
#        return out
#
#
#class ResNet(nn.Module):
#    def __init__(self, block, num_blocks, num_classes=10):
#        super(ResNet, self).__init__()
#        self.in_planes = 64
#
#        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
#        #self.bn1 = nn.BatchNorm2d(64)
#        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
#        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#        self.linear = nn.Linear(512*block.expansion, num_classes)
#
#    def _make_layer(self, block, planes, num_blocks, stride):
#        strides = [stride] + [1]*(num_blocks-1)
#        layers = []
#        for stride in strides:
#            layers.append(block(self.in_planes, planes, stride))
#            self.in_planes = planes * block.expansion
#        return nn.Sequential(*layers)
#
#    def forward(self, x):
#        #out = F.relu(self.bn1(self.conv1(x)))
#        out = F.relu(self.conv1(x))
#        out = self.layer1(out)
#        out = self.layer2(out)
#        out = self.layer3(out)
#        out = self.layer4(out)
#        out = F.avg_pool2d(out, 4)
#        out = out.view(out.size(0), -1)
#        out = self.linear(out)
#        return out

def make_layers(cfg, arch='vgg', batchnorm=False):
    layers = []
    in_channels = IN_CHANNELS
    if arch == 'vgg':
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batchnorm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v

    elif arch == 'fc':
        input_size = 784   ## yucky hard coded for MNIST
        for v in cfg:
            linear = nn.Linear(input_size, v)
            layers += [linear, nn.ReLU(inplace=True)]
            input_size = v

    return nn.Sequential(*layers)

cfg = {
    'vgg1' : [64, 'M'],
    'vgg2' : [64, 64, 'M'],
    'vgg3' : [64, 64, 'M', 128, 'M'],
    'vgg4' : [64, 64, 'M', 128, 128, 'M'],
    'vgg5' : [64, 64, 'M', 128, 128, 'M', 256, 'M'],
    'vgg6' : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M'],
    'vgg7' : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 'M'],
    'vgg8' : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M'],
    'vgg9' : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 'M'],
    'vgg10': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg11': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg12': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    
    'vggm2': [64, 'M', 128, 'M'],
    'vggm3': [64, 'M', 128, 'M', 256, 'M'],
    'vggm4': [64, 'M', 128, 'M', 256, 'M', 512, 'M'],
    'vggm5': [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M'],
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],

    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],

    'stuck': [16, 'M', 32, 'M', 32, 'M'],
    'fc5': [256, 256, 256, 256, 256],
    'fc7w': [512, 512, 512, 512, 512],
    'resnet18': [2,2,2,2],
    'resnet34': [3,4,6,3],
    'resnet50': [3,4,6,3]
}

in_num = {
    'vgg1' : 16384, 
    'vgg2' : 16384, 
    'vgg3' : 8192, 
    'vgg4' : 8192, 
    'vgg5' : 4096, 
    'vgg6' : 4096, 
    'vgg7' : 2048, 
    'vgg8' : 2048, 
    'vgg9' : 512, 
    'vgg10': 512, 
    'vgg11': 512, 
    'vgg12': 512, 
    'vgg13': 512, 
    
    'vggm2': 8192, 
    'vggm3': 4096, 
    'vggm4': 2048, 
    'vggm5': 512, 
    'vgg11': 512, 
    
    'vgg16': 512, 
    'vgg19': 512, 
    'stuck': 512, 
    'fc5': 256, 
    'fc7w': 512
}

def vgg(model, classes=CLASSES, batchnorm=False, dropout=0.0):
    layers = make_layers(cfg[model], batchnorm=batchnorm)
    return VGG(layers, linear_in=in_num[model], dropout=dropout, classes=classes)

def vgg_s(model, classes=CLASSES, batchnorm=False, dropout=0.0):
    return VGG_s(make_layers(cfg[model], batchnorm=batchnorm), linear_in=in_num[model], dropout=dropout, classes=classes)

def fc(model, classes=CLASSES, batchnorm=False, dropout=0.0):
    return FC(make_layers(cfg[model], arch='fc', batchnorm=batchnorm), linear_in=in_num[model], dropout=dropout, classes=classes)

def resnet(model, classes=CLASSES, batchnorm=False, dropout=0.0):
    if '50' in model:
        return ResNet(Bottleneck, cfg[model], num_classes=classes)
    else:
        return ResNet(BasicBlock, cfg[model], num_classes=classes)
