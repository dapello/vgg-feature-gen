'''
Modified from https://github.com/pytorch/vision.git
'''
import math

import torch.nn as nn
import torch.nn.init as init

__all__ = ['VGG', 'vgg', 'vgg_s']

IN_CHANNELS = 3
CLASSES = 10

class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features, linear_in=512, dropout=0.0, classes=CLASSES):
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
         # Initialize weights
        for m in self.modules():
            print(m)
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
    def __init__(self, features, linear_in=16384, dropout=0.0, classes=CLASSES):
        super(VGG_s, self).__init__()
        self.features = features
        print('dropout = {}'.format(dropout))
        self.classifier = nn.Sequential(
            nn.Linear(linear_in, classes), # 65536 is number of inputs
        )
         # Initialize weights
        for m in self.modules():
            print(m)
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batchnorm=False):
    layers = []
    in_channels = IN_CHANNELS
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

    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],

    'stuck': [16, 'M', 32, 'M', 32, 'M']
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
    
    'vgg19': 512, 
    'stuck': 512, 
}

def vgg(model, classes=CLASSES, batchnorm=False, dropout=0.0):
    return VGG(make_layers(cfg[model], batchnorm=batchnorm), linear_in=in_num[model], dropout=dropout, classes=classes)

def vgg_s(model, classes=CLASSES, batchnorm=False, dropout=0.0):
    return VGG_s(make_layers(cfg[model], batchnorm=batchnorm), linear_in=in_num[model], dropout=dropout, classes=classes)
