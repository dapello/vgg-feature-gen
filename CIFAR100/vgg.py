'''
Modified from https://github.com/pytorch/vision.git
'''
import math

import torch.nn as nn
import torch.nn.init as init

__all__ = [
    'VGG', 'vgg1_sc', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]



class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features, dropout=0.0):
        super(VGG, self).__init__()
        self.features = features
        print('dropout = {}'.format(dropout))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 100),
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


class VGG_sc(nn.Module):
    '''
    VGG model with shallow classifier 
    '''
    def __init__(self, features, dropout=0.0):
        super(VGG_sc, self).__init__()
        self.features = features
        print('dropout = {}'.format(dropout))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(65536, 100), # 65536 is number of inputs
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


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'z': [64],
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg1_sc(dropout=0.5):
    """VGG 1-layer model (configuration "A") with shallow classifier"""
    return VGG_sc(make_layers(cfg['z']), dropout=dropout)

def vgg1_sc_bn(dropout=0.5):
    """VGG 1-layer model (configuration "A") with shallow classifier"""
    return VGG_sc(make_layers(cfg['z'], batch_norm=True), dropout=dropout)

def vgg11(dropout=0.5):
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']), dropout=dropout)


def vgg11_bn(dropout=0.5):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True), dropout=dropout)


def vgg13(dropout=0.5):
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']), dropout=dropout)


def vgg13_bn(dropout=0.5):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True), dropout=dropout)


def vgg16(dropout=0.5):
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']), dropout=dropout)


def vgg16_bn(dropout=0.5):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True), dropout=dropout)


def vgg19(dropout=0.5):
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']), dropout=dropout)


def vgg19_bn(dropout=0.5):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True), dropout=dropout)
