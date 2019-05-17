import os
import random

import numpy as np
import h5py as h5
from scipy.io import loadmat

import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

__all__ = ['construct_data_loaders']

data_sets = ['CIFAR10', 'CIFAR100', 'EMNIST', 'MNIST', 'HvM64', 'HvM64_V0', 'HvM64_V3', 'HvM64_V6', 'HvM64.r', 'HvM64_V0.r', 'HvM64_V3.r', 'HvM64_V6.r']

def construct_data_loaders(args):
    print("constructing dataset loaders: ", args.dataset)
    print('Data augmentation:', args.dataaug)
    if args.dataset == 'CIFAR10':
        if args.dataaug:
            train_transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        else:
            print('no data augmentation')
            train_transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

        val_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
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
        train_dataset = datasets.MNIST(
            root='./data/',
            train=True,
            transform=transforms.ToTensor(),
            download=True,
        )
        
        val_dataset = datasets.MNIST(
            root='./data/',
            train=False,
            transform=transforms.ToTensor(),
            download=True,
        )

    elif args.dataset == "EMNIST":
        ## load MNIST data train and test sets
        train_dataset = datasets.EMNIST(
            root='./data/',
            train=True,
            transform=transforms.ToTensor(),
            download=True,
            split='balanced'
        )
        
        val_dataset = datasets.EMNIST(
            root='./data/',
            train=False,
            transform=transforms.ToTensor(),
            download=True,
            split='balanced'
        )

    elif "HvM64" in args.dataset:
        trans = transforms.Compose([
            transforms.ToPILImage(),
            #transforms.Resize([224,224]),
            transforms.ToTensor(),
            transforms.Normalize((0.44688916, 0.44688916, 0.44688916),(0.17869806, 0.17869806, 0.17869806))
        ])

        data = loadmat('data/HvM_224px.mat')
        # HvM is the same for all 3 channels, so it was saved as 1 channel to reduce size. reconstruct 3 channels for vgg16
        data['imgs'] = np.array([data['imgs'], data['imgs'], data['imgs']]).transpose(1,2,3,0)
        
        if 'V0' in args.dataset:
            V = 'V0'
            X = torch.Tensor(data['imgs'][data['obj_version']==V].transpose(0,3,1,2))
            Y = data['obj_id'][data['obj_version']==V]
            print('Loading HvM V0 data')
        elif 'V3' in args.dataset:
            V = 'V3'
            X = torch.Tensor(data['imgs'][data['obj_version']==V].transpose(0,3,1,2))
            Y = data['obj_id'][data['obj_version']==V]
            print('Loading HvM V3 data')
        elif 'V6' in args.dataset:
            V = 'V6'
            X = torch.Tensor(data['imgs'][data['obj_version']==V].transpose(0,3,1,2))
            Y = data['obj_id'][data['obj_version']==V]
            print('Loading HvM V6 data')
        else:
            X = torch.Tensor(data['imgs'].transpose(0,3,1,2))
            Y = data['obj_id']
            print('Loading HvM all data')
        
        if '.r' in args.dataset:
            ## rand perm labels? try this!
            rand = torch.randperm(Y.shape[0])
            Y = Y[rand]

        X = torch.stack([trans(x) for x in X])

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
