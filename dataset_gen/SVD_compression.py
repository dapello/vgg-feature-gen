import matplotlib.pyplot as plt
import h5py as h5
import numpy as np

import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

factor = 1 ## take every n images

compute_pr = lambda S : (sum(S**2)**2)/sum(S**4)
compute_EVD = lambda S, t : sum(np.cumsum((S**2)/sum(S**2))<t)
compute_rank = lambda images : np.linalg.matrix_rank(images)

## implement NC EVD
def NC_EVD(images, S):
    imshuff = images.reshape(-1)
    imshuff = np.random.permutation(imshuff)
    imshuff = imshuff.reshape(images.shape[0],3072)

    Un, Sn, Vn = np.linalg.svd(imshuff)
    return sum(S>Sn)

SVD_compress = lambda U, S, V, s : U[:,:s]@np.diag(S)[:s,:]@V

## recover original image shape after flattening
format_ims = lambda images : images.reshape(-1, 3, 32, 32)


def scale(images):
    images -= np.min(images)
    images /= np.max(images)
    return images

def save_h5(save_path, data):
    f = h5.File(save_path, 'w')
    f.create_dataset('obj_arr', data=data)
    f.close()

train_transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])
            
#val_transform=transforms.Compose([
#    transforms.ToTensor(),
#    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
#])


train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
#val_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=val_transform)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=50000, 
    shuffle=False
)

#val_loader = torch.utils.data.DataLoader(
#    val_dataset,
#    batch_size=10000, 
#    shuffle=False
#)


# load images
for i, (images, targets) in enumerate(train_loader):
    images = images.detach().numpy().reshape(50000,-1)
    
    # downsample by a factor
    images = images[::factor,:]
    images.shape

U, S, V = np.linalg.svd(images)

dim_ests = {
    'pr' : int(compute_pr(S)+1),
    'EV85' : compute_EVD(S, .85),
    'EV90' : compute_EVD(S, .90),
    'EV95' : compute_EVD(S, .95),
    'EV99' : compute_EVD(S, .99),
    'NCEV' : NC_EVD(images,S),
    'Rank' : np.linalg.matrix_rank(images),
    'Half' : int(3072/2),
    'Full' : 3072
} 

del images

N = 10
for key in dim_ests.keys():
    save_path = 'CIFAR100-{}_{}components.h5'.format(key, dim_ests[key])
    
    # process images, reducing to estimated dimensionality
    images_hat = SVD_compress(U,S,V, dim_ests[key])
    images_hat = format_ims(images_hat)
    save_h5(save_path, images_hat)
    
    images_hat = scale(images_hat)
    f, axarr = plt.subplots(1,N, figsize=(20,20))
    for n in range(N):
        axarr[n].imshow(np.transpose(images_hat[n],[1,2,0]))
    #     axarr[n].axis('off')
        axarr[n].set_yticklabels([])
        axarr[n].set_xticklabels([])

    ax = axarr[0]
    title = '{}:{} components'.format(key, dim_ests[key])
    ax.set_ylabel(title, fontsize=12)
    f.savefig(title.replace(" ", ""))
    del images_hat
