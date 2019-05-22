import os
import argparse

import matplotlib.pyplot as plt
import h5py as h5
import numpy as np

parser = argparse.ArgumentParser(description='SVD decompose and reconstruct representations')
parser.add_argument('--factor', dest='factor', default=1, type=int,
                    help='factor to downsample number of representations by')
parser.add_argument('--feature_path', dest='feature_path', type=str,
                    help='The directory used to save the extracted features')
parser.add_argument('--SVD-base-path', dest='SVD_base_path',
                    help='The directory containing the SVD compressed representations',
                    default='SVD_compressed', type=str)


def main():
    args = parser.parse_args()
    
    feature_name = args.feature_path.split('-')[3].split('(')[0]
    target_dir = os.path.join(args.SVD_base_path, feature_name)
    print('SVD processing', args.feature_dir)
    print('target dir', target_dir)

    # load feature representations
    reps = h5.File(args.feature_path, 'r')['obj_arr'].value[::args.factor,:]
    og_shape = reps.shape
    print('original images shape:', og_shape)
    
    # shape rep must be in for network input
    target_shape = eval(catch(args.feature_path, 'ogshape'))
    print('original images shape:', target_shape)
    
    # reps should be num stimuli x num features
    #reps = reps.reshape(-1, og_shape[-1])
    
    return

    # center features, storing original mean for reconstruction
    og_mean = reps.mean(axis=0)
    reps -= og_mean
    
    print('running SVD')
    U, S, V = np.linalg.svd(reps)
    
    print('SVD complete')
    
    save_h5(os.path.join(target_dir, 'U.h5'),U)
    save_h5(os.path.join(target_dir, 'S.h5'),S)
    save_h5(os.path.join(target_dir, 'V.h5'),V)
    
    dim_ests = {
        'PR' : int(round(compute_pr(S))),
        'EV85' : compute_EVD(S, .85),
        'EV90' : compute_EVD(S, .90),
        'EV95' : compute_EVD(S, .95),
        'EV99' : compute_EVD(S, .99),
        # 'NCEV' : NC_EVD(reps,S),
        'Rank' : compute_rank(S, reps),
        'Half' : int(reps.shape[0]/2),
        'Full' : reps.shape[0]
    } 
    
    del reps
    
    N = 10
    for key in dim_ests.keys():
        save_path = os.path.join(target_dir, 'dimest_{}-components_{}.h5'.format(key, dim_ests[key]))
        # process images, reducing to estimated dimensionality
        reps_hat = SVD_compress(U,S,V, dim_ests[key])
        reps_hat += og_mean 
        reps_hat = reps_hat.reshape(target_shape)
        print('save path: {}, images_hat shape {}'.format(save_path, images_hat))
        save_h5(save_path, images_hat)

# various helper functions
compute_pr = lambda S : (sum(S**2)**2)/sum(S**4)
compute_EVD = lambda S, t : sum(np.cumsum((S**2)/sum(S**2))<t)
compute_rank = lambda S, reps : sum(S>(S.max()*max(reps.shape)*np.finfo(reps.dtype).eps))
SVD_compress = lambda U, S, V, s : U[:,:s]@np.diag(S)[:s,:]@V

## implement NC EVD
def NC_EVD(images, S):
    imshuff = images.reshape(-1)
    imshuff = np.random.permutation(imshuff)
    imshuff = imshuff.reshape(images.shape[0],3072)

    Un, Sn, Vn = np.linalg.svd(imshuff)
    return sum(S>Sn)


def save_h5(save_path, data):
    f = h5.File(save_path, 'w')
    f.create_dataset('obj_arr', data=data)
    f.close()


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
if __name__ == '__main__':
    main()
