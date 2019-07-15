import os
import argparse

import matplotlib.pyplot as plt
import h5py as h5
import numpy as np

parser = argparse.ArgumentParser(description='SVD decompose and reconstruct representations')
parser.add_argument('--factor', dest='factor', default=1, type=int,
                    help='factor to downsample number of representations by')
parser.add_argument('--feature-path', dest='feature_path', type=str,
                    help='The directory used to save the extracted features')
parser.add_argument('--SVD-base-path', dest='SVD_base_path',
                    help='The directory containing the SVD compressed representations',
                    default='SVD_compressed', type=str)
parser.add_argument('--layer-types', '-lt', dest='layer_types', default=[], action='append', 
                    help='select layer types to be processed. if left undefined, allow all layer types.')


def main():
    args = parser.parse_args()

    # we'll just do the 1 file case for now.. 
    files = os.listdir(args.feature_path)
    file_ = files[0]
    print('files:', files)
    feature_path = os.path.join(args.feature_path,file_) 
    
    feature_name = file_.split('-')[3].split('(')[0]
    
    # check if there are restricted layer types specified
    if len(args.layer_types) > 0:
        # if the current feature is specified, pass. if not, return
        if feature_name.split('_')[-1] in args.layer_types:
            pass
        else:
            print('feature {} not specified for processing.'.format(feature_name)) 
            return

    target_dir = os.path.join(args.SVD_base_path, feature_name)
    
    print('SVD processing', feature_path)
    print('target dir', target_dir)
    construct_filesystem(target_dir)

    # load feature representations
    reps = h5.File(feature_path, 'r')['obj_arr'].value
    cur_shape = reps.shape
    print('current representation shape:', cur_shape)
    
    labels = np.array([np.repeat(label, cur_shape[1]) for label in range(len(reps))]).reshape(-1)
    print('shape of labels', labels.shape)
    save_h5(os.path.join(target_dir,'labels.h5'), labels)
    
    # reps should be num stimuli x num features for processing
    reps = reps.reshape(-1, cur_shape[-1]).astype('float32')[::args.factor,:]

    # shape rep must ultimately be in format for network input
    target_shape = list(eval(catch(feature_path, 'ogshape')))
    target_shape[1] = reps.shape[0]
    target_shape = target_shape[1:]
    print('target representation shape:', target_shape)
    
    # center features, storing original mean for reconstruction
    og_mean = reps.mean(axis=0)
    reps -= og_mean
    
    U_path = os.path.join(target_dir, 'U.h5')
    S_path = os.path.join(target_dir, 'S.h5')
    V_path = os.path.join(target_dir, 'V.h5')
    
    if (os.path.isfile(U_path))&(os.path.isfile(S_path))&(os.path.isfile(V_path)):
        print('loading computed U S V')
        U = h5.File(U_path, 'r')['obj_arr'].value
        S = h5.File(S_path, 'r')['obj_arr'].value
        V = h5.File(V_path, 'r')['obj_arr'].value
    else:
        print('running SVD on reps, shape:', reps.shape)
        U, S, V = np.linalg.svd(reps)
        save_h5(U_path,U)
        save_h5(S_path,S)
        save_h5(V_path,V)
        print('SVD complete')

    dim_ests = {
        '1' : 1,
        '10' : 10,
        '100' : 100,
        'PR' : int(np.round(compute_pr(S))),
        'EV85' : compute_EVD(S, .85),
        'EV90' : compute_EVD(S, .90),
        'EV95' : compute_EVD(S, .95),
        'EV99' : compute_EVD(S, .99),
        # 'NCEV' : NC_EVD(reps,S),
        'Rank' : compute_rank(S, reps),
        'Half' : int(reps.shape[1]/2),
        'Full' : reps.shape[1]
    } 
    print('dimensionality estimates:', dim_ests) 
    del reps
    
    N = 10
    for key in dim_ests.keys():
        save_path = os.path.join(target_dir, 'dimest_{}-components_{}.h5'.format(key, dim_ests[key]))
        # process images, reducing to estimated dimensionality
        reps_hat = SVD_compress(U,S,V, dim_ests[key])
        reps_hat += og_mean 
        reps_hat = reps_hat.reshape(target_shape)
        print('save path: {}, images_hat shape {}'.format(save_path, reps_hat.shape))
        save_h5(save_path, reps_hat)

# various helper functions
compute_pr = lambda S : (sum(S**2)**2)/sum(S**4)
compute_EVD = lambda S, t : sum(np.cumsum((S**2)/sum(S**2))<t)
compute_rank = lambda S, reps : sum(S>(S.max()*max(reps.shape)*np.finfo(reps.dtype).eps))
#SVD_compress = lambda U, S, V, s : U[:,:s]@np.diag(S)[:s,:]@V
SVD_compress = lambda U, S, V, s : np.dot(np.dot(U[:,:s],np.diag(S)[:s,:]),V)

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

def construct_filesystem(target_path):
    # Check the save_dir exists or not
    if not os.path.exists(target_path):
        os.makedirs(target_path)

if __name__ == '__main__':
    main()
