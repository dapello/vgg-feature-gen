import os

import numpy as np
import h5py as h5
import sys
print("running script: ", sys.argv[0])
print("experiment tag: ", sys.argv[1])
tag = sys.argv[1]

print("sort criteria: ", sys.argv[2])
sortBy = sys.argv[2]

print("seed: ", sys.argv[3])
seed = int(sys.argv[3])

print("subfolders: ", sys.argv[4])
NUM_SUB_TARGET_DIRS = int(sys.argv[4])
SUB_TARGET_DIRS = [str(i) for i in range(NUM_SUB_TARGET_DIRS)]
DATA_DIR = './{}/'.format(tag)
TARGET_DIR = tag.replace('features', "formatted_features-sort_{}/".format(sortBy))

def main():
    if not os.path.exists(TARGET_DIR):
        print('making dir: {}'.format(TARGET_DIR))
        os.makedirs(TARGET_DIR)

    # DATA_DIR contains folders for train and val; process them independently. 
    # (with the exception of the down sampling; downsampler selects the same features across train and val)
    all_data_folders = os.listdir(DATA_DIR)
    for folder in all_data_folders:
        j = 0
        
        data_dir = DATA_DIR+folder+'/'
        all_data_paths = os.listdir(data_dir)
        layer_ids = np.unique([path.split('-')[3] for path in all_data_paths if ('features' in path or 'classifier' in path)])
        epochs = np.unique([path.split('-')[1] for path in all_data_paths if ('features' in path or 'classifier' in path)])
        print("processing {} for epochs: {}, at layers: {} ".format(folder, epochs, layer_ids))
        
        last_layer_data = load_and_sort(data_dir, ['-ep_300-', 'out_features=100,'])
        label_data = load_and_sort(data_dir, ['-ep_300-', 'labels'])
        if sortBy == 'betasoftmax':
            # sort by scaled softmax "confidence" estimate
            beta = 0.4
            confs, orders = get_conf_order(last_layer_data, beta)
        elif sortBy == 'reversebetasoftmax':
            beta = 0.4
            confs, orders = get_conf_order(last_layer_data, beta)
            orders = np.flip(orders,1)
        elif sortBy == 'random':
            orders = get_rand_order(last_layer_data)

        
        input_data = load_and_sort(data_dir, ['-ep_0-', 'inputs'], downsample=True)

        conf_ordered_input_data = apply_order(input_data, orders)

        target_dir = TARGET_DIR+SUB_TARGET_DIRS[j%len(SUB_TARGET_DIRS)]+'/'
        if not os.path.exists(target_dir):
            print('making dir: {}'.format(target_dir))
            os.makedirs(target_dir)

        new_file = folder+'-input.h5' 
        new_path = target_dir+new_file
        f = h5.File(new_path, 'w')
        f.create_dataset('obj_arr', data=conf_ordered_input_data)
        f.close()
        
        j += 1
        # for all unique layers, load the data (concatenating steps across the epochs), apply downsample and sort, and save.
        for epoch in epochs:
            epoch = "-"+epoch+"-"
            print(epoch)
            for i, layer_id in enumerate(layer_ids):
                # appending '_' stops features_1 from matching features_11
                print(layer_id)
                layer_paths = get_paths(data_dir, [epoch, layer_id])
            
                # load data files of interest
                layer_data = load_and_sort(data_dir, [epoch, layer_id], downsample=True)
            
                conf_ordered_layer_data = apply_order(layer_data, orders)
                del layer_data
                
                target_dir = TARGET_DIR+SUB_TARGET_DIRS[j%len(SUB_TARGET_DIRS)]+'/'
                j += 1
                if not os.path.exists(target_dir):
                    print('making dir: {}'.format(target_dir))
                    os.makedirs(target_dir)

                new_file = layer_paths[0].replace('-step_0-', '-')
                new_path = target_dir+new_file

                f = h5.File(new_path, 'w')
                f.create_dataset('obj_arr', data=conf_ordered_layer_data)
                f.close()
                del conf_ordered_layer_data
                print('saved data for epoch {}, layer {} to {}'.format(epoch, layer_id, new_path)) 

def sort(paths):
    paths = np.array(paths)
    order = np.argsort([int(path.split('-')[2].split('_')[1]) for path in paths])
    return paths[order]

def match_strings(strings, path):
    return all([string in path for string in strings])

def get_paths(target_dir, strings):
    paths = os.listdir(target_dir)
    return sort([path for path in paths if match_strings(strings, path)])

def load_paths(target_dir, paths):
    data = []
    for path in paths:
        datum = np.array(h5.File(target_dir+path)['obj_arr'])
        data.append(datum)
    return np.concatenate(data)

def label_sort(label_data, layer_data):
    return np.array([layer_data[np.squeeze(label_data==label)] for label in np.unique(label_data).astype(int)])

def load_and_sort(data_dir, strings, downsample=False):
    paths = get_paths(data_dir, strings)
    label_paths = get_paths(data_dir, ['-ep_300-', 'labels'])
    data = load_paths(data_dir, paths)
    if downsample:
        data = ds.downsample(data)

    label_data = load_paths(data_dir, label_paths)
    data_sorted = label_sort(label_data, data)
    print('Loaded and sorted {} data. Shape:{}'.format(strings, data_sorted.shape))
    return data_sorted

def compose_img(x):
    return np.transpose(x.reshape([3,32,32]).astype(float), [1,2,0])

def softmax(X, beta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats. 
    beta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the 
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter, 
    y = y * float(beta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    
    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p

def get_rand_order(layer_data):
    np.random.seed(seed)
    indices = np.arange(layer_data.shape[1])
    rand_indices = np.random.permutation(indices)
    stacked_rand_indices = np.tile(rand_indices,(layer_data.shape[0],1))
    return stacked_rand_indices

def get_conf_order(layer_data, beta):
    confs = np.zeros(layer_data.shape[:2])
    orders = np.zeros(layer_data.shape[:2]).astype(int)
    for p in range(layer_data.shape[0]):
        xsoft = softmax(layer_data[p,:,:], beta=beta, axis=1)
        confs[p,:] = np.sort(xsoft[:,p])[::-1]
        orders[p,:] = np.argsort(xsoft[:,p])[::-1].astype(int)
    return confs, orders

def apply_order(layer_data, orders):
    ordered_layer_data = np.array([layer_data[p,orders[p],:] for p in range(layer_data.shape[0])])
    assert layer_data.shape == ordered_layer_data.shape
    print('reordered data!')
    return ordered_layer_data

class Downsampler(object):
    def __init__(self, **kwargs):
        self.keys = []
        self.perms_dict = {}
        self.samples=5000

    def downsample(self, layer):
        np.random.seed(seed)
        print('original layer shape: {}'.format(layer.shape))
        fsize = layer.shape[-1]
        if fsize > self.samples:
            if fsize in self.keys:
                print('using perm for layer size: ',fsize)
                perm = self.perms_dict[fsize]
            else:
                print('creating new perm for layer size: ',fsize)
                perm = np.sort(np.random.permutation(fsize)[:self.samples])
                self.keys.append(fsize)
                self.perms_dict[fsize] = perm
            layer = layer[:,perm]
        print('new layer shape: {}'.format(layer.shape)) 
        return layer

if __name__ == '__main__':
    ds = Downsampler()
    main()
