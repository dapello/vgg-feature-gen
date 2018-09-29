import os
import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

def load_data(mani_dir):
    paths = np.sort(np.array(os.listdir(mani_dir)))
    data = np.array([loadmat(mani_dir+path) for path in paths])
    return paths, data

def format_str(s0):
    s1 = ''
    counter = 0
    for l in s0:
        if l == '_':
            counter += 1
        if l == '_' and counter in [2,4,6] :
            l = '-'
        s1+=l
    return s1

def get_layer_type(path, types):
    for t in types:
        if t in path:
            return t

def frame_constructor(paths, data, key, tag=None, mean=False):
#     layers = [path.split('-')[1] for path in paths]
    perm_seed = [path.split('_')[0] for path in paths]
    ft_size = [path.split('_')[1] for path in paths]
    lnum = [path.split('_')[4] for path in paths]
    coding = [path.split('_')[3] for path in paths]
#     layers = np.array([format_str(path).split('_')[3] for path in paths])
#     epochs = np.array([int(format_str(path).split('_')[1].split('-')[1]) for path in paths])
    image_set = np.array([path.split('_')[0] for path in paths])
    
    data_vec = np.array([np.squeeze(datum[key]) for datum in data])
    if mean:
        data_vec = np.mean(data_vec,axis=1)
    data_vec = np.atleast_2d(data_vec)    
    print(data_vec.shape)
    if data_vec.shape[0]<data_vec.shape[1]:
        data_vec = data_vec.T
        
    df = pd.DataFrame(
        columns=[
            'path', 
            'image set', 
            'layer number',
            'coding',
            'perm seed', 
            'feature size', 
            'value', 
            'measure',
            'tag'
        ], 
        data=np.array([
            np.repeat([paths],data_vec.shape[-1],axis=0).T.reshape(-1),
            np.repeat([image_set],data_vec.shape[-1],axis=0).T.reshape(-1),
            np.repeat([lnum],data_vec.shape[-1],axis=0).T.reshape(-1),
            np.repeat([coding],data_vec.shape[-1],axis=0).T.reshape(-1),
            np.repeat([perm_seed],data_vec.shape[-1],axis=0).T.reshape(-1),
            np.repeat([ft_size],data_vec.shape[-1],axis=0).T.reshape(-1),
            data_vec.reshape(-1),
            np.repeat(key,data_vec.size),
            np.repeat(tag,data_vec.size)
        ]).T
    )
    
    types = ['MaxPool2d', 'Conv2d', 'ReLU', 'Linear', 'BatchNorm2d']
    df['type'] = df.path.apply(lambda x: get_layer_type(x, types))
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df['perm seed'] = pd.to_numeric(df['perm seed'], errors='coerce')
    df['feature size'] = pd.to_numeric(df['feature size'], errors='coerce')
    df['layer number'] = pd.to_numeric(df['layer number'], errors='coerce')
    return df

def multi_frame_constructor(mani_dirs, tags, measures):
    df = None
    for i, mani_dir in enumerate(mani_dirs):
        paths, data = load_data(mani_dir)
        for measure in measures:
            mean = True
            if measure == "CCcorr": mean = False
            if type(df) == type(None):
                df = frame_constructor(paths, data, measure, tag=tags[i], mean=mean)
            else:
                df = df.append(frame_constructor(paths, data, measure, tag=tags[i], mean=mean))
    return df
  
def make_contiguous(a):
    return np.arange(len(a))
    
def display(df, measure, coding, title, dims=(12,7)):
    unique_tags = np.unique(df.tag.values)
    data = df[
        (df['measure']==measure) &
        (df['coding']==coding)
    ].sort_values(by=['layer number']).copy()
    
    for unique_tag in unique_tags:
        contiguous_layer_num = make_contiguous(data[data['tag']==unique_tag]['layer number'].values)
        data.loc[data['tag']==unique_tag, 'layer number'] = contiguous_layer_num
    
    # re sort by layer number, as everything will be shifted if one set was not contiguous
    data = data.sort_values(by=['layer number'])
    
    fig, ax = plt.subplots(figsize=dims)

    ax = sns.scatterplot(x="layer number", 
                         y="value", 
                         ax=ax,
                         hue="tag",
                         data=data)
    ax.set_title(title)
    ax.set_ylabel('mean {}'.format(measure.replace('_vec','')))
    ax.set_xticks(ticks=range(len(data.type)/len(unique_tags)))
    ax.set_xticklabels(data.type.values[::len(unique_tags)],rotation=90)
    ax.set_xlabel('layer type')
    return data

def get_losses(log, epochs=300):
    f = open(log)
    val_loss = 100-np.array([float(line.split(' ')[3]) for line in f if " * Prec@1" in line])
    f = open(log)
    train_loss = np.array([float(line.split(" ")[8]) for line in f if ("Prec@1" in line) & ("Epoch" in line)])
    step_num = len(train_loss)/epochs
    train_loss = 100-np.array([train_loss[i*step_num:i*step_num+step_num].mean() for i in range(epochs)])
    
    return train_loss, val_loss

def plot_losses(log):
    train_loss, val_loss = get_losses(log)
    ax = pd.DataFrame(columns=['training error', 'validation error'], data=np.array([train_loss, val_loss]).T).plot()
    ax.set_xlabel('Epoch number')
    ax.set_ylabel('Error (%)')
    ax.set_title('Training curves')
