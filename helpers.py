import os
import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import seaborn as sns

def match_strings(strings, path, any_or_all='any'):
    if any_or_all == 'any':
        return any([string in path for string in strings])
    elif any_or_all == 'all':
        return all([string in path for string in strings])

def fix(datum, path):
    ## hopefully vestigial code used previously used in load_data to fix a massive diagonal EV matrix. 
    if datum['EV_vec'].shape[0] > 1:
        print('fixing EV_vec')
        print(datum['EV_vec'].shape)
        print(path)
        datum['EV_vec'] = np.diag(datum['EV_vec'])
        savemat(path, datum, appendmat=False)
    return datum

def load_data(mani_dir, exclude=[]):
    # takes a directory and loads all the .mat files from batch_manifold_analysis.m
    # return paths and data
    paths = np.sort(np.array(os.listdir(mani_dir)))
    if len(exclude)>0:
        paths = [path for path in paths if not match_strings(exclude, path)] 
    data = []
    for path in paths:
        datum = loadmat(mani_dir+path)
        if 'EV_vec' in datum.keys():
            #datum = fix(datum, mani_dir+path)
            pass
        data.append(datum)
        
    return paths, np.array(data)

#def load_data(mani_dir, exclude=[]):
#    # takes a directory and loads all the .mat files from batch_manifold_analysis.m
#    # return paths and data
#    paths = np.sort(np.array(os.listdir(mani_dir)))
#    if len(exclude)>0:
#        paths = [path for path in paths if not match_strings(exclude, path)] 
#    data = np.array([loadmat(mani_dir+path) for path in paths])
#    return paths, data

def get_layer_type(path, types):
    for t in types:
        if t in path:
            return t

def mi_outliers(data_vec):
    for i in range(len(data_vec)):
        row_mean = data_vec[i].mean()
        row_std = data_vec[i].std()    
        for j in range(len(data_vec[i])):
            if np.abs(data_vec[i][j] - row_mean) > row_std*2:
                data_vec[i][j] = row_mean
    return data_vec

def fill_input_features(df, input_features=3072):
    df['featnum'] = df['featnum'].fillna(value=input_features)

def frame_constructor(paths, data, key, tag=None, mean=False, verbose=False, rm_outliers=True):
    perm_seed = [catch(path, 'seed') for path in paths]
    featnum = [catch(path, 'featnum') for path in paths]
    acc = [catch(path, 'acc') for path in paths]
    arch = [catch(path, 'arch') for path in paths]
    RP = [catch(path, 'RP') for path in paths]
    lnum = [path.split('-')[3].split('_')[1] for path in paths]
    coding = [path.split('-')[3].split('_')[0] for path in paths]
    epochs = np.array([int(path.split('-')[1].split('_')[1]) for path in paths])
    image_set = np.array([path.split('-')[0] for path in paths])
    data_vec = np.array([np.squeeze(datum[key]) for datum in data])
    
    if mean:
        if rm_outliers:
            mi_outliers(data_vec)
    
        data_vec = np.mean(data_vec,axis=1)

    data_vec = np.atleast_2d(data_vec)    
    if verbose:
        print('data_vec.shape: ', data_vec.shape)
    if data_vec.shape[0]<data_vec.shape[1]:
        data_vec = data_vec.T
        
    df = pd.DataFrame(
        columns=[
            'path', 
            'imageset',
            'epoch',
            'layer number',
            'coding',
            'seed', 
            'featnum', 
            'acc', 
            'arch', 
            'RP', 
            'value', 
            'measure',
            'tag'
        ], 
        data=np.array([
            np.repeat([paths],data_vec.shape[-1],axis=0).T.reshape(-1),
            np.repeat([image_set],data_vec.shape[-1],axis=0).T.reshape(-1),
            np.repeat([epochs],data_vec.shape[-1],axis=0).T.reshape(-1),
            np.repeat([lnum],data_vec.shape[-1],axis=0).T.reshape(-1),
            np.repeat([coding],data_vec.shape[-1],axis=0).T.reshape(-1),
            np.repeat([perm_seed],data_vec.shape[-1],axis=0).T.reshape(-1),
            np.repeat([featnum],data_vec.shape[-1],axis=0).T.reshape(-1),
            np.repeat([acc],data_vec.shape[-1],axis=0).T.reshape(-1),
            np.repeat([arch],data_vec.shape[-1],axis=0).T.reshape(-1),
            np.repeat([RP],data_vec.shape[-1],axis=0).T.reshape(-1),
            data_vec.reshape(-1),
            np.repeat(key,data_vec.size),
            np.repeat(tag,data_vec.size)
        ]).T
    )
    types = ['input', 'AvgPool2d', 'MaxPool2d', 'Conv2d', 'ReLU', 'Sequential', 'Linear', 'BatchNorm2d', 'Softmax']
    df['type'] = df.path.apply(lambda x: get_layer_type(x, types))
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df['acc'] = pd.to_numeric(df['acc'], errors='coerce')
    df['epoch'] = pd.to_numeric(df['epoch'], errors='coerce')
    df['seed'] = pd.to_numeric(df['seed'], errors='coerce')
    df['featnum'] = pd.to_numeric(df['featnum'], errors='coerce')
    df['layer number'] = pd.to_numeric(df['layer number'], errors='coerce')
    df['layer number og'] = pd.to_numeric(df['layer number'], errors='coerce')
    df['layer name'] = df['coding'].values + '.'+  df['layer number og'].astype('str').values
    df.loc[df['coding']=='features', 'layer number'] += 1
    df.loc[df['coding']=='classifier', 'layer number'] = df.loc[
        df['coding']=='classifier', 'layer number'] + df[
        (df['coding']=='features') & (df['imageset']=='train') & (df['epoch']==epochs.max()) # this breaks if epoch 1 is not in the mix..
    ].shape[0] + 1   
    df.round(2) 
    return df

def compile_info(mani_dir, path):
    mani_dir.replace('manifold_', '-')
    info = path.replace('.h5', '')
    info += '-seed_'+catch(mani_dir, 'seed', ind=1)
    info += '-arch_'+catch(mani_dir, 'arch')
    info += '-RP_'+catch(mani_dir, 'RP')
    
    return info

def multi_frame_constructor(mani_dirs, tags, measures, exclude=[], verbose=False):
    df = None
    for i, mani_dir in enumerate(mani_dirs):
        paths, data = load_data(mani_dir, exclude=exclude)
        paths_info = [compile_info(mani_dir, path) for path in paths]
        for measure in measures:
            mean = True
            single = ['CCcorr', 'pr', 'K0', 'S_vec', 'D_pr', 'D_expvar', 'D_feature_ds', 'asim0_g', 'asim0_m', 'Nc0_g', 'Nc0_m',] 
            if measure in single: mean = False
            if type(df) == type(None):
                df = frame_constructor(paths_info, data, measure, tag=tags[i], mean=mean, verbose=verbose)
            else:
                df = df.append(frame_constructor(paths_info, data, measure, tag=tags[i], mean=mean))
    return df
  
def make_contiguous(a):
    return np.arange(len(a))
    
def display(df, x, y, measure, coding, title, opts={'sortby':[], 'hue':'tag', 'fix_legend':False, 'dims': (12,7)}):
    unique_tags = np.unique(df.tag.values)
    data = df[
        (df['measure']==measure)
#         &(df['coding']==coding)
    ].sort_values(by=['layer number']).copy()
    
    for unique_tag in unique_tags:
        contiguous_layer_num = make_contiguous(data[data['tag']==unique_tag]['layer number'].values)
        data.loc[data['tag']==unique_tag, 'layer number'] = contiguous_layer_num
    
    # re sort by layer number, as everything will be shifted if one set was not contiguous
    data = data.sort_values(by=['layer number'])
    layer_types = data['type'].values[::len(unique_tags)]
    layer_features = data['featnum'].values[::len(unique_tags)]
    xlabels = [layer_types[i]+' ({})'.format(layer_features[i]) for i in range(len(layer_types))]

    if len(opts['sortby'])>0:
        data = data.sort_values(by=opts['sortby'])
    
    fig, ax = plt.subplots(figsize=opts['dims'])
    p = sns.cubehelix_palette(len(unique_tags), start=1, rot=0, dark=.20, light=.80)
    sns.set_palette('Reds')
    with sns.color_palette("PuBuGn_d"):
        ax = sns.scatterplot(x=x, 
                         y=y, 
#                          units="tag",
                         # size=opts['hue'],
                         sizes=(150,150),
                         ax=ax,
                         hue=opts['hue'],
                         palette=p,
                         legend='brief',
                         data=data)

    if opts['fix_legend']:
        handles, labels = ax.get_legend_handles_labels()
        l = plt.legend(handles[0:1+len(unique_tags)], labels[0:1+len(unique_tags)], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    else:
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    ax.set_title(title)
    ax.set_ylabel(names[measure])
    ax.set_xticks(ticks=range(len(data.type)/len(unique_tags)))
    ax.set_xticklabels(xlabels,rotation=90)
    ax.set_xlabel('layer type')
    return data, ax

names = {
    'D_pr': 'Total Data PR',
    'D_expvar': 'Total Data EV Dimension',
    'CCcorr': 'Mean Manifold Center Correlation',
    'D_S_vec': 'Mean Category EV Dimension',
    'Dpr_S_vec': 'Mean Category PR',
    'R_S_vec': 'Mean Category Radius',
    'D_M_vec': 'Mean Dichotomy Dimension',
    'R_M_vec': 'Mean Dichotomy Radius',
    'a_Mfull_vec': 'Mean Dichotomy Capacity',
}

def get_losses(log, epochs=300, accuracy=False):
    f = open(log)
    val_loss = np.array([float(line.split(' ')[3]) for line in f if " * Prec@1" in line])
    f = open(log)
    train_loss = np.array([float(line.split(" ")[8]) for line in f if ("Prec@1" in line) & ("Epoch" in line)])
    step_num = len(train_loss)/epochs
    train_loss = np.array([train_loss[i*step_num:i*step_num+step_num].mean() for i in range(epochs)])
    
    if not accuracy:
        val_loss = 100-val_loss
        train_loss = 100-train_loss
    
    return train_loss, val_loss

def add_loss(df, loss_df):
    df['loss'] = df.apply(lambda x : loss_df[(loss_df['epoch']==x['epoch'])&
                            (loss_df['seed']==x['seed'])&
                            (loss_df['arch']==x['arch'])&
                            (loss_df['imageset']==x['imageset'])]['loss'].values[0], axis=1)

    df['trainacc'] = df.apply(lambda x : 100-loss_df[(loss_df['epoch']==x['epoch'])&
                            (loss_df['seed']==x['seed'])&
                            (loss_df['arch']==x['arch'])&
                            (loss_df['imageset']=='train')]['loss'].values[0], axis=1)

    df['valacc'] = df.apply(lambda x : 100-loss_df[(loss_df['epoch']==x['epoch'])&
                            (loss_df['seed']==x['seed'])&
                            (loss_df['arch']==x['arch'])&
                            (loss_df['imageset']=='val')]['loss'].values[0], axis=1)
    
    return df

def delta_data(df):
    layer_nums = np.unique(df['layer number'])
    measure_id = df.loc[df['type']=='input', 'measure'].values
    initial = np.zeros([2, measure_id.shape[0]]).astype('object')
    initial[0,:] = measure_id
    deltas = np.zeros([layer_nums.shape[0], measure_id.shape[0]]).astype('object')
    deltas[0,:] = measure_id
    for n in layer_nums:
        if n == 0:
            initial[1,:] = df.loc[df['layer number']==n, 'value'].values
        else:
            deltas[n,:] = df.loc[df['layer number']==n, 'value'].values - df.loc[df['layer number']==n-1, 'value'].values

    return deltas.T, initial.T 

def get_delta_frame(dir_template, ep, seeds=[0,10], expand_input_files=False, measures=[], skip=[], exclude=[], length=0, verbose=False):
    success = []
    for seed in range(*seeds):
        if seed in skip:
            pass
        else:
            mani_dirs = [dir_template.format(seed)]
            tags = ["seed_{}".format(seed)]

            if expand_input_files:
                expand_input(mani_dirs)

            df = multi_frame_constructor(mani_dirs, tags, measures, exclude=exclude)
            if df.shape[0] < length:
                pass
            else:
                if verbose:
                    print('try seed:', seed)

                df = add_volume(df)

                df = df[(df['imageset']=='train')&(df['epoch']==ep)]

                data, initial = delta_data(df)
                
                columns = (df.sort_values(by=['layer number'])['type']+'_'+df.sort_values(by=['layer number'])['layer number'].astype('str')).unique()
                columns[0] = 'measure'

                #print(columns)
                #print(data)
                if seed == seeds[0]:
                    delta_df = pd.DataFrame(columns=columns, data=data)
                    initial_df = pd.DataFrame(columns=['measure','initial'], data=initial)
                else:
                    delta_df = delta_df.append(pd.DataFrame(columns=columns, data=data))
                    initial_df = initial_df.append(pd.DataFrame(columns=['measure','initial'], data=initial))

                if verbose:
                    print('success')
                    
                success.append(seed)
                
    print('success for seeds:', success)

    return delta_df, initial_df

def plot_losses(log):
    train_loss, val_loss = get_losses(log)
    ax = pd.DataFrame(columns=['training error', 'validation error'], data=np.array([train_loss, val_loss]).T).plot()
    ax.set_xlabel('Epoch number')
    ax.set_ylabel('Error (%)')
    ax.set_title('Training curves')
   
   
def delta_plot(delta_df, x, y, name, minmax=True, hline=[-0.5,0.5], vline=[-3,3]):
    xy_df = delta_df[delta_df['measure']==x].melt('measure')
    y_df = delta_df[delta_df['measure']==y].melt('measure')
    xy_df['value2'] = y_df['value'].values
    
    xy_df['type'] = xy_df['variable'].apply(lambda x : x.split('_')[0])
    xy_df['depth'] = xy_df['variable'].apply(lambda x : float(x.split('_')[1])).astype('float')
    xy_df['depth'] = xy_df['variable'].apply(lambda x : float(x.split('_')[1])).astype('float')
    
    sns.reset_defaults()

#     unique_tags = xy_df['variable'].unique()
#     p = sns.cubehelix_palette(len(unique_tags), light=.8, start=.5, rot=-.75)
#     ax = sns.scatterplot(x='value', y='value2', hue='variable', style='type', palette=p, data=xy_df)

    ax = sns.scatterplot(x='value', y='value2', hue='depth', style='type', legend='brief', data=xy_df)   
    
    ax.set_xlabel('delta '+x.replace('_vec',''))
    ax.set_ylabel('delta '+y.replace('_vec',''))
    if minmax:
        ax.hlines(0,xy_df['value'].min()-.01,xy_df['value'].max()+.01)
        ax.set_ylim(xy_df['value'].min()-.01,xy_df['value'].max()+.01)
        ax.vlines(0,xy_df['value2'].min()-.01,xy_df['value2'].max()+.01)
        ax.set_ylim(xy_df['value2'].min()-.01,xy_df['value2'].max()+.01)
    else:
        ax.hlines(0,*hline)
        ax.set_xlim(*hline)
        ax.vlines(0,*vline)
        ax.set_ylim(*vline)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax.set_title('{}, d{} vs d{}'.format(name, x, y)) 
    return xy_df, ax

   
#def delta_plot(delta_df, x, y, name, minmax=True, hline=[-0.5,0.5], vline=[-3,3]):
#    xy_df = delta_df[delta_df['measure']==x].melt('measure')
#    y_df = delta_df[delta_df['measure']==y].melt('measure')
#    xy_df['value2'] = y_df['value'].values
#
#    ax = sns.scatterplot(x='value', y='value2', hue='variable', data=xy_df)
#    ax.set_xlabel('delta '+x.replace('_vec',''))
#    ax.set_ylabel('delta '+y.replace('_vec',''))
#    if minmax:
#        ax.hlines(0,xy_df['value'].min()-.01,xy_df['value'].max()+.01)
#        ax.set_ylim(xy_df['value'].min()-.01,xy_df['value'].max()+.01)
#        ax.vlines(0,xy_df['value2'].min()-.01,xy_df['value2'].max()+.01)
#        ax.set_ylim(xy_df['value2'].min()-.01,xy_df['value2'].max()+.01)
#    else:
#        ax.hlines(0,*hline)
#        ax.set_xlim(*hline)
#        ax.vlines(0,*vline)
#        ax.set_ylim(*vline)
#    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#    ax.set_title('{}, d{} vs d{}'.format(name, x, y))
#    return ax
    
def catch(filepath, target, ind=1, verbose=False):
    parts = filepath.split('-')
    match = [part for part in parts if target in part]
    if len(match) == 1:
        return match[0].split('_')[ind]
    else:
        if verbose:
            print('target {} not found in filepath {}'.format(target,filepath))
        return None

def get_meta_dict(filepath,targets):
    """get_meta_dict(fs, ['seed', 'drop', 'imageset'])"""
    meta_dict = {}
    for target in targets:
        meta_dict[target] = catch(filepath,target)
        
    return meta_dict

def add_meta(df,targets):
    for target in targets:
        df[target] = df['log'].apply(lambda x : catch(x,target))
        
def add_volume(df):
    v_df = df.loc[df['measure']=='D_M_vec'].copy()
    v_df['measure'] = 'Rm*sqrt(Dm)'
    v_df['value'] = np.sqrt(df.loc[df['measure']=='D_M_vec', 'value'].values)*df.loc[df['measure']=='R_M_vec', 'value'].values
    return df.append(v_df)

def expand_input(mani_dirs):
    from shutil import copyfile 

    for md in mani_dirs:
        files = os.listdir(md)
        eps = np.unique([catch(p, 'ep_') for p in files if not match_strings(['input'], p)])
        inputs = [p for p in files if match_strings(['-input'], p)]
        template = files[0]

        for og_file in inputs:
            for ep in eps:
                og_file_path = os.path.join(md, og_file)
                dest = og_file.replace('-', '-ep_{}-'.format(ep)).replace('input', 'acc_0-layer_0-type_input-features_3072')
                dest = os.path.join(md, dest)
                copyfile(og_file_path, dest)
            os.remove(og_file_path)
