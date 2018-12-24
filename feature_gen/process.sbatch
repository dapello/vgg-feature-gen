#!/bin/bash
#SBATCH -n 10                # Number of cores
#SBATCH -t 10000             # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH --gres=gpu:titan-x:1
#SBATCH --mem=30G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o train_sample_process_MNIST_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e train_sample_process_MNIST_%j.err  # File to which STDERR will be written, %j inserts jobid

module load openmind/anaconda
source activate p27torch

cd /om2/user/dapello/code/vgg-feature-gen/CIFAR10_cmm

for ((i = 50; i <= 65; i++)); 
#for i in 0 
do
    for model in vgg1_mp_sc vgg_mp_sc_16
    do
        drop=0
        epochs=30
        tag=seed_${i}-drop_${drop}
        folder=network_data
        log=${folder}/${tag}-${model}-log
        save_dir=${folder}/${tag}-${model}-save
        feature_dir=${folder}/${tag}-${model}-features
        echo "python main.py -a=$model --save-dir=$save_dir  --feature-dir=$feature_dir  -gpu  |& tee -a $log"
        
        # train
#        python main.py -a=$model --save-dir=$save_dir  --feature-dir=$feature_dir  --dropout=0.0  --seed=${i}  -gpu  --epochs=$epochs  --optimizer='SGD'  --wd=0  |& tee -a $log
        
        # sample at select epochs
        for epoch in 30 
        do
            python main.py -a=$model  --resume=${save_dir}/checkpoint_${epoch}.tar  -gpu -s --start-epoch=${epoch}  --feature-dir=$feature_dir
        done
        
        # process resultant feature dirs
        python process.py $feature_dir betasoftmax $epochs ${i} 15 
        rm -r $feature_dir
        # python process.py $feature_dir random ${i}
    done
done
