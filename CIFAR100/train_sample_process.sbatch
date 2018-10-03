#!/bin/bash
#SBATCH -n 10                # Number of cores
#SBATCH -t 10000             # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH --gres=gpu:1
#SBATCH --mem=30G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o train_sample_process_CIFAR100_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e train_sample_process_CIFAR100_%j.err  # File to which STDERR will be written, %j inserts jobid

module load openmind/anaconda
source activate p27torch

cd /om2/user/dapello/code/vgg-feature-gen/CIFAR100

for ((i = 0; i <= 5; i++)); do
    for model in vgg1_sc vgg1_sc_bn
    do
        drop=0
        tag=drop_${drop}
        folder=network_data
        log=${folder}/${tag}-${model}-log
        save_dir=${folder}/${tag}-${model}-save
        feature_dir=${folder}/${tag}-${model}-features
        echo "python main.py -a=$model --save-dir=$save_dir  --feature-dir=$feature_dir  -gpu  |& tee -a $log"
        
        # train
        # python main.py -a=$model --save-dir=$save_dir  --feature-dir=$feature_dir  --dropout=0.${drop}  --seed=${seed}  -gpu  |& tee -a $log
        
        # sample at select epochs
        for epoch in 0 1 10 50 100 300
        do
            python main.py -a=$model  --resume=${save_dir}/checkpoint_${epoch}.tar  -gpu -s --start-epoch=${epoch}  --feature-dir=$feature_dir
        done
        
        # process resultant feature dirs
        # python process.py $feature_dir betasoftmax 
        # python process.py $feature_dir random 
    done
done
