#!/bin/bash
#SBATCH -n 10                # Number of cores
#SBATCH -t 10000             # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH --gres=gpu:titan-x:1
#SBATCH --mem=30G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o slurm_logs/fc_train_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e slurm_logs/fc_train_%j.err  # File to which STDERR will be written, %j inserts jobid

module load openmind/anaconda
source activate p27torch

cd /om2/user/dapello/code/vgg-feature-gen/feature_gen

#for ((i = 0; i <= 3; i++)); 
for i in 0 
do
    for model in vgg4 vgg8 vgg16
    do
        dataset=CIFAR10
        classes=10
        drop=0
        epochs=50
        tag=seed_${i}-drop_${drop}-dataset_${dataset}-classes_${classes}
        folder=network_data
        log=${folder}/${tag}-arch_${model}-log
        save_dir=${folder}/${tag}-arch_${model}-save
        feature_dir=${folder}/${tag}-arch_${model}-features
        
        touch $log
        cat train_sample_process.sbatch >& $log
        
        # train
        python main.py --seed=${i} --archclass=vgg -a=${model} \
            -d=${dataset}  --classes=$classes  \
            --optimizer='SGD' --wd=0 --epochs=${epochs} \
            --save-dir=$save_dir  --feature-dir=$feature_dir  \
            -gpu \
            |& tee -a $log   
    done
done
