#!/bin/bash
#SBATCH -n 10                # Number of cores
#SBATCH -t 10000             # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH --gres=gpu:1
#SBATCH --mem=30G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o slurm_logs/train_CIFAR100_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e slurm_logs/train_CIFAR100_%j.err  # File to which STDERR will be written, %j inserts jobid

module load openmind/anaconda
source activate p27torch

cd /om2/user/dapello/code/vgg-feature-gen/CIFAR100

for seed in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
do
    for model in vgg1_sc_bn
    do
        drop=0
        tag=seed_${seed}-drop_${drop}
        folder=network_data
        log=${folder}/${tag}-${model}-log
        save_dir=${folder}/${tag}-${model}-save
        feature_dir=${folder}/${tag}-${model}-features
        echo "python main.py -a=$model --save-dir=$save_dir  --feature-dir=$feature_dir  |& tee -a $log"
        python main.py -a=$model --save-dir=$save_dir  --feature-dir=$feature_dir  --dropout=0.${drop}  --seed=${seed}  -gpu  |& tee -a $log
    done
done
