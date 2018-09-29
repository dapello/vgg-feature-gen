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

for model in vgg16 vgg16_bn
do
    tag=drop_25
    log=${tag}-${model}-log
    save_dir=${tag}-${model}-save
    feature_dir=${tag}-${model}-features
    echo "python main.py -a=$model --save-dir=$save_dir  --feature-dir=$feature_dir  -gpu  |& tee -a $log"
    python main.py -a=$model --save-dir=$save_dir  --feature-dir=$feature_dir  -gpu  --dropout=0.25  |& tee -a $log
    python main.py -a=$model --resume=${save_dir}/checkpoint_300.tar -gpu -s --start-epoch=300  --feature-dir=$feature_dir
    python process.py $feature_dir random 
done
