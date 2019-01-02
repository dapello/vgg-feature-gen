#!/bin/bash
#SBATCH -n 10                # Number of cores
#SBATCH -t 10000             # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH --gres=gpu:titan-x:1
#SBATCH --mem=30G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o slurm_logs/train_sample_process_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e slurm_logs/train_sample_process_%j.err  # File to which STDERR will be written, %j inserts jobid

module load openmind/anaconda
source activate p27torch

cd /om2/user/dapello/code/vgg-feature-gen/feature_gen

for ((i = 0; i <= 5; i++)); 
#for i in 0 
do
    for ((j = 2; j <= 13; j++)); 
    #for j in 1
    #for classes in 50 
    #for model in vgg1_mp_sc
    do
        model=vgg${j}
        dataset=CIFAR100
        classes=50
        drop=0
        epochs=50
        tag=seed_${i}-drop_${drop}-dataset_${dataset}-classes_${classes}
        folder=network_data
        log=${folder}/${tag}-arch_${model}-log
        save_dir=${folder}/${tag}-arch_${model}-save
        feature_dir=${folder}/exp_transfer-${tag}-arch_${model}-features
        echo "python main.py -a=$model --save-dir=$save_dir  --feature-dir=$feature_dir  -gpu  |& tee -a $log"
        
        # sample at select epochs
        for epoch in 0 1 2 4 8 16 32 50
        do
            python main.py --archclass=vgg_s -a=${model} --resume=${save_dir}/checkpoint_${epoch}.tar  -gpu -s --start-epoch=${epoch}  --feature-dir=$feature_dir -d=${dataset} --classes=${classes} --invert=1
            # python main.py -a=vgg16  --pretrained  -gpu -s --start-epoch=30  --feature-dir=tfeature_dir  -b=16  -d=HvM64
        done
        
        # process resultant feature dirs
        python process.py $feature_dir betasoftmax $epochs ${i} 100
        python process.py $feature_dir reversebetasoftmax $epochs ${i} 100
        python process.py $feature_dir random $epochs ${i} 100
        rm -r $feature_dir
    done
done
