#!/bin/bash

for seed in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
do
    for model in vgg1_sc
    do
        drop=0
        tag=seed_${seed}-drop_${drop}
        log=${tag}-${model}-log
        save_dir=${tag}-${model}-save
        feature_dir=${tag}-${model}-features
        echo "python main.py -a=$model --save-dir=$save_dir  --feature-dir=$feature_dir  |& tee -a $log"
        python main.py -a=$model --save-dir=$save_dir  --feature-dir=$feature_dir  --dropout=0.${drop}  --seed=${seed}  -gpu  |& tee -a $log
    done
done
#for model in vgg11 vgg11_bn vgg13 vgg13_bn vgg16 vgg16_bn vgg19 vgg19_bn
#do
#    echo "python main.py  --arch=$model --half --save-dir=save_half_$model |& tee -a log_half_$model"
#    python main.py  --arch=$model --half --save-dir=save_half_$model |& tee -a log_half_$model
#done
