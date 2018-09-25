#!/bin/bash

#for model in vgg11 vgg11_bn vgg13 vgg13_bn vgg16 vgg16_bn vgg19 vgg19_bn
for model in vgg16
do
    echo "python main.py -a=$model --feature-dir=feature_rand_$model -gpu -s --epoch=0"
    # python main.py -a=$model --resume=save_vgg16_0/checkpoint_299.tar -gpu -s --start_epoch=300
    python main.py -a=$model --feature-dir=feature_rand_$model -s -gpu --start-epoch=0
done

#for model in vgg11 vgg11_bn vgg13 vgg13_bn vgg16 vgg16_bn vgg19 vgg19_bn
#do
#    echo "python main.py  --arch=$model --half --save-dir=save_half_$model |& tee -a log_half_$model"
#    python main.py  --arch=$model --half --save-dir=save_half_$model |& tee -a log_half_$model
#done

