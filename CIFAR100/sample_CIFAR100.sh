#!/bin/bash

#for model in vgg11 vgg11_bn vgg13 vgg13_bn vgg16 vgg16_bn vgg19 vgg19_bn
for model in vgg16
do
    echo "python main.py -a=$model --resume=save_vgg16_0/checkpoint_299.tar -s --epoch=300"
    python main.py -a=$model --resume=save_vgg16_0/checkpoint_299.tar -gpu -s --epoch=300
done

#for model in vgg11 vgg11_bn vgg13 vgg13_bn vgg16 vgg16_bn vgg19 vgg19_bn
#do
#    echo "python main.py  --arch=$model --half --save-dir=save_half_$model |& tee -a log_half_$model"
#    python main.py  --arch=$model --half --save-dir=save_half_$model |& tee -a log_half_$model
#done

