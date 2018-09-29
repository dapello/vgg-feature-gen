#!/bin/bash

#for model in vgg11 vgg11_bn vgg13 vgg13_bn vgg16 vgg16_bn vgg19 vgg19_bn
for model in vgg16
do
    echo "python main.py -a=$model --resume=drop_0_save_vgg16_bn/checkpoint_299.tar -gpu -s --start_epoch=300  --feature-dir=drop_0_feature_$model"
    python main.py -a=$model --resume=drop_0_save_vgg16/checkpoint_299.tar -gpu -s --start-epoch=300  --feature-dir=drop_0_feature_$model
    python main.py -a=$model --resume=drop_25_save_vgg16/checkpoint_299.tar -gpu -s --start-epoch=300  --feature-dir=drop_25_feature_$model
    python main.py -a=$model --resume=drop_75_save_vgg16/checkpoint_299.tar -gpu -s --start-epoch=300  --feature-dir=drop_75_feature_$model
    # python main.py -a=$model --feature-dir=feature_rand_$model -s -gpu --start-epoch=0
done
