#!/bin/bash

#for model in vgg11 vgg11_bn vgg13 vgg13_bn vgg16 vgg16_bn vgg19 vgg19_bn
for model in vgg16
do
    echo "python main.py  --arch=$model  --save-dir=drop_75_save_$model  --dropout=0.75  --feature-dir=drop_75_feature_$model  -gpu  --seed=0 |& tee -a drop_75_log_$model"
    python main.py  --arch=$model  --save-dir=drop_75_save_$model  --dropout=0.75  --feature-dir=drop_75_feature_$model  -gpu  --seed=0 |& tee -a drop_75_log_$model
done

#for model in vgg11 vgg11_bn vgg13 vgg13_bn vgg16 vgg16_bn vgg19 vgg19_bn
#do
#    echo "python main.py  --arch=$model --half --save-dir=save_half_$model |& tee -a log_half_$model"
#    python main.py  --arch=$model --half --save-dir=save_half_$model |& tee -a log_half_$model
#done
