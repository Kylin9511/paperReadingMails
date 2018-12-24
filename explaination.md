# RPT
<center>
<img src="https://github.com/luzhilin19951120/personal-docs/blob/master/RPT.png?raw=true">
</center>

__RPT(RocksPyTorch)__ is a deep learning code library based on [pytorch](https://github.com/pytorch) and [horovod](https://github.com/uber/horovod). You can use "run.sh" to set hyperparameters and start experiments.

# Overview
We provide RPT platform for those who want to start many experiments easily with pytorch. We use horovod to accelerate the training process.

One can pick a backbone or dataset, set the hyper-parameters and establish [visdom](https://github.com/facebookresearch/visdom) visualization by merely changing a few flags listed in [Flags Provided](#flags) section.

RPT support convenient model inference, hook register, training restarting from snapshot, pre-trained model loading and other useful operations.

Many valuable training tricks would be included in RPT to boost the model performacnce. They can be easily called by adding flags as well.

So far RPT supports CIFAR10 and ImageNet1k dataset for image classification pipeline. Detection pipeline would be added to RPT soon.

More backbones and tricks coming...

# Flags provided <a id="flags"></a>

- --dataset: the dataset you choose
    - "ImageNet" for ImageNet
    - "Cifar10" for CIFAR10
- --data: path to the dataset
    - "/share1/classification_data/imagenet1k/" for imageNet1k on matrix1
    - "/share5/public/classification_data/imagenet1k/" for imageNek1k on matrix5
    - "/share1/public/data" for CIFAR on matrix1
- --arch: the backbone you choose
    - you can go into the model source files or use the following codes to check all the available models.
    ``` python
    import models
    for model in models.__dict__.keys():
        print(model)
    ```
- --arch-cfg: network architecture configure
    - this is one of the key design in RPT.
    - arch-cfg would be send down to the most basic block of your model, using the form of **kwargs. _This brings large flexibility when you create new models based on the RPT platform_.
- --epochs: default=90
- --lr or --learning-rate: default=0.1
- --step-size: lr decay step list
    - <100 150> recommended for Cifar10-200epochs
    - <30 60 90> recommended for ImageNet-100epochs
- -j or --workers: set the number of data loading workers for the dataloader, default=4.
- -b or --batch-size, default=256
    - This is the mini batch size for __all GPUs instead of single GPU.__
- SGD related:
    - --momentum: default=0.9
    - --wd or --weight-decay: default=1e-4
- --print-freq: how many iterations per log, default=20
- --test-freq: how many epochs per test, default=1
- --save-freq: how many epochs per, default=1
- --save-path: path to save checkpoints, default=none
- --save-best: path to save the best checkpoint, default=none
- --seed: random seed for both python and torch, default=None
- --gpu: gpu id to use, default=None
    - it can be a list like <0,1,2,8>
- --forward-hook: register forward hook to monitor featuremaps in training
- --backward-hook: register forward hook to monitor gradients in training
- visdom related:
    - --visdom: store-true, add to enable visdom monitor
    - --visdom-clear: store-true, add to clear all you visdom envs history in certain matrix
    - --visdom-server: visdom server & port, default="http://172.16.30.201:8098"
- other store-true actions:
    - --cpu: add to disable GPU training 
    - --pretrained: add to load the pre-trained model
    - --arch-print: add to print NN architecture in the log file
    - -e or --evaluate: add to evaluate the model

# Supported Backbones

- AlexNet
- VGG
- Inception
    - Inception-SE
- ResNet
    - ResNet-SE
    - ResNet-mix
- DenseNet
- SqueezeNet

# Visualization with visdom

when you start training with flag "--visdom" given, you can simply monitor its training procedure in http://172.16.30.201:8098。Please do not change the server or the port if you don't fully understand how visdom works.

If you want to save any of the envs, simple press "save" button. You can fork the envs to rename it as well.

The envs file will be saved at "/share1/pubilc/.visdom", each env is represented by a json file.

To reload all your saved files, you can use the following command.
``` bash
python -m visdom.server -port 8099 -env_path /share1/public/.visdom  -readonly
```
And you will see all your saved file on http://172.16.30.201:8099

__Remark: This reload can only be done on matrix1, moving your json file elsewhere will be a good idea if you want to reload them flexibly.__

Right now the traing procedure monitoring and network architecture visualization of RPT is based on visdom.

There are many things not matual enough in visdom, so some of the RPT visualization usage will be updated in the future.

# Contributing

We appreciate all kinds of contributions. Feel free for bug-fixs. If you want to add new feature, it would be a good idea to discuss it with us in an issue first.

