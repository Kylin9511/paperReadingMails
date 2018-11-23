# Headline

大家好：

 <b><span style="color:green">这是2018年度第11篇Arxiv Weekly。</span> </b>

本文是 __模型研究__ 方向的文章。

# Highlight

> __自动数据增强(Auto Augment)算法，利用更高级的数据增强涨点__

# Information

### Title
_AutoAugment: Learning Augmentation Policies from Data_

### Link
https://arxiv.org/pdf/1805.09501.pdf

### Code
https://github.com/tensorflow/models/tree/master/
research/autoaugment

### Source

- 谷歌（Google Brain）<span style="color:gray">[Quoc V.Le 组]</span>

# Introduction

本文中试图进一步探究数据增强来提升分类网络的性能。文章提出了一种简明的数据增强流程，称为AutoAugment，下文简称AA。AA能够自动地搜索出针对数据集的最佳增强策略，相比于传统的经验主义增强更能提升网络在测试集上的表现。

AA的关键在于合理构建一个数据增强的待搜索空间，并且直接依据数据集的某些特性来给出某个候选增强策略的优劣。

在AA的实际实现中，我们设计了一个包含很多sub-policies的数据增强空间，并对每个mini-batch的数据随机挑选一种sub-policy进行增强。

<span style="color:blue">每个sub-policy包含两个能被连续执行的子操作，所有操作均为某种意义上的图像增强算法（包括平移、旋转、切割、对称）等与一个概率函数的组合。这个函数决定这个图像增强算法以什么概率和什么强度被执行。

本文的方法在CIFAR、SVHN和ImageNet上都进行了实验，均取得了很好的效果。其中在ImageNet上获得了83.54%的t1；在CIFAR10上获得了98.52%的acc。

文章也通过实验证明，学习到的数据增强策略本身具备一定的可迁移性，能够不加finetune地应用在别的一些场景下。

Wide-ResNet, Shake-Shake和ShakeDrop backbone附加AA的版本已经开源在Codes中供大家参考查阅。

# Keys

AA的核心非常容易解释，也即在通用的数据增强之余，想办法进一步进行一些能捞到好处的额外增强。

要简明扼要地捞到好处，就需要一个简明扼要的机制。<span style="color:blue">文中最后给出的机制是：对于每个mini-batch，从一个sub-policy集合中随机挑选一种进行额外的增强。</span>ImageNet、CIFAR和SVHN如下：

<center><img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/011/011_03.png?raw=true" width = "80%" /></center>

<center><img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/011/011_01.png?raw=true" width = "80%" /></center>

<center><img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/011/011_02.png?raw=true" width = "80%" /></center>

__实际上，这是本文最有实际价值的部分。实用主义地讲，文章的讲解已经结束了__。

但是我们总会关心这个结果是怎么得到的，答案就是RL暴力搜索，并不很令人激动。

实际上，作者考虑如下的16种图像增强算法作为操作全空间,并且考虑10种不同的执行强度和11个不同的执行概率。

<center><img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/011/011_04.png?raw=true" width = "80%" /></center>

假如每次都从所有策略中找出top10 best构成一个集合来随机选择，我们的问题就具体到了如何在$(16 * 10 * 11)^{10} = 2.9 * 10^{32}$的集合中搜索出最好的10个。

然后很自然的，从大的数据集中采样一个子集、给一个小的toy model就能利用AA进行训练，得到reward，最终给出搜索的10个结果。

实际上我们能从RL得到一系列的结果，每个结果是5个sub-policies构成的sub-policies set（包含10个操作）。为了提供一定的鲁棒性，作者最终把reward最高的5个set concatenate起来，就得到了上面的AA操作空间。

# Results

### 1.在CIFAR上的效果

<center><img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/011/011_05.png?raw=true" width = "80%" /></center>

基于baseline和cutout已经很低的error，添加AA后能够进一步压低之，还是非常舒服的。

### 2.在ImageNet上的效果

<center><img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/011/011_06.png?raw=true" width = "80%" /></center>

可以看到，纯粹通过数据增强提升了Resnet50上的一个点，可以说很开心的。

### 3.在SVHN上的效果

<center><img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/011/011_07.png?raw=true" width = "80%" /></center>

# Insights

提出了RL结合数据增强的算法，并给出了ImageNet和CIFAR上的数据增强搜索结果。

如果能够复现出文章中的结果，非常make sense，属于免费的午餐。
