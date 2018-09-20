# Headline

大家好：

 <b><span style="color:green">这是2018年度第2篇Arxiv Weekly。</span> </b>

本文是检测方向的文章。

# Highlight

> __利用IoU-Net提升NMS选框的性能，并利用新的方法微调选出的bbox最终提高检测性能。__

# Information

### Title
_Acquisition of Localization Confidence for Accurate Object Detection_

### Link
https://arxiv.org/pdf/1807.11590.pdf

### Source

- 北京大学（Peking University）
- 清华大学（Tsinghua University）
- 旷视科技 （Face++）
- 头条人工智能实验室（Toutiao AI Lab）

# Introduction

现代CNN目标检测器中，一般都存在 <span style="color:brown">Non-maximum suppression   (NMS)</span> 选框和bbox回归的过程。然而这里面的逻辑是有一定的问题的。 <span style="color:red">NMS操作的置信度本质上来源于classification的label，而不是localization，这不可避免地导致了一些localization性能的降退</span>，即使后续有进一步的回归。

本文提出了IoU-Net结构，能够预测每个候选框和对应的ground truth框之间的IoU。由于在选框的时候真正考虑了localization信息，IoU-Net能够优化NMS过程，给出更精准的预选框。另外，本文也提出了bbox refinement算法，来进一步提高最终框的精度。

在MSCOCO上的实验表明本文提出的方法能在经典detection pipeline上取得涨点，并且具有良好的兼容性和可迁移性。

# Keys

## 1.问题描述

-  <span style="color:red">misalignment between classification and localization accuracy. </span> <span style="color:grey">[也即NMS会舍弃更好的框，挑选烂框来回归]</span>

<center><img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/002/002_01.png?raw=true" width = "55%" /></center>

-  <span style="color:red">the absence of localization confidence makes bbox regression less interpretable.</span> <span style="color:grey"> [也即随着迭代的进行，bbox的回归结果反而恶化]</span>

<center><img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/002/002_02.png?raw=true" width = "55%" /></center>

## 2.解决方案part1：IoU-guided NMS

<center><img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/002/002_03.png?raw=true" width = "45%" /></center>

此为文中提出的新NMS算法，其核心为利用bbox对应区域和真值的IoU作为挑选maximum bbox的依据，而不是像传统的NMS一样直接用classification score来作为依据，这就弥合了misalignment between classification and localization accuracy。

另外值得一提的是，IoU-guided NMS算法中，在抑制那些和最大框交集超过阈值的框的同时，会参考它们的分类信息。如果它们的分类置信度高于我们选出的最大框，则会更新最大框的分类。<span style="color:grey">[这个操作中，分离classification和localization的意味也很重，甚至有点localization指导优化classification的意思在里面。不过个人感觉这种情况并不常见]</span>

剩下的一个关键问题是，给出一个bbox，我们怎么知道它和真值的bbox的IoU是多少……

文章中采用了如下的IoU-Net网络来解决这个问题。

<center><img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/002/002_04.png?raw=true" width = "55%" /></center>

<span style="color:blue">网络的训练数据是通过ground truth + augmentation & randomization生成的。</span>

## 3.解决方案part2：optimization based bounding box refinement

<center><img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/002/002_05.png?raw=true" width = "45%" /></center>

核心的思路在于利用IoU Net对bbox的IoU预测能力，指导后续的bbox微调。微调的时候用最简单的梯度上升优化方法，并设定两个阈值，一个用来判定收敛；一个用来判定是否已经进入localization degeneration的阶段并予以遏制。

p.s. 我们可以看到，不同的refinement方法，都是在寻找一种方式解决如下的最优化问题：

<center><img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/002/002_06.png?raw=true" width = "55%" /></center>

其中det是检测网络输出的bbox；gt是真值bbox；c是网络微调器transfomer的参数；crit是loss函数形式<span style="color:grey">[一般选取smooth-L1 distance]</span>

## 4.关于precise ROI pooling

为了能够实现3中的算法，我们需要给出一个可导的RoI Pooling算法。RoI Align算法已经比较好地解决了RoI中misalignment的问题，因此其基本想法是应该沿用的。所需要解决的是导数求解的问题。

因此<span style="color:blue">本文提出了精准RoI Pooling (PrRoI Pooling)算法，其实质就是把feature map上的点利用双线性插值 (bilinear interpolation)算法转化为连续的函数</span>，如下公式所示：

<center><img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/002/002_07.png?raw=true" width = "55%" /></center>

有了连续的feature map值，就能够把bbox以浮点数的形式直接套在feature map对应的位置上，并且进行pooling操作了。<span style="color:grey">[不过由于连续性的操作，我理解这里的PrRoI Pooling更加像一种average pooling的变体]</span>

<center><img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/002/002_12.png?raw=true" width = "55%" /></center>

PrRoI Pooling和两种经典Pooling算法的对比示意图如下：

<center><img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/002/002_08.png?raw=true" width = "55%" /></center>

# Results

所有实验均在MSCOCO上进行。

传统NMS、Soft-NMS、IoU-NMS的对比里面，<span style="color:blue">高阈值AP下IoU-NMS表现最为突出，这是因为IoU-NMS能够最大程度地优化具体的边界位置。</span>

<center><img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/002/002_09.png?raw=true" width = "50%" /></center>

将refinement操作附加在主流的pipeline上，都取得了不错的涨点。同样地，高阈值下涨点突出很多。

<center><img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/002/002_10.png?raw=true" width = "50%" /></center>

以上方法联合使用效果更佳。

<center><img src="https://github.com/luzhilin19951120/paperReadingMails/blob/master/2018/002/002_11.png?raw=true" width = "50%" /></center>

# Insights

针对检测中通用的NMS模块提出了新的改进，并且提出了在最后对bbox进行refinement操作的想法。比较好地缓解了分类和定位标准误用以及随着训练迭代bbox反而退化的问题。

另外，提出了对ROI Pooling求导数的一种思路：PrRoI Pooling。